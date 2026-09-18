//! A generic, lease-based exclusive lock for GPU-heavy local generation
//! work (Aivyx-Vision's mold backend). Deliberately independent of
//! `scheduler.rs`'s `Scheduler` -- no KV-cache awareness, no prefix
//! hints, no multi-slot spreading, no seeded-across-restart
//! reconciliation. Those are all specific to llama-server's own slot
//! semantics; a GPU generation job has no equivalent of any of them. See
//! `aivyx-ecosystem/docs/superpowers/specs/
//! 2026-09-18-aivyx-vision-v1-design.md` §5.
//!
//! The waiter-queueing shape (bare `oneshot::Sender<()>` wake signals,
//! every waiter re-deriving admission from scratch under a fresh lock on
//! each wake, `release`/`reap_expired` waking the *entire* queue rather
//! than stopping at the first successful send) is deliberately copied
//! from `Scheduler`'s own design -- that file's test suite documents the
//! exact bug class (Fix 3: stopping at the first successful wake strands
//! every waiter behind a dropped one) this shape exists to avoid.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::oneshot;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LeaseId(uuid::Uuid);

impl LeaseId {
    fn new() -> Self {
        LeaseId(uuid::Uuid::new_v4())
    }

    #[cfg(test)]
    pub(crate) fn new_for_test() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for LeaseId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Wraps an already-parsed UUID as a `LeaseId` -- needed by callers (e.g.
/// `server.rs`'s `POST /gpu-lock/release` handler) that must reconstruct a
/// `LeaseId` from an inbound string id, since `LeaseId`'s tuple field is
/// private and `new()` is deliberately not `pub` (only this crate's own
/// `acquire()` should ever mint a *fresh* lease id).
impl From<uuid::Uuid> for LeaseId {
    fn from(uuid: uuid::Uuid) -> Self {
        LeaseId(uuid)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GpuLockError {
    #[error("timed out waiting for the GPU lock")]
    Timeout,
    #[error("lease not found, already released, or expired")]
    UnknownLease,
}

struct Held {
    lease: LeaseId,
    acquired_at: Instant,
}

struct Inner {
    held: Option<Held>,
    waiters: VecDeque<oneshot::Sender<()>>,
}

#[derive(Clone)]
pub struct GpuLock {
    inner: Arc<Mutex<Inner>>,
    max_hold: Duration,
}

impl GpuLock {
    pub fn new(max_hold: Duration) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                held: None,
                waiters: VecDeque::new(),
            })),
            max_hold,
        }
    }

    /// Waits for and claims exclusive access, or returns `Timeout` if
    /// `queue_timeout` elapses first. Re-derives admission from scratch
    /// under a fresh lock on every wake -- never assumes ownership from
    /// the wake signal alone -- so a wake that goes to a since-dropped
    /// waiter is harmless (see module doc comment).
    pub async fn acquire(&self, queue_timeout: Duration) -> Result<LeaseId, GpuLockError> {
        let deadline = tokio::time::Instant::now() + queue_timeout;
        loop {
            let rx = {
                let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                if inner.held.is_none() {
                    let lease = LeaseId::new();
                    inner.held = Some(Held {
                        lease,
                        acquired_at: Instant::now(),
                    });
                    return Ok(lease);
                }
                let (tx, rx) = oneshot::channel();
                inner.waiters.push_back(tx);
                rx
            };
            match tokio::time::timeout_at(deadline, rx).await {
                Ok(Ok(())) => continue,
                Ok(Err(_)) | Err(_) => return Err(GpuLockError::Timeout),
            }
        }
    }

    /// Releases `lease`. Returns `UnknownLease` if it doesn't match the
    /// current holder (already released, expired via `reap_expired`, or
    /// never valid) -- the caller has nothing to release either way, but
    /// distinguishing this from success lets a caller notice a double-
    /// release bug in its own code rather than silently no-op-ing.
    pub fn release(&self, lease: LeaseId) -> Result<(), GpuLockError> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        match &inner.held {
            Some(h) if h.lease == lease => {
                inner.held = None;
                Self::wake_all(&mut inner);
                Ok(())
            }
            _ => Err(GpuLockError::UnknownLease),
        }
    }

    /// Force-releases the current holder if it has held the lock longer
    /// than `max_hold` -- the safety valve against a crashed or
    /// disconnected client that never calls `release`. No-op if nothing
    /// is held or the current hold is still within `max_hold`. Intended
    /// to be called periodically from a background task (see `main.rs`'s
    /// `reconcile_seeded_slots` for the existing analogous pattern).
    pub fn reap_expired(&self) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let expired = inner
            .held
            .as_ref()
            .is_some_and(|h| h.acquired_at.elapsed() > self.max_hold);
        if expired {
            if let Some(h) = inner.held.as_ref() {
                tracing::warn!(
                    lease = %h.lease,
                    held_for_secs = h.acquired_at.elapsed().as_secs(),
                    "gpu-lock: force-releasing a lease that exceeded max_hold -- \
                     the holder likely crashed or disconnected without releasing"
                );
            }
            inner.held = None;
            Self::wake_all(&mut inner);
        }
    }

    /// Wakes every queued waiter with a bare signal, not just the first --
    /// see the module doc comment for why stopping early strands waiters.
    /// Ignores individual send failures (a `Receiver` already dropped
    /// from timing out or being cancelled).
    fn wake_all(inner: &mut Inner) {
        while let Some(tx) = inner.waiters.pop_front() {
            let _ = tx.send(());
        }
    }

    #[cfg(test)]
    pub(crate) fn is_held(&self) -> bool {
        self.inner.lock().unwrap().held.is_some()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[tokio::test]
    async fn acquire_and_release_round_trips() {
        let lock = GpuLock::new(Duration::from_secs(60));
        let lease = lock.acquire(Duration::from_secs(1)).await.unwrap();
        lock.release(lease).unwrap();
    }

    #[tokio::test]
    async fn a_second_acquire_waits_for_the_first_release() {
        let lock = GpuLock::new(Duration::from_secs(60));
        let first = lock.acquire(Duration::from_secs(1)).await.unwrap();

        let lock2 = lock.clone();
        let waiter = tokio::spawn(async move { lock2.acquire(Duration::from_secs(5)).await });

        tokio::time::sleep(Duration::from_millis(50)).await;
        lock.release(first).unwrap();

        let second = waiter.await.unwrap().unwrap();
        lock.release(second).unwrap();
    }

    #[tokio::test]
    async fn acquire_times_out_clearly_when_never_released() {
        let lock = GpuLock::new(Duration::from_secs(60));
        let _held = lock.acquire(Duration::from_secs(1)).await.unwrap();

        let result = lock.acquire(Duration::from_millis(50)).await;
        assert!(matches!(result, Err(GpuLockError::Timeout)));
    }

    #[tokio::test]
    async fn releasing_an_unknown_or_already_released_lease_is_a_clear_error() {
        let lock = GpuLock::new(Duration::from_secs(60));
        let lease = lock.acquire(Duration::from_secs(1)).await.unwrap();
        lock.release(lease).unwrap();

        // Releasing the same lease twice must not panic or silently succeed --
        // the second caller has no lock to release.
        let result = lock.release(lease);
        assert!(matches!(result, Err(GpuLockError::UnknownLease)));
    }

    #[tokio::test]
    async fn release_rejects_a_lease_that_does_not_match_the_current_holder() {
        let lock = GpuLock::new(Duration::from_secs(60));
        let real_holder = lock.acquire(Duration::from_secs(1)).await.unwrap();

        // A foreign/stale lease id -- not the one the current holder was
        // actually issued -- must not be able to release the real holder's
        // lock. This is the single defining property of a lease-based lock:
        // proven here by mutation (temporarily loosening release()'s guard
        // to `Some(_h) => ...` made this test the only one of the whole
        // suite to fail, confirming it's load-bearing for this property).
        let foreign_lease = LeaseId::new_for_test();
        let result = lock.release(foreign_lease);
        assert!(matches!(result, Err(GpuLockError::UnknownLease)));

        // The real holder's lease must still be valid -- releasing it must
        // still succeed, proving the foreign release attempt didn't corrupt
        // state or silently free the real holder's slot.
        lock.release(real_holder).unwrap();
    }

    #[tokio::test]
    async fn release_wakes_every_queued_waiter_not_just_the_first() {
        // Same regression shape as Scheduler's own
        // `release_wakes_a_second_waiter_even_if_the_first_woken_one_is_dropped_unpolled`
        // (src/scheduler.rs) -- if `release` stopped after the first
        // successful wake `send()`, a waiter whose future is dropped
        // (e.g. an HTTP client disconnect) in the window between that send
        // succeeding and being re-polled would silently strand every waiter
        // behind it until their own timeout, even though the lock is free.
        let lock = GpuLock::new(Duration::from_secs(60));
        let held = lock.acquire(Duration::from_secs(5)).await.unwrap();

        // Waiter A: poll exactly once by hand so it registers itself in the
        // wait queue and suspends, without ever being polled again.
        let lock_a = lock.clone();
        let mut waiter_a = Box::pin(lock_a.acquire(Duration::from_secs(5)));
        assert!(futures::poll!(&mut waiter_a).is_pending());

        // Waiter B: queued for real on the runtime.
        let lock_b = lock.clone();
        let waiter_b = tokio::spawn(async move { lock_b.acquire(Duration::from_secs(5)).await });
        tokio::time::sleep(Duration::from_millis(20)).await;

        lock.release(held).unwrap();
        drop(waiter_a); // simulates A's client disconnecting before re-polling

        let admitted = tokio::time::timeout(Duration::from_secs(1), waiter_b)
            .await
            .expect("B must be admitted well before its timeout, not stranded by A's drop")
            .unwrap()
            .unwrap();
        lock.release(admitted).unwrap();
        assert!(!lock.is_held(), "lock must be free after the final release");
    }

    #[tokio::test]
    async fn reap_expired_force_releases_a_lease_past_max_hold() {
        let lock = GpuLock::new(Duration::from_millis(50));
        let _held = lock.acquire(Duration::from_secs(1)).await.unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;
        lock.reap_expired();

        // The lock must now be free -- a fresh acquire should succeed
        // immediately rather than timing out.
        let result = tokio::time::timeout(
            Duration::from_millis(50),
            lock.acquire(Duration::from_secs(1)),
        )
        .await;
        assert!(
            result.unwrap().is_ok(),
            "lock should be free after reap_expired force-released the stale lease"
        );
    }

    #[tokio::test]
    async fn reap_expired_is_a_no_op_when_nothing_is_held_or_the_hold_is_still_fresh() {
        let lock = GpuLock::new(Duration::from_secs(60));
        lock.reap_expired(); // nothing held -- must not panic

        let held = lock.acquire(Duration::from_secs(1)).await.unwrap();
        lock.reap_expired(); // held, but well within max_hold -- must not release it

        // Still held -- releasing the real lease must still succeed (proves
        // reap_expired did not already release it).
        lock.release(held).unwrap();
    }

    #[tokio::test]
    async fn a_late_release_after_reap_cannot_steal_the_next_holders_lock() {
        // The reap path is the only way a holder loses its lease without
        // knowing -- so it's the only realistic source of a stale release
        // in production. A held lease that gets reaped and then the
        // original (now-stale) holder calls release() must not be able to
        // affect whatever new lease has since been issued.
        let lock = GpuLock::new(Duration::from_millis(50));
        let stale = lock.acquire(Duration::from_secs(1)).await.unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;
        lock.reap_expired();

        let fresh = lock.acquire(Duration::from_secs(1)).await.unwrap();

        // The stale holder's late release must be rejected, not silently
        // succeed and free the fresh holder's lock out from under it.
        let result = lock.release(stale);
        assert!(matches!(result, Err(GpuLockError::UnknownLease)));
        assert!(lock.is_held(), "fresh holder's lease must still be held");

        lock.release(fresh).unwrap();
    }

    #[tokio::test]
    async fn reap_expired_wakes_a_queued_waiter() {
        // Same reasoning as Scheduler's own
        // `reconcile_seeded_idle_wakes_a_queued_waiter` -- force-freeing the
        // lock via the reap path must wake queued waiters too, not just the
        // ordinary `release()` path.
        let lock = GpuLock::new(Duration::from_millis(50));
        let _held = lock.acquire(Duration::from_secs(1)).await.unwrap();

        let lock2 = lock.clone();
        let waiter = tokio::spawn(async move { lock2.acquire(Duration::from_secs(60)).await });
        tokio::time::sleep(Duration::from_millis(20)).await;

        tokio::time::sleep(Duration::from_millis(60)).await; // let the held lease exceed max_hold
        lock.reap_expired();

        let admitted = tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .expect(
                "waiter must be woken promptly by reap_expired, not stranded until its own timeout",
            )
            .unwrap()
            .unwrap();
        lock.release(admitted).unwrap();
    }
}
