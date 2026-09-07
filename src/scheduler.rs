use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::oneshot;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SlotHint {
    pub prefix_hash: String,
    pub preferred_slot: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct SlotSnapshot {
    pub slot_id: u32,
    pub busy: bool,
    pub resident_prefix: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct Admission {
    pub slot_id: u32,
    pub cache_ready: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    #[error("timed out waiting for a free slot")]
    Timeout,
}

#[derive(Debug, Default)]
struct SlotState {
    busy: bool,
    resident_prefix: Option<String>,
    /// `true` from `seed_busy()` until the very first time this slot is
    /// successfully claimed by `try_admit_locked` -- i.e. "busy because
    /// `llama-server` reported it mid-generation across a broker restart,
    /// not (yet) because this broker instance itself admitted a request
    /// onto it." Once cleared it stays `false` forever, even across later
    /// `release()`/re-admit cycles -- a slot the broker has claimed even
    /// once is fully broker-owned from then on. See `reconcile_seeded_idle`.
    seeded_unowned: bool,
    /// LRU clock value as of this slot's most recent successful admission.
    /// Used by the fallback tier of `try_admit_locked` to pick the least-
    /// recently-used idle slot instead of always the lowest free index.
    last_used: u64,
}

struct Inner {
    slots: Vec<SlotState>,
    /// FIFO waiters for a specific slot id. Each sender carries a bare wake
    /// signal, not a slot id -- see `release()`'s doc comment for why
    /// ownership is never handed off on the sender's behalf.
    slot_waiters: HashMap<u32, VecDeque<oneshot::Sender<()>>>,
    /// FIFO waiters for "any free slot". Same bare-wake-signal contract as
    /// `slot_waiters`.
    global_waiters: VecDeque<oneshot::Sender<()>>,
    /// Monotonically increasing counter, bumped on every successful
    /// admission and stamped onto that slot's `last_used`. Powers the LRU
    /// fallback-tier choice in `try_admit_locked`.
    clock: u64,
}

/// In-memory, in-process (single `aivyx-broker` instance) slot admission
/// scheduler. Cheap to clone -- all state lives behind the shared `Arc`.
/// See `docs/superpowers/specs/2026-09-07-aivyx-broker-design.md`
/// "Scheduling & failure handling" for the admission rules this
/// implements.
#[derive(Clone)]
pub struct Scheduler {
    inner: Arc<Mutex<Inner>>,
}

impl Scheduler {
    pub fn new(num_slots: u32) -> Self {
        let slots = (0..num_slots).map(|_| SlotState::default()).collect();
        Self {
            inner: Arc::new(Mutex::new(Inner {
                slots,
                slot_waiters: HashMap::new(),
                global_waiters: VecDeque::new(),
                clock: 0,
            })),
        }
    }

    /// Marks `slot_id` busy with no known resident prefix -- used at
    /// startup to reflect a slot `llama-server` reports mid-generation
    /// across a broker restart (see spec's "Broker startup/restart").
    pub fn seed_busy(&self, slot_id: u32) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(slot) = inner.slots.get_mut(slot_id as usize) {
            slot.busy = true;
            slot.seeded_unowned = true;
        }
    }

    /// Records that `slot_id` now holds `prefix_hash`'s content -- called
    /// by the HTTP handler after a successful restore/warm-then-save, or
    /// on a fresh admission before forwarding the real request.
    pub fn mark_resident(&self, slot_id: u32, prefix_hash: String) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(slot) = inner.slots.get_mut(slot_id as usize) {
            slot.resident_prefix = Some(prefix_hash);
        }
    }

    /// Clears `slot_id`'s resident-prefix bookkeeping without touching
    /// `busy` -- called by the HTTP handler when a hint-less request is
    /// admitted onto a slot, so the occupancy table stops naming a prefix
    /// whose real KV content was just silently overwritten by that
    /// request. See Fix 7: without this, a later request for the old
    /// prefix would wrongly see `cache_ready: true` and skip restoring
    /// content that's actually gone.
    pub fn clear_resident(&self, slot_id: u32) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(slot) = inner.slots.get_mut(slot_id as usize) {
            slot.resident_prefix = None;
        }
    }

    /// Reconciles a slot that was marked busy by `seed_busy()` at startup
    /// but has since been observed idle by `llama-server`'s own `/slots`
    /// (see `main.rs`'s background reconciliation poll) -- without this,
    /// a seeded-busy slot that the broker itself never admitted a request
    /// onto (so has no `ReleaseGuard` to ever call `release()`) would stay
    /// permanently unavailable, a real capacity loss across every broker
    /// restart that catches `llama-server` mid-generation.
    ///
    /// No-op if `slot_id`'s `seeded_unowned` flag is already `false` --
    /// either it was never seeded, or (far more likely) the broker has
    /// since claimed it at least once itself via `try_admit_locked`, at
    /// which point it becomes fully broker-owned forever and this method
    /// must never touch it again. That guarantee is what makes it safe to
    /// call this against a slot the broker currently has a live guard on:
    /// a slot with a live guard was necessarily claimed by the broker, so
    /// its `seeded_unowned` flag is already `false` by construction, so
    /// this is a no-op for it -- it will never mark a live-held slot free
    /// out from under its holder.
    ///
    /// Deliberately does *not* touch `resident_prefix` (a seeded slot's
    /// prefix was already unknown -- `seed_busy` never set one). When this
    /// call actually performs the busy -> free transition, it wakes queued
    /// waiters exactly like `release()` does (see `wake_waiters_for` and
    /// Finding 2): a waiter may already be queued -- on this slot
    /// specifically, or on `global_waiters` -- for a slot that just became
    /// free by this route rather than via `release()`, and without waking
    /// them they'd sit idle until their own queue timeout even though the
    /// slot they wanted is now available.
    pub fn reconcile_seeded_idle(&self, slot_id: u32) {
        let mut inner = self.inner.lock().unwrap();
        let became_free = if let Some(slot) = inner.slots.get_mut(slot_id as usize)
            && slot.seeded_unowned
        {
            slot.busy = false;
            true
        } else {
            false
        };
        if became_free {
            Self::wake_waiters_for(&mut inner, slot_id);
        }
    }

    /// Admits `hint` (or waits for a slot if none is free), retrying under
    /// lock every time a wake signal arrives rather than trusting `release`
    /// to have handed a specific slot to this call. See `release()`'s doc
    /// comment for why: a wake signal carries no slot id and has no effect
    /// on scheduler state by itself, so it's always safe -- and necessary
    /// -- to re-derive admission from scratch after being woken.
    pub async fn admit(
        &self,
        hint: Option<SlotHint>,
        timeout: Duration,
    ) -> Result<Admission, SchedulerError> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let rx = {
                let mut inner = self.inner.lock().unwrap();
                match Self::try_admit_locked(&mut inner, &hint) {
                    Some(admission) => return Ok(admission),
                    None => {
                        let (tx, rx) = oneshot::channel();
                        // Same out-of-range filtering as `try_admit_locked`
                        // (Fix 4/Finding 1): a `preferred_slot` that's out
                        // of range for the current slot count must never be
                        // used as a `slot_waiters` key, because nothing
                        // will ever pop it -- `release()` only drains
                        // `slot_waiters` for slot ids that actually exist.
                        // Queuing there strands the caller for the full
                        // timeout even when a different slot frees up.
                        // Treat it exactly like no preference: queue on
                        // `global_waiters` instead.
                        match hint
                            .as_ref()
                            .and_then(|h| h.preferred_slot)
                            .filter(|&slot_id| (slot_id as usize) < inner.slots.len())
                        {
                            Some(slot_id) => {
                                inner.slot_waiters.entry(slot_id).or_default().push_back(tx)
                            }
                            None => inner.global_waiters.push_back(tx),
                        }
                        rx
                    }
                }
            };

            // Wait for a bare wake signal (not a slot id -- see
            // `release()`). On wake, don't assume ownership of any
            // specific slot: loop back and re-acquire the lock to retry
            // the exact same admission attempt from scratch. This is what
            // makes a wake signal that nobody ever received (because this
            // future itself got dropped before reaching here) harmless to
            // scheduler state -- it never marked any slot busy on the
            // dropped waiter's behalf, so the slot stays free for the next
            // attempt to claim.
            match tokio::time::timeout_at(deadline, rx).await {
                Ok(Ok(())) => continue,
                Ok(Err(_)) | Err(_) => return Err(SchedulerError::Timeout),
            }
        }
    }

    /// Attempts admission immediately, without waiting. Marks the chosen
    /// slot busy on success. Returns `None` if nothing is admittable now.
    fn try_admit_locked(inner: &mut Inner, hint: &Option<SlotHint>) -> Option<Admission> {
        // A `preferred_slot` that's out of range for the current slot count
        // (e.g. a stale hint from before a broker restart with fewer
        // slots) is treated exactly like no preference at all -- fall
        // through to the "any free slot" tiers below using the hint's
        // `prefix_hash` for the resident-match preference. Critically:
        // never queue on an id nothing will ever pop (see Fix 4).
        if let Some(h) = hint
            && let Some(preferred) = h.preferred_slot
            && (preferred as usize) < inner.slots.len()
        {
            let slot = &mut inner.slots[preferred as usize];
            if !slot.busy {
                slot.busy = true;
                slot.seeded_unowned = false;
                inner.clock += 1;
                slot.last_used = inner.clock;
                let cache_ready = slot.resident_prefix.as_deref() == Some(h.prefix_hash.as_str());
                return Some(Admission {
                    slot_id: preferred,
                    cache_ready,
                });
            }
            return None; // must queue for this specific slot
        }
        // No usable preference: any free slot, prefer one already resident
        // with this exact prefix, else the least-recently-used idle slot
        // (see Fix 2 -- "first free slot found" previously kept re-using
        // the same low-numbered slot instead of spreading load).
        if let Some(h) = hint
            && let Some((idx, slot)) = inner.slots.iter_mut().enumerate().find(|(_, s)| {
                !s.busy && s.resident_prefix.as_deref() == Some(h.prefix_hash.as_str())
            })
        {
            slot.busy = true;
            slot.seeded_unowned = false;
            inner.clock += 1;
            slot.last_used = inner.clock;
            return Some(Admission {
                slot_id: idx as u32,
                cache_ready: true,
            });
        }
        let (idx, _) = inner
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.busy)
            .min_by_key(|(_, s)| s.last_used)?;
        let slot = &mut inner.slots[idx];
        slot.busy = true;
        slot.seeded_unowned = false;
        inner.clock += 1;
        slot.last_used = inner.clock;
        let cache_ready = hint
            .as_ref()
            .is_some_and(|h| slot.resident_prefix.as_deref() == Some(h.prefix_hash.as_str()));
        Some(Admission {
            slot_id: idx as u32,
            cache_ready,
        })
    }

    /// The broker's own occupancy view -- used by the `GET /slots` route.
    pub fn snapshot(&self) -> Vec<SlotSnapshot> {
        let inner = self.inner.lock().unwrap();
        inner
            .slots
            .iter()
            .enumerate()
            .map(|(idx, s)| SlotSnapshot {
                slot_id: idx as u32,
                busy: s.busy,
                resident_prefix: s.resident_prefix.clone(),
            })
            .collect()
    }

    /// Frees `slot_id` and wakes every waiter queued for it (both this
    /// slot's specific queue and the global "any free slot" queue) via
    /// `wake_waiters_for` -- see that method's doc comment for why waking
    /// every queued waiter, not just the first, is deliberate. The slot is
    /// left `busy = false` here unconditionally.
    ///
    /// This is deliberate: the previous design sent the slot id itself as
    /// the oneshot payload and marked the slot busy on the sender's side,
    /// which handed ownership to a waiter before that waiter's own future
    /// was ever polled again. If that future was dropped in the window
    /// between `send()` succeeding and being polled (e.g. Axum drops the
    /// connection handler because the client disconnected), the sent
    /// value was silently discarded along with the dropped `Receiver`,
    /// but the slot stayed marked busy forever -- a permanent leak with no
    /// owner and no way to release it.
    ///
    /// With a bare `()` wake signal, `release()` never mutates `busy` on a
    /// waiter's behalf. A woken `admit()` call re-acquires the lock itself
    /// and retries admission (see `admit()`); if it was already dropped,
    /// the wake goes nowhere and the slot simply stays free, exactly as if
    /// it had never been claimed by anyone.
    pub fn release(&self, slot_id: u32) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(slot) = inner.slots.get_mut(slot_id as usize) {
            slot.busy = false;
        } else {
            return;
        }
        Self::wake_waiters_for(&mut inner, slot_id);
    }

    /// Wakes every waiter queued for `slot_id` (both this slot's specific
    /// queue and the global "any free slot" queue) with a bare wake
    /// *signal*, never a handed-off slot id. Shared by `release()` and
    /// `reconcile_seeded_idle` (Finding 2) -- both are places where a slot
    /// transitions from busy to free and any queued waiter needs a chance
    /// to retry, not just `release()`'s own callers.
    ///
    /// Waking every queued waiter, not just the first, is deliberate (Fix
    /// 3): stopping after the first *successful* `send()` leaves a narrow
    /// but real starvation window. If that first waiter's future is
    /// dropped (client disconnect) in the window between `send()`
    /// succeeding and that waiter being re-polled, no one else in the
    /// queue is ever notified even though the slot is sitting free -- the
    /// next queued waiter silently starves until its own timeout. Since
    /// the wake payload is bare `()` and every waiter re-validates by
    /// calling `try_admit_locked` itself under a fresh lock upon waking
    /// (see `admit()`), waking more than one is safe: only whichever one
    /// wins the race to re-lock first actually succeeds, the rest simply
    /// re-queue and keep waiting. This trades strict FIFO ordering for
    /// eliminating the starvation window entirely -- admission is
    /// approximately FIFO by arrival order, not strictly ordered.
    fn wake_waiters_for(inner: &mut Inner, slot_id: u32) {
        // Wake this slot's specific waiters first (rule: don't let a
        // global waiter steal a slot someone is specifically waiting on).
        // Drain the entire queue -- do NOT return after the first
        // successful send (see above for why: a wake that succeeds but is
        // never re-polled before the waiter's future is dropped would
        // otherwise strand everyone behind it). Best-effort: ignore
        // individual send failures (a `Receiver` already dropped from
        // timing out or being cancelled).
        if let Some(queue) = inner.slot_waiters.get_mut(&slot_id) {
            while let Some(tx) = queue.pop_front() {
                let _ = tx.send(());
            }
        }

        // Also wake every global "any free slot" waiter -- this slot is
        // free now and any of them may be able to claim it (or another
        // slot that became free in the meantime).
        while let Some(tx) = inner.global_waiters.pop_front() {
            let _ = tx.send(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn admits_free_slot_with_no_hint() {
        let sched = Scheduler::new(2);
        let admission = sched.admit(None, Duration::from_secs(1)).await.unwrap();
        assert!(admission.slot_id < 2);
        assert!(!admission.cache_ready);
    }

    #[tokio::test]
    async fn second_admission_for_same_prefix_on_idle_slot_is_cache_ready() {
        let sched = Scheduler::new(2);
        let first = sched.admit(None, Duration::from_secs(1)).await.unwrap();
        sched.mark_resident(first.slot_id, "prefix-a".to_string());
        sched.release(first.slot_id);

        let hint = Some(SlotHint {
            prefix_hash: "prefix-a".to_string(),
            preferred_slot: None,
        });
        let second = sched.admit(hint, Duration::from_secs(1)).await.unwrap();
        assert_eq!(second.slot_id, first.slot_id);
        assert!(second.cache_ready);
    }

    #[tokio::test]
    async fn preferred_slot_busy_queues_for_that_slot_specifically() {
        let sched = Scheduler::new(2);
        let hint_a = Some(SlotHint {
            prefix_hash: "a".to_string(),
            preferred_slot: Some(0),
        });
        let first = sched
            .admit(hint_a.clone(), Duration::from_secs(1))
            .await
            .unwrap();
        assert_eq!(first.slot_id, 0);

        // A second request preferring the same slot must wait, not steal
        // slot 1.
        let sched2 = sched.clone();
        let waiter =
            tokio::spawn(async move { sched2.admit(hint_a, Duration::from_secs(5)).await });

        tokio::time::sleep(Duration::from_millis(50)).await;
        sched.release(0);

        let admitted = waiter.await.unwrap().unwrap();
        assert_eq!(admitted.slot_id, 0);
    }

    #[tokio::test]
    async fn no_free_slots_and_no_preference_times_out_with_clear_error() {
        let sched = Scheduler::new(1);
        let _held = sched.admit(None, Duration::from_secs(1)).await.unwrap();

        let result = sched.admit(None, Duration::from_millis(50)).await;
        assert!(matches!(result, Err(SchedulerError::Timeout)));
    }

    #[tokio::test]
    async fn seed_busy_marks_slot_unavailable_until_explicitly_released() {
        let sched = Scheduler::new(1);
        sched.seed_busy(0);

        let result = sched.admit(None, Duration::from_millis(50)).await;
        assert!(matches!(result, Err(SchedulerError::Timeout)));

        sched.release(0);
        let admission = sched.admit(None, Duration::from_secs(1)).await.unwrap();
        assert_eq!(admission.slot_id, 0);
    }

    #[tokio::test]
    async fn snapshot_reflects_busy_and_resident_state() {
        let sched = Scheduler::new(2);
        let admission = sched.admit(None, Duration::from_secs(1)).await.unwrap();
        sched.mark_resident(admission.slot_id, "p".to_string());

        let snap = sched.snapshot();
        let this_slot = snap
            .iter()
            .find(|s| s.slot_id == admission.slot_id)
            .unwrap();
        assert!(this_slot.busy);
        assert_eq!(this_slot.resident_prefix.as_deref(), Some("p"));
    }

    #[tokio::test]
    async fn release_retries_timed_out_waiters_preserving_fifo_for_slot() {
        // Regression test for: release() must retry queue on failed send,
        // not strand waiters.
        //
        // When a waiter for a specific slot times out and drops its receiver
        // before release() tries to send, the previous code would return
        // immediately instead of trying the next waiter in that slot's queue.
        // This test ensures the fix tries all queued waiters in order.
        let sched = Scheduler::new(1);
        let hint = Some(SlotHint {
            prefix_hash: "test".to_string(),
            preferred_slot: Some(0),
        });

        // Admit the slot to make it busy.
        let _held = sched
            .admit(hint.clone(), Duration::from_secs(1))
            .await
            .unwrap();

        // Queue first waiter with a very short timeout so it will expire
        // before we call release().
        let sched_first = sched.clone();
        let first_waiter = tokio::spawn(async move {
            sched_first
                .admit(hint.clone(), Duration::from_millis(100))
                .await
        });

        // Give the first waiter time to queue.
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Queue second waiter with a long timeout - this one should be
        // admitted when we release().
        let sched_second = sched.clone();
        let hint2 = Some(SlotHint {
            prefix_hash: "test".to_string(),
            preferred_slot: Some(0),
        });
        let second_waiter =
            tokio::spawn(async move { sched_second.admit(hint2, Duration::from_secs(5)).await });

        // Give the second waiter time to queue (after the first).
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Wait for the first waiter to time out.
        tokio::time::sleep(Duration::from_millis(150)).await;
        let first_result = first_waiter.await.unwrap();
        assert!(matches!(first_result, Err(SchedulerError::Timeout)));

        // Now release the slot. This should try the first waiter's oneshot
        // (which will fail because it timed out), then try the second waiter's
        // (which should succeed).
        sched.release(0);

        // The second waiter should be admitted.
        let second_result = second_waiter.await.unwrap().unwrap();
        assert_eq!(second_result.slot_id, 0);
    }

    #[tokio::test]
    async fn dropping_a_woken_waiter_before_repolling_does_not_leak_the_slot() {
        // Regression test for Finding 4 (Task 4 review): release() used to
        // hand a slot off to a queued waiter unconditionally -- marking it
        // busy and sending the slot id as the oneshot payload -- before
        // that waiter's own future was ever polled again. If the future
        // was dropped in the window between `send()` succeeding and being
        // re-polled (e.g. Axum drops the connection handler because the
        // client disconnected), the sent value was silently discarded
        // along with the dropped `Receiver`, but the slot stayed marked
        // busy forever: a permanent, unrecoverable leak.
        let sched = Scheduler::new(1);
        let held = sched.admit(None, Duration::from_secs(5)).await.unwrap();

        // Queue a second admit() call for the only slot. Poll it exactly
        // once, by hand, so it runs up through registering itself as a
        // waiter and suspending on its oneshot receiver -- without ever
        // spawning it onto a runtime that might poll it again on its own.
        let sched2 = sched.clone();
        let mut waiter = Box::pin(sched2.admit(None, Duration::from_secs(5)));
        assert!(
            futures::poll!(&mut waiter).is_pending(),
            "waiter should have nothing to admit to yet -- the only slot is held"
        );

        // release() hands off: under the old design this would mark the
        // slot busy on the waiter's behalf and send it the slot id; under
        // the fix it only sends a bare wake signal and leaves `busy`
        // alone (the slot was already freed at the top of `release()`).
        sched.release(held.slot_id);

        // Simulate the client disconnecting: the waiter's future is
        // dropped *without ever being polled again* to receive that wake
        // signal and re-claim the slot.
        drop(waiter);

        // The slot must be free for the next caller, not stuck busy
        // forever with no owner and no way to release it.
        let snap = sched.snapshot();
        let slot = snap.iter().find(|s| s.slot_id == held.slot_id).unwrap();
        assert!(
            !slot.busy,
            "slot leaked: still busy after its woken waiter was dropped unpolled"
        );

        // And it must actually be admittable again, not just cosmetically
        // "not busy" in the snapshot.
        let fresh = sched.admit(None, Duration::from_secs(1)).await;
        assert!(
            fresh.is_ok(),
            "slot should still be admittable after the leak repro"
        );
    }

    // --- Fix 1: seed-busy slots can never recover ---------------------

    #[tokio::test]
    async fn reconcile_seeded_idle_frees_a_never_claimed_seeded_slot() {
        let sched = Scheduler::new(1);
        sched.seed_busy(0);

        // Seeded busy, never claimed by this broker -- must not be
        // admittable yet.
        let result = sched.admit(None, Duration::from_millis(50)).await;
        assert!(matches!(result, Err(SchedulerError::Timeout)));

        // Background reconciliation observes llama-server now reports the
        // slot idle.
        sched.reconcile_seeded_idle(0);

        // Now admittable, with no `release()` ever having been called.
        let admission = sched.admit(None, Duration::from_secs(1)).await.unwrap();
        assert_eq!(admission.slot_id, 0);
    }

    #[tokio::test]
    async fn reconcile_seeded_idle_is_a_no_op_for_a_slot_claimed_by_the_broker() {
        // Safety property: once a slot has been claimed by the broker
        // itself (via `admit()`, not `seed_busy()`), `reconcile_seeded_idle`
        // must never touch it -- in particular it must never mark a slot
        // free out from under a live guard.
        let sched = Scheduler::new(1);
        let held = sched.admit(None, Duration::from_secs(1)).await.unwrap();

        // The slot is busy because the broker itself admitted it, not
        // because it was seeded. Calling reconcile on it while still busy
        // must be a no-op.
        sched.reconcile_seeded_idle(held.slot_id);

        let snap = sched.snapshot();
        let slot = snap.iter().find(|s| s.slot_id == held.slot_id).unwrap();
        assert!(
            slot.busy,
            "reconcile_seeded_idle must not free a slot the broker itself claimed"
        );
    }

    // --- Fix 2: slot allocation should be LRU, not first-free ---------

    #[tokio::test]
    async fn fallback_admission_spreads_across_slots_via_lru_not_first_free() {
        let sched = Scheduler::new(4);

        // Admit+release with 4 different hints in sequence. With first-free
        // allocation every one of these would land on slot 0 every time
        // (release always frees slot 0 back up before the next admit).
        // With LRU, each successive admission should prefer a slot other
        // than the one most recently used.
        let mut slots_used = Vec::new();
        for i in 0..4 {
            let hint = Some(SlotHint {
                prefix_hash: format!("p{i}"),
                preferred_slot: None,
            });
            let admission = sched.admit(hint, Duration::from_secs(1)).await.unwrap();
            slots_used.push(admission.slot_id);
            sched.release(admission.slot_id);
        }

        let distinct: std::collections::HashSet<_> = slots_used.iter().collect();
        assert_eq!(
            distinct.len(),
            4,
            "expected all 4 slots to be used across 4 sequential admissions, got {:?}",
            slots_used
        );
    }

    #[tokio::test]
    async fn ping_pong_between_two_prefixes_uses_more_than_one_slot() {
        // Regression for the review's exact scenario: two alternating
        // distinct prefixes on 4 slots should no longer pile onto a
        // single slot across repeated admit/release cycles.
        let sched = Scheduler::new(4);
        let mut slots_used = std::collections::HashSet::new();

        for i in 0..8 {
            let prefix = if i % 2 == 0 { "prefix-a" } else { "prefix-b" };
            let hint = Some(SlotHint {
                prefix_hash: prefix.to_string(),
                preferred_slot: None,
            });
            let admission = sched.admit(hint, Duration::from_secs(1)).await.unwrap();
            slots_used.insert(admission.slot_id);
            sched.release(admission.slot_id);
        }

        assert!(
            slots_used.len() > 1,
            "expected more than 1 distinct slot across 8 ping-pong requests, got {:?}",
            slots_used
        );
    }

    // --- Fix 3: release() must wake every queued waiter, not just one -

    #[tokio::test]
    async fn release_wakes_a_second_waiter_even_if_the_first_woken_one_is_dropped_unpolled() {
        // Regression for Fix 3: previously `release()` returned after the
        // first successful `send()`. If that first waiter (A) is dropped
        // without being re-polled (simulating a client disconnect in the
        // narrow window between the send succeeding and A's future being
        // polled again), no one else was ever woken -- B would strand
        // until its own timeout even though the slot sat free the whole
        // time. After the fix, B must be woken directly by `release()`
        // too and get admitted well before its timeout.
        let sched = Scheduler::new(1);
        let held = sched.admit(None, Duration::from_secs(5)).await.unwrap();

        // Queue waiter A, poll it once by hand so it registers itself in
        // the wait queue and suspends -- without ever spawning it, so
        // nothing else can ever poll it again on our behalf.
        let sched_a = sched.clone();
        let mut waiter_a = Box::pin(sched_a.admit(None, Duration::from_secs(5)));
        assert!(futures::poll!(&mut waiter_a).is_pending());

        // Queue waiter B for real, on the runtime, with a short-ish
        // timeout so the test fails fast if B is never woken.
        let sched_b = sched.clone();
        let waiter_b =
            tokio::spawn(async move { sched_b.admit(None, Duration::from_secs(5)).await });

        // Give B a moment to register behind A in the queue.
        tokio::time::sleep(Duration::from_millis(20)).await;

        // Release wakes both A and B's wake signals (Fix 3). Simulate A's
        // future being dropped without ever being re-polled to consume
        // its wake and reclaim the slot.
        sched.release(held.slot_id);
        drop(waiter_a);

        // B must still get admitted -- well before its 5s timeout -- even
        // though A "won" the initial wake and then vanished.
        let admitted = tokio::time::timeout(Duration::from_secs(1), waiter_b)
            .await
            .expect("B should be admitted well before its timeout, not stranded by A's drop")
            .unwrap()
            .unwrap();
        assert_eq!(admitted.slot_id, held.slot_id);
    }

    // --- Fix 4: out-of-range preferred_slot must fall back, not hang --

    #[tokio::test]
    async fn out_of_range_preferred_slot_falls_back_instead_of_hanging_the_full_timeout() {
        let sched = Scheduler::new(2);
        let hint = Some(SlotHint {
            prefix_hash: "p".to_string(),
            preferred_slot: Some(99),
        });

        // Should succeed quickly by falling back to a real slot, not queue
        // on slot id 99 (which nothing will ever pop) and burn the whole
        // timeout.
        let admission = tokio::time::timeout(
            Duration::from_millis(200),
            sched.admit(hint, Duration::from_secs(60)),
        )
        .await
        .expect("admit() should not hang the full queue timeout for an out-of-range preferred_slot")
        .unwrap();
        assert!(admission.slot_id < 2);
    }

    // --- Finding 1 (blocking, review round 3): out-of-range
    // preferred_slot must not hang the full timeout under contention -----

    #[tokio::test]
    async fn out_of_range_preferred_slot_under_contention_falls_back_instead_of_hanging() {
        // Regression for Finding 1: `try_admit_locked` already filtered an
        // out-of-range `preferred_slot` down to "no preference" -- but only
        // reachable when it actually runs. When every slot is already
        // busy, `try_admit_locked` returns `None` for the ordinary reason
        // "nothing free right now," and `admit()`'s own queue-selection
        // logic used to re-read the *raw* `preferred_slot` straight off the
        // hint, queuing this caller on `slot_waiters[99]` -- a key
        // `release()` can never pop for a 2-slot broker. That stranded the
        // caller for its entire queue timeout even though a slot freed up
        // well before then.
        let sched = Scheduler::new(2);

        // Occupy both slots first so `try_admit_locked` returns `None` for
        // "nothing idle," not for the out-of-range check -- the prior test
        // (`out_of_range_preferred_slot_falls_back_instead_of_hanging_the_full_timeout`)
        // only covers the case where a slot IS free, which doesn't reach
        // this bug at all.
        let held_a = sched.admit(None, Duration::from_secs(1)).await.unwrap();
        let _held_b = sched.admit(None, Duration::from_secs(1)).await.unwrap();

        let hint = Some(SlotHint {
            prefix_hash: "p".to_string(),
            preferred_slot: Some(99),
        });
        let sched2 = sched.clone();
        let waiter = tokio::spawn(async move { sched2.admit(hint, Duration::from_secs(60)).await });

        // Give the waiter time to register itself in the (correct) queue.
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Free one of the two occupied slots.
        sched.release(held_a.slot_id);

        // The waiter must be admitted promptly -- well before its 60s
        // timeout -- not stuck on a `slot_waiters[99]` key nothing pops.
        let admission = tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .expect("waiter should be admitted promptly after a slot frees, not hang on a dead out-of-range queue key")
            .unwrap()
            .unwrap();
        assert!(admission.slot_id < 2);
    }

    // --- Finding 2 (high, review round 3): seed-busy reconciliation must
    // wake waiters when it frees a slot -----------------------------------

    #[tokio::test]
    async fn reconcile_seeded_idle_wakes_a_queued_waiter() {
        // Regression for Finding 2: `reconcile_seeded_idle` correctly flips
        // `busy` to `false` for a still-seeded-unowned slot, but previously
        // woke no one -- any waiter already queued (global or
        // slot-specific) for that slot sat there until its own timeout
        // even though the slot it wanted just became free.
        let sched = Scheduler::new(1);
        sched.seed_busy(0);

        // Queue a waiter (global, no preference) behind the seeded-busy
        // slot, with a long timeout so the test fails fast (not slow) if
        // the wake never happens.
        let sched2 = sched.clone();
        let waiter = tokio::spawn(async move { sched2.admit(None, Duration::from_secs(60)).await });

        // Give the waiter time to register itself in the global queue.
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Background reconciliation observes llama-server now reports the
        // slot idle.
        sched.reconcile_seeded_idle(0);

        // The waiter must be admitted promptly -- well before its 60s
        // timeout -- not only reflected in `snapshot()`'s busy flag.
        let admission = tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .expect("waiter should be woken promptly by reconcile_seeded_idle, not stranded until its own timeout")
            .unwrap()
            .unwrap();
        assert_eq!(admission.slot_id, 0);
    }

    // --- Fix 7: clear_resident ------------------------------------------

    #[tokio::test]
    async fn clear_resident_removes_stale_prefix_bookkeeping() {
        let sched = Scheduler::new(1);
        let hint = Some(SlotHint {
            prefix_hash: "a".to_string(),
            preferred_slot: None,
        });
        let first = sched.admit(hint, Duration::from_secs(1)).await.unwrap();
        sched.mark_resident(first.slot_id, "a".to_string());
        sched.release(first.slot_id);

        // Hint-less admission lands on the same (only) slot, silently
        // overwriting its real KV content. The HTTP handler is expected
        // to call `clear_resident` in this case.
        let second = sched.admit(None, Duration::from_secs(1)).await.unwrap();
        assert_eq!(second.slot_id, first.slot_id);
        sched.clear_resident(second.slot_id);

        let snap = sched.snapshot();
        let slot = snap.iter().find(|s| s.slot_id == second.slot_id).unwrap();
        assert_eq!(slot.resident_prefix, None);
    }
}
