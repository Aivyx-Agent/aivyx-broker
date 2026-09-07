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
                        match hint.as_ref().and_then(|h| h.preferred_slot) {
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
        if let Some(h) = hint
            && let Some(preferred) = h.preferred_slot {
                let slot = inner.slots.get_mut(preferred as usize)?;
                if !slot.busy {
                    slot.busy = true;
                    let cache_ready = slot.resident_prefix.as_deref() == Some(h.prefix_hash.as_str());
                    return Some(Admission { slot_id: preferred, cache_ready });
                }
                return None; // must queue for this specific slot
            }
        // No preference: any free slot, prefer one already resident with
        // this exact prefix, else LRU-first among idle (first free found).
        if let Some(h) = hint
            && let Some((idx, slot)) = inner
                .slots
                .iter_mut()
                .enumerate()
                .find(|(_, s)| !s.busy && s.resident_prefix.as_deref() == Some(h.prefix_hash.as_str()))
            {
                slot.busy = true;
                return Some(Admission { slot_id: idx as u32, cache_ready: true });
            }
        let (idx, slot) = inner.slots.iter_mut().enumerate().find(|(_, s)| !s.busy)?;
        slot.busy = true;
        let cache_ready = hint
            .as_ref()
            .is_some_and(|h| slot.resident_prefix.as_deref() == Some(h.prefix_hash.as_str()));
        Some(Admission { slot_id: idx as u32, cache_ready })
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

    /// Frees `slot_id` and, if anyone is waiting, wakes exactly one
    /// waiter -- but only with a bare wake *signal*, never a handed-off
    /// slot id. The slot is left `busy = false` here unconditionally.
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

        // Wake this slot's specific waiters first (rule: don't let a
        // global waiter steal a slot someone is specifically waiting on).
        // Loop through slot-specific waiters until one successfully
        // receives the wake signal, skipping any whose `Receiver` was
        // already dropped (timed out, or the waiter's future was
        // cancelled).
        if let Some(queue) = inner.slot_waiters.get_mut(&slot_id) {
            while let Some(tx) = queue.pop_front() {
                if tx.send(()).is_ok() {
                    return;
                }
                // This waiter is gone; continue to the next one. `busy`
                // stays false -- nothing was ever handed to it.
            }
        }

        // No slot-specific waiter accepted; try global waiters.
        while let Some(tx) = inner.global_waiters.pop_front() {
            if tx.send(()).is_ok() {
                return;
            }
            // This waiter is gone; continue to the next one.
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

        let hint = Some(SlotHint { prefix_hash: "prefix-a".to_string(), preferred_slot: None });
        let second = sched.admit(hint, Duration::from_secs(1)).await.unwrap();
        assert_eq!(second.slot_id, first.slot_id);
        assert!(second.cache_ready);
    }

    #[tokio::test]
    async fn preferred_slot_busy_queues_for_that_slot_specifically() {
        let sched = Scheduler::new(2);
        let hint_a = Some(SlotHint { prefix_hash: "a".to_string(), preferred_slot: Some(0) });
        let first = sched.admit(hint_a.clone(), Duration::from_secs(1)).await.unwrap();
        assert_eq!(first.slot_id, 0);

        // A second request preferring the same slot must wait, not steal
        // slot 1.
        let sched2 = sched.clone();
        let waiter = tokio::spawn(async move {
            sched2.admit(hint_a, Duration::from_secs(5)).await
        });

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
        let this_slot = snap.iter().find(|s| s.slot_id == admission.slot_id).unwrap();
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
        let _held = sched.admit(hint.clone(), Duration::from_secs(1)).await.unwrap();

        // Queue first waiter with a very short timeout so it will expire
        // before we call release().
        let sched_first = sched.clone();
        let first_waiter = tokio::spawn(async move {
            sched_first.admit(hint.clone(), Duration::from_millis(100)).await
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
        let second_waiter = tokio::spawn(async move {
            sched_second.admit(hint2, Duration::from_secs(5)).await
        });

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
        assert!(!slot.busy, "slot leaked: still busy after its woken waiter was dropped unpolled");

        // And it must actually be admittable again, not just cosmetically
        // "not busy" in the snapshot.
        let fresh = sched.admit(None, Duration::from_secs(1)).await;
        assert!(fresh.is_ok(), "slot should still be admittable after the leak repro");
    }
}
