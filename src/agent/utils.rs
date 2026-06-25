pub(crate) use std::sync::Mutex;

/// Lock a `std::sync::Mutex`, recovering from poisoning if necessary.
///
/// A poisoned mutex means another thread panicked while holding the lock.
/// Since we only hold the lock briefly (for `.clone()` / `.extend()` on Vec),
/// we can safely recover by clearing the poisoned state and returning the
/// guard.
pub(crate) fn lock_mutex<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            // Recover from poison — the inner data is still accessible.
            // This is safe because we only hold locks for short operations
            // (clone/extend on Vec) that don't leave the data in an
            // inconsistent state.
            poisoned.into_inner()
        }
    }
}
