use crate::domain::errors::DocumentError;
use std::sync::{Arc, RwLock};

use super::config::SandboxConfig;

/// A thread-safe, cheaply-cloneable handle to a [`SandboxConfig`] that supports
/// runtime hot-swapping of sandbox roots.
///
/// All `SharedSandbox` validation functions snapshot the inner config via a brief
/// read lock and delegate to the existing `&SandboxConfig` helpers. The `set`
/// method replaces the config under a write lock after re-canonicalizing roots.
///
/// # Poisoning
///
/// This type uses [`std::sync::RwLock`]. A panic while holding a write lock will
/// poison the lock; subsequent `snapshot`/`set` calls will panic via `.expect()`.
/// This is the deliberate choice: a poisoned sandbox indicates a logic error that
/// should be surfaced immediately rather than silently returning stale roots.
#[derive(Debug, Clone)]
pub struct SharedSandbox {
    inner: Arc<RwLock<SandboxConfig>>,
}

impl SharedSandbox {
    /// Creates a new `SharedSandbox` wrapping the given config.
    pub fn new(initial: SandboxConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(initial)),
        }
    }

    /// Returns a clone of the current [`SandboxConfig`].
    ///
    /// Acquires a read lock, clones the inner config, and releases the lock.
    /// The clone is small (two `Vec<PathBuf>`) and the lock is uncontended for
    /// typical single-agent use.
    pub fn snapshot(&self) -> SandboxConfig {
        #[allow(clippy::expect_used)]
        self.inner.read().expect("sandbox rwlock poisoned").clone()
    }

    /// Replaces the sandbox configuration, re-canonicalizing all roots.
    ///
    /// # Errors
    ///
    /// Returns [`DocumentError::Rag`] if `new_config.roots()` is empty.
    /// Returns [`DocumentError::Io`] if any root cannot be canonicalized.
    pub fn set(&self, new_config: SandboxConfig) -> Result<(), DocumentError> {
        let re_canonicalized = SandboxConfig::new(new_config.roots().to_vec())?;
        #[allow(clippy::expect_used)]
        let mut guard = self.inner.write().expect("sandbox rwlock poisoned");
        *guard = re_canonicalized;
        Ok(())
    }
}

impl From<SandboxConfig> for SharedSandbox {
    fn from(config: SandboxConfig) -> Self {
        Self::new(config)
    }
}

impl From<&SandboxConfig> for SharedSandbox {
    fn from(config: &SandboxConfig) -> Self {
        Self::new(config.clone())
    }
}

impl Default for SharedSandbox {
    fn default() -> Self {
        Self::new(SandboxConfig::default())
    }
}
