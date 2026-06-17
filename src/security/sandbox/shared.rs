use crate::agent::permission::{PermissionPolicy, PermissionResult};
use crate::domain::errors::DocumentError;
use std::path::{Path, PathBuf};
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

    /// Evaluates the permission `policy` for `tool_name`/`description`.
    ///
    /// This is the gate-check half of [`Self::resolve_path_with_permission`],
    /// exposed separately for tools whose path-resolution logic branches on
    /// the arguments (e.g. `GlobSearchTool`'s optional `directory` arg).
    ///
    /// **Semantics:**
    /// - [`PermissionResult::Allow`]: returns `Ok(())`. The caller proceeds and
    ///   resolves paths using [`Self::resolve_path_unchecked`]; the sandbox
    ///   validation is *not* run. The gate is the sole authority for
    ///   out-of-sandbox access.
    /// - [`PermissionResult::Deny`] / [`PermissionResult::DeferToUser`]:
    ///   returns [`DocumentError::PermissionDenied`] immediately. Matches how
    ///   MCP tools short-circuit on a gate denial.
    ///
    /// # Arguments
    ///
    /// * `policy` - The permission policy to evaluate.
    /// * `tool_name` - Tool identifier passed to the gate.
    /// * `description` - Human-readable action description passed to the gate.
    ///
    /// # Errors
    ///
    /// Returns [`DocumentError::PermissionDenied`] when the gate returns `Deny`
    /// or `DeferToUser`.
    pub async fn check_permission(
        &self,
        policy: &PermissionPolicy,
        tool_name: &str,
        description: &str,
    ) -> Result<(), DocumentError> {
        match policy.evaluate(tool_name, description).await {
            PermissionResult::Allow => Ok(()),
            PermissionResult::Deny { reason } => Err(DocumentError::PermissionDenied(format!(
                "{description}: {reason}"
            ))),
            PermissionResult::DeferToUser => Err(DocumentError::PermissionDenied(format!(
                "{description}: defer-to-user not yet supported"
            ))),
        }
    }

    /// Resolves `user_path` against the sandbox roots *without* enforcing
    /// sandbox containment.
    ///
    /// - Absolute paths: canonicalized and returned (falls back to the
    ///   original path on IO error).
    /// - Relative paths: each canonical root is searched in order; the first
    ///   existing match is canonicalized and returned. If no match exists, the
    ///   path is joined onto the primary (first) root — this is the default for
    ///   new-file writes.
    ///
    /// This function does **not** reject out-of-sandbox paths. Permission for
    /// out-of-sandbox access is the responsibility of a permission gate
    /// evaluated by the caller (typically via [`Self::check_permission`]).
    pub fn resolve_path_unchecked(&self, user_path: &Path) -> PathBuf {
        if user_path.is_absolute() {
            return user_path
                .canonicalize()
                .unwrap_or_else(|_| user_path.to_path_buf());
        }

        let snapshot = self.snapshot();
        let roots = snapshot.canonical_roots();

        for canonical_root in roots {
            let candidate = canonical_root.join(user_path);
            if candidate.exists() {
                return candidate.canonicalize().unwrap_or(candidate);
            }
        }

        // No existing match — default to the primary root (for new writes).
        roots
            .first()
            .map(|r| r.join(user_path))
            .unwrap_or_else(|| user_path.to_path_buf())
    }

    /// Convenience wrapper: [`Self::check_permission`] then
    /// [`Self::resolve_path_unchecked`].
    ///
    /// Used by tools whose call-site is a linear "gate, then resolve a single
    /// path" sequence — `ReadDocumentTool`, `WriteDocumentTool`,
    /// `ListDirectoryTool`, `GrepSearchTool`, and the `ManageRagTool`
    /// `add`/`remove` actions. Tools with more complex control flow
    /// (e.g. `GlobSearchTool`'s optional `directory` argument) should call
    /// the two methods separately so the gate is only evaluated once.
    ///
    /// # Errors
    ///
    /// Returns [`DocumentError::PermissionDenied`] when the gate returns `Deny`
    /// or `DeferToUser`.
    pub async fn resolve_path_with_permission(
        &self,
        policy: &PermissionPolicy,
        tool_name: &str,
        description: &str,
        user_path: &Path,
    ) -> Result<PathBuf, DocumentError> {
        self.check_permission(policy, tool_name, description).await?;
        Ok(self.resolve_path_unchecked(user_path))
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
