use std::path::{Path, PathBuf};

use crate::domain::errors::DocumentError;

use super::resolve::{find_containing_root, relative_display_path, validate_sandboxed_path};
use super::shared::SharedSandbox;

/// Shared-sandbox variant of [`validate_sandboxed_path`].
///
/// Snapshots the current [`super::config::SandboxConfig`] from `sandbox` and
/// delegates to the `&SandboxConfig` version.
pub fn validate_sandboxed_path_shared(
    sandbox: &SharedSandbox,
    user_path: &Path,
) -> Result<PathBuf, DocumentError> {
    validate_sandboxed_path(&sandbox.snapshot(), user_path)
}

/// Shared-sandbox variant of [`find_containing_root`].
///
/// Unlike the original which returns a borrow of the inner `SandboxConfig`'s
/// root, this function returns an **owned** `PathBuf` because the snapshot is a
/// temporary — borrowing from it would require leaking the lock guard, which
/// `snapshot()` does not expose.
pub fn find_containing_root_shared(sandbox: &SharedSandbox, path: &Path) -> Option<PathBuf> {
    let config = sandbox.snapshot();
    find_containing_root(&config, path).cloned()
}

/// Shared-sandbox variant of [`relative_display_path`].
///
/// Snapshots the current [`super::config::SandboxConfig`] from `sandbox` and
/// delegates to the `&SandboxConfig` version.
pub fn relative_display_path_shared(sandbox: &SharedSandbox, path: &Path) -> String {
    relative_display_path(&sandbox.snapshot(), path)
}
