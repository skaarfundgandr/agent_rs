use crate::domain::errors::DocumentError;
use std::path::{Path, PathBuf};

use super::config::SandboxConfig;

/// Validates that `user_path` resolves to a path within one of the sandbox roots.
///
/// Tries each canonical root in order. For each root, joins `user_path` onto
/// the root and canonicalizes the result. If the canonicalized path falls
/// within a root, it is returned.
///
/// For **existing** paths: returns the first root where the file exists.
/// For **non-existent** paths (writes): walks up the tree to find the nearest
/// existing ancestor under any root, using the primary root as default.
///
/// # Arguments
///
/// * `sandbox` - The sandbox configuration containing allowed roots.
/// * `user_path` - The target path provided by the user.
///
/// # Returns
///
/// Returns the canonicalized target path if it lies within the sandbox.
///
/// # Errors
///
/// Returns [`DocumentError::SandboxEscape`] if the resolved path falls
/// outside all configured sandbox roots. The error message includes the
/// configured roots for debugging.
pub fn validate_sandboxed_path(
    sandbox: &SandboxConfig,
    user_path: &Path,
) -> Result<PathBuf, DocumentError> {
    // Phase 1: Find which root the file actually exists in (for reads)
    for canonical_root in sandbox.canonical_roots() {
        if let Some(resolved) = try_resolve_within_root(canonical_root, user_path)?
            && resolved.exists()
        {
            return Ok(resolved);
        }
    }

    // Phase 2: File doesn't exist in any root — use primary root for writes
    // This allows creating new files under the primary root
    let primary_root = &sandbox.canonical_roots()[0];
    if let Some(resolved) = try_resolve_within_root(primary_root, user_path)? {
        return Ok(resolved);
    }

    Err(DocumentError::SandboxEscape(format!(
        "Path '{}' is not within any sandbox root: {:?}",
        user_path.display(),
        sandbox.roots()
    )))
}

/// Attempts to resolve `user_path` within a single canonical root.
///
/// Returns `Ok(Some(path))` if the path is within the root,
/// `Ok(None)` if the path escapes the root, or `Err` on IO failure.
fn try_resolve_within_root(
    canonical_root: &Path,
    user_path: &Path,
) -> Result<Option<PathBuf>, DocumentError> {
    let target = canonical_root.join(user_path);

    let canonical_target = if target.exists() {
        target.canonicalize().map_err(DocumentError::Io)?
    } else {
        let mut existing_parent = target.as_path();
        let mut components_to_append = Vec::new();

        while !existing_parent.exists() {
            if let Some(parent) = existing_parent.parent() {
                if let Some(file_name) = existing_parent.file_name() {
                    components_to_append.push(file_name);
                }
                existing_parent = parent;
            } else {
                break;
            }
        }

        let mut canonical_path = existing_parent.canonicalize().map_err(DocumentError::Io)?;
        for comp in components_to_append.into_iter().rev() {
            canonical_path.push(comp);
        }
        canonical_path
    };

    if canonical_target.starts_with(canonical_root) {
        Ok(Some(canonical_target))
    } else {
        Ok(None)
    }
}

/// Returns the original (non-canonicalized) root that contains `path`,
/// or `None` if the path does not fall under any configured root.
///
/// Uses canonical roots for comparison to handle Windows `\\?\` prefix
/// correctly. Falls back to non-canonical comparison for non-existent paths.
///
/// # Arguments
///
/// * `sandbox` - The sandbox configuration containing allowed roots.
/// * `path` - The path to check containing root for.
///
/// # Returns
///
/// Returns `Some(&PathBuf)` of the containing sandbox root, or `None` if not found.
pub fn find_containing_root<'a>(sandbox: &'a SandboxConfig, path: &Path) -> Option<&'a PathBuf> {
    // Try canonical comparison first (works for existing, canonicalized paths)
    let canonical_path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    for (original_root, canonical_root) in
        sandbox.roots().iter().zip(sandbox.canonical_roots().iter())
    {
        if canonical_path.starts_with(canonical_root) {
            return Some(original_root);
        }
    }
    None
}

/// Computes a relative display path for a file, preferring the shortest
/// prefix among the sandbox roots for readability.
///
/// Tries canonical comparison first (cheap, works for already-canonical paths
/// returned from [`validate_sandboxed_path`]). Falls back to original-root
/// comparison for non-existent paths without a canonicalize syscall.
///
/// # Arguments
///
/// * `sandbox` - The sandbox configuration.
/// * `path` - The absolute/relative path to convert.
///
/// # Returns
///
/// Returns the relative display path as a `String`.
pub fn relative_display_path(sandbox: &SandboxConfig, path: &Path) -> String {
    // Canonical comparison first — works for paths returned from
    // validate_sandboxed_path (already canonicalized, no extra syscall)
    for (original_root, canonical_root) in
        sandbox.roots().iter().zip(sandbox.canonical_roots().iter())
    {
        if let Ok(rel) = path.strip_prefix(canonical_root) {
            return rel.to_string_lossy().into_owned();
        }
        // Fallback: original-root comparison (works for non-existent paths,
        // avoids canonicalize syscall)
        if let Ok(rel) = path.strip_prefix(original_root) {
            return rel.to_string_lossy().into_owned();
        }
    }
    path.to_string_lossy().into_owned()
}
