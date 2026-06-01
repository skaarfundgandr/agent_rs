use crate::domain::errors::DocumentError;
use std::path::{Path, PathBuf};

/// Configuration for filesystem sandboxing with multiple allowed roots.
///
/// The first root in the list is the **primary** — used as the default
/// target when creating new files/directories at paths that don't exist
/// under any root. All roots participate equally in read/search/glob.
///
/// # Examples
///
/// ```rust,no_run
/// # fn main() -> Result<(), agent_rs_lib::domain::errors::DocumentError> {
/// use std::path::PathBuf;
/// use agent_rs_lib::security::SandboxConfig;
///
/// // Single root (backward-compatible)
/// let single = SandboxConfig::single("/home/user/workspace")?;
///
/// // Multiple roots — primary first
/// let multi = SandboxConfig::new(vec![
///     PathBuf::from("/home/user/workspace"),
///     PathBuf::from("/tmp/shared-docs"),
/// ])?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    /// Ordered list of sandbox roots as provided (primary first).
    /// Used for display paths and user-facing operations.
    roots: Vec<PathBuf>,
    /// Pre-canonicalized versions for security validation.
    canonical_roots: Vec<PathBuf>,
}

impl SandboxConfig {
    /// Creates a sandbox configuration with a single root.
    ///
    /// # Errors
    ///
    /// Returns [`DocumentError::Io`] if the root cannot be canonicalized
    /// (e.g., the path does not exist).
    pub fn single(root: impl Into<PathBuf>) -> Result<Self, DocumentError> {
        Self::new(vec![root.into()])
    }

    /// Creates a sandbox configuration with multiple roots.
    ///
    /// The first root is the primary (default target for new file writes).
    /// All roots must be canonicalizable at construction time.
    ///
    /// # Errors
    ///
    /// Returns [`DocumentError::Rag`] if `roots` is empty.
    /// Returns [`DocumentError::Io`] if any root cannot be canonicalized
    /// (e.g., the path does not exist).
    pub fn new(roots: Vec<PathBuf>) -> Result<Self, DocumentError> {
        if roots.is_empty() {
            return Err(DocumentError::Rag(
                "SandboxConfig requires at least one root".to_string(),
            ));
        }

        let canonical_roots: Result<Vec<PathBuf>, DocumentError> = roots
            .iter()
            .map(|r| r.canonicalize().map_err(DocumentError::Io))
            .collect();

        let canonical_roots = canonical_roots?;

        Ok(Self {
            roots,
            canonical_roots,
        })
    }

    /// Returns the primary (first) root.
    pub fn primary(&self) -> &Path {
        &self.roots[0]
    }

    /// Returns the original (non-canonicalized) roots as provided.
    /// Use for display paths and user-facing operations.
    pub fn roots(&self) -> &[PathBuf] {
        &self.roots
    }

    /// Returns all canonicalized roots.
    /// Use for security validation.
    pub fn canonical_roots(&self) -> &[PathBuf] {
        &self.canonical_roots
    }

    /// Returns the number of configured roots.
    pub fn len(&self) -> usize {
        self.roots.len()
    }

    /// Returns `true` if no roots are configured (should never happen).
    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }
}

impl Default for SandboxConfig {
    /// Creates a default config with the current directory as the sole root.
    ///
    /// # Panics
    ///
    /// Panics if the current directory cannot be determined or canonicalized.
    /// This is acceptable because the current directory is always a valid
    /// concept in a running process.
    fn default() -> Self {
        Self::single(".").expect("current directory is always a valid sandbox root")
    }
}

impl TryFrom<PathBuf> for SandboxConfig {
    type Error = DocumentError;

    fn try_from(root: PathBuf) -> Result<Self, DocumentError> {
        Self::single(root)
    }
}

impl TryFrom<&Path> for SandboxConfig {
    type Error = DocumentError;

    fn try_from(root: &Path) -> Result<Self, DocumentError> {
        Self::single(root)
    }
}

impl TryFrom<&str> for SandboxConfig {
    type Error = DocumentError;

    fn try_from(root: &str) -> Result<Self, DocumentError> {
        Self::single(root)
    }
}

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
    for canonical_root in &sandbox.canonical_roots {
        if let Some(resolved) = try_resolve_within_root(canonical_root, user_path)?
            && resolved.exists()
        {
            return Ok(resolved);
        }
    }

    // Phase 2: File doesn't exist in any root — use primary root for writes
    // This allows creating new files under the primary root
    let primary_root = &sandbox.canonical_roots[0];
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
pub fn find_containing_root<'a>(sandbox: &'a SandboxConfig, path: &Path) -> Option<&'a PathBuf> {
    // Try canonical comparison first (works for existing, canonicalized paths)
    let canonical_path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    for (original_root, canonical_root) in sandbox.roots.iter().zip(sandbox.canonical_roots.iter())
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
pub fn relative_display_path(sandbox: &SandboxConfig, path: &Path) -> String {
    // Canonical comparison first — works for paths returned from
    // validate_sandboxed_path (already canonicalized, no extra syscall)
    for (original_root, canonical_root) in sandbox.roots.iter().zip(sandbox.canonical_roots.iter())
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_single_root() {
        let tmp = tempfile::tempdir().unwrap();
        let config = SandboxConfig::single(tmp.path()).unwrap();
        assert_eq!(config.len(), 1);
        assert_eq!(config.primary(), tmp.path());
    }

    #[test]
    fn test_multi_root() {
        let tmp1 = tempfile::tempdir().unwrap();
        let tmp2 = tempfile::tempdir().unwrap();
        let config =
            SandboxConfig::new(vec![tmp1.path().to_path_buf(), tmp2.path().to_path_buf()]).unwrap();
        assert_eq!(config.len(), 2);
        assert_eq!(config.primary(), tmp1.path());
    }

    #[test]
    fn test_validate_within_root() {
        let tmp = tempfile::tempdir().unwrap();
        let config = SandboxConfig::single(tmp.path()).unwrap();
        fs::write(tmp.path().join("test.txt"), "hello").unwrap();

        let result = validate_sandboxed_path(&config, Path::new("test.txt")).unwrap();
        assert!(result.starts_with(&config.canonical_roots()[0]));
    }

    #[test]
    fn test_validate_escape_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let config = SandboxConfig::single(tmp.path()).unwrap();

        let err = validate_sandboxed_path(&config, Path::new("../escaped.txt")).unwrap_err();
        assert!(matches!(err, DocumentError::SandboxEscape(_)));
    }

    #[test]
    fn test_multi_root_read_from_secondary() {
        let tmp1 = tempfile::tempdir().unwrap();
        let tmp2 = tempfile::tempdir().unwrap();
        let config =
            SandboxConfig::new(vec![tmp1.path().to_path_buf(), tmp2.path().to_path_buf()]).unwrap();

        fs::write(tmp2.path().join("shared.txt"), "from secondary").unwrap();

        let result = validate_sandboxed_path(&config, Path::new("shared.txt")).unwrap();
        assert!(result.to_string_lossy().contains("shared.txt"));
    }

    #[test]
    fn test_multi_root_escape_rejected() {
        let tmp1 = tempfile::tempdir().unwrap();
        let tmp2 = tempfile::tempdir().unwrap();
        let config =
            SandboxConfig::new(vec![tmp1.path().to_path_buf(), tmp2.path().to_path_buf()]).unwrap();

        let err = validate_sandboxed_path(&config, Path::new("../../etc/passwd")).unwrap_err();
        assert!(matches!(err, DocumentError::SandboxEscape(_)));
    }

    #[test]
    fn test_find_containing_root() {
        let tmp1 = tempfile::tempdir().unwrap();
        let tmp2 = tempfile::tempdir().unwrap();
        let config =
            SandboxConfig::new(vec![tmp1.path().to_path_buf(), tmp2.path().to_path_buf()]).unwrap();

        // Create the file so canonicalize() works on Windows
        fs::write(tmp2.path().join("file.txt"), "content").unwrap();
        let path = tmp2.path().join("file.txt");
        let found = find_containing_root(&config, &path).unwrap();
        assert_eq!(found, tmp2.path());
    }

    #[test]
    fn test_relative_display_path() {
        let tmp = tempfile::tempdir().unwrap();
        let config = SandboxConfig::single(tmp.path()).unwrap();
        let file = tmp.path().join("src/main.rs");

        let display = relative_display_path(&config, &file);
        assert_eq!(display, "src/main.rs");
    }

    #[test]
    fn test_relative_display_path_multi_root() {
        let tmp1 = tempfile::tempdir().unwrap();
        let tmp2 = tempfile::tempdir().unwrap();
        let config =
            SandboxConfig::new(vec![tmp1.path().to_path_buf(), tmp2.path().to_path_buf()]).unwrap();

        let file = tmp2.path().join("docs/readme.md");
        let display = relative_display_path(&config, &file);
        assert_eq!(display, "docs/readme.md");
    }

    #[test]
    fn test_validate_nonexistent_path_under_primary() {
        let tmp = tempfile::tempdir().unwrap();
        let config = SandboxConfig::single(tmp.path()).unwrap();

        let result = validate_sandboxed_path(&config, Path::new("newdir/newfile.txt")).unwrap();
        assert!(result.to_string_lossy().contains("newdir"));
        assert!(result.to_string_lossy().contains("newfile.txt"));
    }

    #[test]
    fn test_empty_roots_returns_error() {
        let err = SandboxConfig::new(vec![]).unwrap_err();
        assert!(matches!(err, DocumentError::Rag(_)));
        assert!(
            err.to_string()
                .contains("SandboxConfig requires at least one root")
        );
    }

    #[test]
    fn test_sandbox_config_clone() {
        let tmp = tempfile::tempdir().unwrap();
        let config = SandboxConfig::single(tmp.path()).unwrap();
        let cloned = config.clone();
        assert_eq!(config.len(), cloned.len());
    }
}
