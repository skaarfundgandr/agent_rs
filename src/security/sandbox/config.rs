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
/// # fn main() -> Result<(), agent_rs::domain::errors::DocumentError> {
/// use std::path::PathBuf;
/// use agent_rs::security::SandboxConfig;
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
    /// # Arguments
    ///
    /// * `root` - The directory path to use as the sandbox root.
    ///
    /// # Returns
    ///
    /// Returns the initialized `SandboxConfig`.
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
    /// # Arguments
    ///
    /// * `roots` - A vector of directory paths to use as allowed roots.
    ///
    /// # Returns
    ///
    /// Returns the initialized `SandboxConfig`.
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
    ///
    /// # Returns
    ///
    /// Returns a reference to the primary sandbox root `Path`.
    pub fn primary(&self) -> &Path {
        &self.roots[0]
    }

    /// Returns the original (non-canonicalized) roots as provided.
    /// Use for display paths and user-facing operations.
    ///
    /// # Returns
    ///
    /// Returns a slice of the original, non-canonicalized sandbox root paths.
    pub fn roots(&self) -> &[PathBuf] {
        &self.roots
    }

    /// Returns all canonicalized roots.
    /// Use for security validation.
    ///
    /// # Returns
    ///
    /// Returns a slice of the canonicalized sandbox root paths.
    pub fn canonical_roots(&self) -> &[PathBuf] {
        &self.canonical_roots
    }

    /// Returns the number of configured roots.
    ///
    /// # Returns
    ///
    /// Returns the count of sandbox roots.
    pub fn len(&self) -> usize {
        self.roots.len()
    }

    /// Returns `true` if no roots are configured (should never happen).
    ///
    /// # Returns
    ///
    /// Returns `true` if there are no roots configured, `false` otherwise.
    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }

    /// Adds a single root to the sandbox configuration.
    ///
    /// The root is canonicalized; if its canonical form already exists, the
    /// add is a no-op (idempotent).
    ///
    /// # Errors
    ///
    /// Returns [`DocumentError::Io`] if the root cannot be canonicalized.
    pub fn add_root<P: AsRef<Path>>(&mut self, root: P) -> Result<(), DocumentError> {
        let new_canonical = root.as_ref().canonicalize().map_err(DocumentError::Io)?;
        if self.canonical_roots.iter().any(|r| r == &new_canonical) {
            return Ok(());
        }
        self.roots.push(root.as_ref().to_path_buf());
        self.canonical_roots.push(new_canonical);
        Ok(())
    }

    /// Adds multiple roots atomically (per-item atomicity).
    ///
    /// A successful `add_root` mutates before the next item is validated.
    /// A failing item after a successful one leaves partial state.
    ///
    /// # Errors
    ///
    /// Returns the first [`DocumentError::Io`] encountered; partial
    /// additions from prior successful items remain.
    pub fn add_roots<I, P>(&mut self, roots: I) -> Result<(), DocumentError>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        for root in roots {
            self.add_root(root)?;
        }
        Ok(())
    }

    /// Removes a root from the sandbox configuration.
    ///
    /// The root is canonicalized for lookup; if not found, the call is a
    /// no-op (idempotent on miss). Removing the last root is an error.
    ///
    /// # Errors
    ///
    /// Returns [`DocumentError::Io`] if the root cannot be canonicalized.
    /// Returns [`DocumentError::Sandbox`] if it is the last remaining root.
    pub fn remove_root<P: AsRef<Path>>(&mut self, root: P) -> Result<(), DocumentError> {
        let target = root.as_ref().canonicalize().map_err(DocumentError::Io)?;
        let Some(i) = self.canonical_roots.iter().position(|r| r == &target) else {
            return Ok(());
        };
        if self.roots.len() == 1 {
            return Err(DocumentError::Sandbox(
                "cannot remove the last sandbox root".to_string(),
            ));
        }
        self.roots.remove(i);
        self.canonical_roots.remove(i);
        Ok(())
    }

    /// Checks whether a root exists in the sandbox configuration.
    ///
    /// The root is canonicalized for comparison.
    ///
    /// # Errors
    ///
    /// Returns [`DocumentError::Io`] if the root cannot be canonicalized.
    pub fn contains_root<P: AsRef<Path>>(&self, root: P) -> Result<bool, DocumentError> {
        let target = root.as_ref().canonicalize().map_err(DocumentError::Io)?;
        Ok(self.canonical_roots.iter().any(|r| r == &target))
    }
}

impl Default for SandboxConfig {
    /// Creates a default config with the current directory as the sole root.
    ///
    /// # Panics
    ///
    /// Panics if the current working directory cannot be determined or
    /// canonicalized (e.g., it was deleted while the process was running).
    /// For daemon or long-running use cases where the CWD may change,
    /// prefer [`SandboxConfig::new`] with an explicit root path.
    #[allow(clippy::expect_used)]
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
