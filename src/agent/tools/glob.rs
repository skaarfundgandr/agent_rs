use crate::agent::permission::PermissionPolicy;
use crate::domain::errors::DocumentError;
use crate::security::SharedSandbox;
use crate::security::validate_sandboxed_path_shared;
use rig_core::completion::ToolDefinition;
use rig_core::tool::Tool;
use serde_json::json;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};

/// Validate a glob match path: skip symlinks, enforce sandbox, normalize,
/// and insert into `seen`/`matches` if new. Returns `true` if the match
/// limit has been reached.
fn collect_glob_match(
    path: &Path,
    root: &Path,
    sandbox: &SharedSandbox,
    seen: &mut HashSet<String>,
    matches: &mut Vec<String>,
    max_results: usize,
) -> Result<bool, std::io::Error> {
    if let Ok(meta) = std::fs::symlink_metadata(path)
        && meta.file_type().is_symlink()
    {
        return Ok(matches.len() >= max_results);
    }
    validate_sandboxed_path_shared(sandbox, path)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::PermissionDenied, e.to_string()))?;

    let relative = path.strip_prefix(root).unwrap_or(path);
    let normalized = relative.to_string_lossy().replace('\\', "/");

    if seen.insert(normalized.clone()) {
        matches.push(normalized);
    }
    Ok(matches.len() >= max_results)
}

/// Arguments for the `glob_search` tool.
#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct GlobSearchArgs {
    /// Glob pattern to match (e.g. `"src/**/*.rs"`, `"*.md"`).
    pub pattern: String,
    /// Optional directory to search within (relative to sandbox root).
    /// When provided, the pattern is matched relative to this directory
    /// instead of the sandbox root. The directory must exist and be
    /// within the sandbox.
    pub directory: Option<String>,
}

/// Finds files and directories matching a glob pattern within the sandbox root(s).
///
/// Uses the [`glob`] crate for pattern matching. Rejects absolute patterns
/// and path traversals containing `..`. Returns up to 100 matches with
/// forward-slash-normalized paths relative to the sandbox root.
/// When multiple roots are configured, searches all roots and deduplicates.
#[derive(Debug, Clone)]
pub struct GlobSearchTool {
    sandbox: Arc<SharedSandbox>,
    policy: PermissionPolicy,
}

static GLOB_DEF: LazyLock<ToolDefinition> = LazyLock::new(|| {
    ToolDefinition {
    name: "glob_search".to_string(),
    description: "Find files and directories matching a glob pattern (e.g., 'src/**/*.rs' or '*.md') within the sandbox root(s). Use 'directory' to narrow the search to a specific subdirectory.".to_string(),
    parameters: json!({
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "The glob pattern to match against (relative to sandbox root, e.g. 'src/**/*.rs')"
            },
            "directory": {
                "type": "string",
                "description": "Optional directory to search within (relative to sandbox root). When provided, the pattern is matched relative to this directory."
            }
        },
        "required": ["pattern"]
    }),
}
});

impl GlobSearchTool {
    /// Creates a new `GlobSearchTool` restricted to the given sandbox.
    ///
    /// # Arguments
    ///
    /// * `sandbox` - The sandbox configuration containing allowed roots.
    /// * `policy` - The permission policy to evaluate before glob searching.
    ///
    /// # Returns
    ///
    /// Returns the initialized `GlobSearchTool`.
    pub fn new(sandbox: Arc<SharedSandbox>, policy: PermissionPolicy) -> Self {
        Self { sandbox, policy }
    }
}

impl Tool for GlobSearchTool {
    const NAME: &'static str = "glob_search";

    type Error = DocumentError;
    type Args = GlobSearchArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        GLOB_DEF.clone()
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let description = format!(
            "Wants to match glob pattern '{}' in [{}]",
            args.pattern,
            args.directory.as_deref().unwrap_or(".")
        );

        let pattern = &args.pattern;

        // Safety: Reject absolute patterns or path traversals containing '..'
        if Path::new(pattern).is_absolute() || pattern.split('/').any(|segment| segment == "..") {
            return Err(DocumentError::SandboxEscape(format!(
                "Access denied: Absolute patterns or path traversals containing '..' are not allowed: {}",
                pattern
            )));
        }

        // In-sandbox access is auto-allowed; the gate is consulted only for
        // out-of-sandbox directories. When no `directory` is given the search
        // iterates the sandbox roots themselves, so it is inherently in-sandbox.
        let dir_path: Option<PathBuf> = if let Some(ref directory) = args.directory {
            match crate::security::validate_sandboxed_path_shared(
                &self.sandbox,
                Path::new(directory),
            ) {
                Ok(resolved) => Some(resolved),
                Err(_) => {
                    self.sandbox
                        .check_permission(&self.policy, Self::NAME, &description)
                        .await?;
                    Some(self.sandbox.resolve_path_unchecked(Path::new(directory)))
                }
            }
        } else {
            None
        };

        let mut matches = Vec::new();
        let mut seen = HashSet::new();
        let max_results = 100;

        if let Some(dir_path) = dir_path {
            // Search within a specific (resolved) directory.
            let full_pattern = dir_path.join(pattern);
            let pattern_str = full_pattern.to_string_lossy();

            let paths = glob::glob(&pattern_str).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string())
            })?;

            for entry in paths {
                let path = entry.map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::PermissionDenied, e.to_string())
                })?;
                if collect_glob_match(
                    &path,
                    &dir_path,
                    &self.sandbox,
                    &mut seen,
                    &mut matches,
                    max_results,
                )? {
                    break;
                }
            }
        } else {
            // No directory specified — search from all sandbox roots
            let snapshot = self.sandbox.snapshot();
            'roots: for canonical_root in snapshot.canonical_roots() {
                let full_pattern = canonical_root.join(pattern);
                let pattern_str = full_pattern.to_string_lossy();

                let paths = glob::glob(&pattern_str).map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string())
                })?;

                for entry in paths {
                    let path = entry.map_err(|e| {
                        std::io::Error::new(std::io::ErrorKind::PermissionDenied, e.to_string())
                    })?;
                    if collect_glob_match(
                        &path,
                        canonical_root,
                        &self.sandbox,
                        &mut seen,
                        &mut matches,
                        max_results,
                    )? {
                        break 'roots;
                    }
                }
            }
        }

        if matches.is_empty() {
            Ok(format!("No files matched pattern: '{}'", pattern))
        } else {
            let count = matches.len();
            let mut output = matches.join("\n");
            if count >= max_results {
                output.push_str(&format!(
                    "\n... [Truncated: reached limit of {} matches]",
                    max_results
                ));
            }
            Ok(output)
        }
    }
}
