use crate::agent::permission::PermissionPolicy;
use crate::agent::tools::document::validate_sandboxed_path;
use crate::domain::errors::DocumentError;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde_json::json;
use std::path::{Path, PathBuf};

/// Arguments for the `glob_search` tool.
#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct GlobSearchArgs {
    /// Glob pattern to match (e.g. `"src/**/*.rs"`, `"*.md"`).
    pub pattern: String,
}

/// Finds files and directories matching a glob pattern within the sandbox root.
///
/// Uses the [`glob`] crate for pattern matching. Rejects absolute patterns
/// and path traversals containing `..`. Returns up to 100 matches with
/// forward-slash-normalized paths relative to the sandbox root.
#[derive(Debug, Clone)]
pub struct GlobSearchTool {
    sandbox_root: PathBuf,
    policy: PermissionPolicy,
}

impl GlobSearchTool {
    /// Creates a new `GlobSearchTool` restricted to `sandbox_root`.
    pub fn new(sandbox_root: impl Into<PathBuf>, policy: PermissionPolicy) -> Self {
        Self {
            sandbox_root: sandbox_root.into(),
            policy,
        }
    }
}

impl Tool for GlobSearchTool {
    const NAME: &'static str = "glob_search";

    type Error = DocumentError;
    type Args = GlobSearchArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Find files and directories matching a glob pattern (e.g., 'src/**/*.rs' or '*.md') within the sandbox root.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The glob pattern to match against (relative to sandbox root, e.g. 'src/**/*.rs')"
                    }
                },
                "required": ["pattern"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let description = format!("Wants to match glob pattern '{}'", args.pattern);
        if !self.policy.evaluate(Self::NAME, &description).await {
            return Err(DocumentError::PermissionDenied(description));
        }

        let pattern = &args.pattern;

        // Safety: Reject absolute patterns or path traversals containing '..'
        if pattern.contains("..") || Path::new(pattern).is_absolute() {
            return Err(DocumentError::SandboxEscape(format!(
                "Access denied: Absolute patterns or path traversals containing '..' are not allowed: {}",
                pattern
            )));
        }

        let canonical_root = self
            .sandbox_root
            .canonicalize()
            .map_err(DocumentError::Io)?;
        let full_pattern = self.sandbox_root.join(pattern);
        let pattern_str = full_pattern.to_string_lossy();

        let mut matches = Vec::new();
        let max_results = 100;

        let paths = glob::glob(&pattern_str)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string()))?;

        for entry in paths {
            let path = entry.map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::PermissionDenied, e.to_string())
            })?;

            // Validate sandboxing for each match
            if let Ok(validated_path) = validate_sandboxed_path(&self.sandbox_root, &path) {
                let relative = validated_path
                    .strip_prefix(&canonical_root)
                    .unwrap_or(&validated_path)
                    .to_string_lossy()
                    .into_owned();

                // Normalize path separator to '/' for cross-platform consistency
                let relative_normalized = relative.replace('\\', "/");
                matches.push(relative_normalized);

                if matches.len() >= max_results {
                    break;
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
