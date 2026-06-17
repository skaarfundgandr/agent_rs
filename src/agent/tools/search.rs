use crate::agent::permission::PermissionPolicy;
use crate::domain::errors::DocumentError;
use crate::security::{SandboxConfig, SharedSandbox, relative_display_path};
use rig_core::completion::ToolDefinition;
use rig_core::tool::Tool;
use serde_json::json;
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::Arc;

/// Arguments for the `grep_search` tool.
#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct GrepSearchArgs {
    /// The substring query to search for.
    pub query: String,
    /// Relative path to search within (directory or specific file, defaults to sandbox root).
    pub path: Option<String>,
    /// Whether to perform a case-sensitive search (defaults to false).
    pub case_sensitive: Option<bool>,
}

/// Searches for a substring pattern in workspace text files within the sandbox root.
///
/// Only searches files whose extension is in the `allowed_extensions` set.
/// Results are returned in `path:line: content` format, capped at 100 matches.
/// Supports case-insensitive (default) and case-sensitive modes.
#[derive(Debug, Clone)]
pub struct GrepSearchTool {
    sandbox: Arc<SharedSandbox>,
    allowed_extensions: HashSet<String>,
    policy: PermissionPolicy,
}

impl GrepSearchTool {
    /// Creates a new `GrepSearchTool` restricted to the given sandbox and extension allowlist.
    ///
    /// # Arguments
    ///
    /// * `sandbox` - The sandbox configuration containing allowed roots.
    /// * `allowed_extensions` - The set of allowed file extensions (without leading dots).
    /// * `policy` - The permission policy to evaluate before grep searching.
    ///
    /// # Returns
    ///
    /// Returns the initialized `GrepSearchTool`.
    pub fn new(
        sandbox: Arc<SharedSandbox>,
        allowed_extensions: HashSet<String>,
        policy: PermissionPolicy,
    ) -> Self {
        Self {
            sandbox,
            allowed_extensions,
            policy,
        }
    }
}

impl Tool for GrepSearchTool {
    const NAME: &'static str = "grep_search";

    type Error = DocumentError;
    type Args = GrepSearchArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        let supported = self
            .allowed_extensions
            .iter()
            .map(|e| format!(".{e}"))
            .collect::<Vec<_>>()
            .join(", ");

        ToolDefinition {
            name: Self::NAME.to_string(),
            description: format!(
                "Search for a pattern/substring in workspace text files. Supports extensions: {supported}. Paths are relative to sandbox root(s)."
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The substring query to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "The relative path to search within (directory or specific file, defaults to sandbox root)"
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether to perform a case-sensitive search (defaults to false)"
                    }
                },
                "required": ["query"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let relative_path = args.path.clone().unwrap_or_else(|| ".".to_string());
        let description = format!(
            "Wants to search for substring '{}' at [{}]",
            args.query, relative_path
        );
        let path = self
            .sandbox
            .resolve_path_with_permission(&self.policy, Self::NAME, &description, Path::new(&relative_path))
            .await?;

        let case_sensitive = args.case_sensitive.unwrap_or(false);
        let max_results = 100;
        let mut results = Vec::new();

        let snapshot = self.sandbox.snapshot();
        search_recursive(
            &path,
            &args.query,
            case_sensitive,
            &self.allowed_extensions,
            &snapshot,
            &mut results,
            max_results,
            10,
            0,
        )?;

        if results.is_empty() {
            Ok(format!("No matches found for query: '{}'", args.query))
        } else {
            let count = results.len();
            let mut output = results.join("\n");
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

/// Recursively walks `target` and searches each file for `query`.
#[allow(clippy::too_many_arguments)]
fn search_recursive(
    target: &Path,
    query: &str,
    case_sensitive: bool,
    allowed_extensions: &HashSet<String>,
    sandbox: &SandboxConfig,
    results: &mut Vec<String>,
    max_results: usize,
    max_depth: usize,
    current_depth: usize,
) -> Result<(), std::io::Error> {
    if results.len() >= max_results {
        return Ok(());
    }

    if current_depth > max_depth {
        return Ok(());
    }

    if target.is_dir() {
        for entry in fs::read_dir(target)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                search_recursive(
                    &path,
                    query,
                    case_sensitive,
                    allowed_extensions,
                    sandbox,
                    results,
                    max_results,
                    max_depth,
                    current_depth + 1,
                )?;
            } else {
                search_file(
                    &path,
                    query,
                    case_sensitive,
                    allowed_extensions,
                    sandbox,
                    results,
                    max_results,
                )?;
            }
        }
    } else if target.is_file() {
        search_file(
            target,
            query,
            case_sensitive,
            allowed_extensions,
            sandbox,
            results,
            max_results,
        )?;
    }

    Ok(())
}

/// Searches a single file for `query`, appending matched lines to `results`.
fn search_file(
    file_path: &Path,
    query: &str,
    case_sensitive: bool,
    allowed_extensions: &HashSet<String>,
    sandbox: &SandboxConfig,
    results: &mut Vec<String>,
    max_results: usize,
) -> Result<(), std::io::Error> {
    if results.len() >= max_results {
        return Ok(());
    }

    let ext = match file_path.extension().and_then(|e| e.to_str()) {
        Some(ext) => ext,
        None => return Ok(()),
    };
    if !allowed_extensions.contains(ext) {
        return Ok(());
    }

    let content = match fs::read_to_string(file_path) {
        Ok(c) => c,
        Err(_) => return Ok(()),
    };

    let relative_path = relative_display_path(sandbox, file_path);

    let query_lower = if !case_sensitive {
        Some(query.to_lowercase())
    } else {
        None
    };

    for (line_num, line) in content.lines().enumerate() {
        let matches = if case_sensitive {
            line.contains(query)
        } else if let Some(ref ql) = query_lower {
            line.to_lowercase().contains(ql)
        } else {
            false // This should be unreachable since query_lower is only None if case_sensitive is true
        };

        if matches {
            results.push(format!(
                "{}:{}: {}",
                relative_path,
                line_num + 1,
                line.trim()
            ));
            if results.len() >= max_results {
                return Ok(());
            }
        }
    }

    Ok(())
}
