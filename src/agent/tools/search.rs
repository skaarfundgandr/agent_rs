use crate::agent::permission::PermissionPolicy;
use crate::agent::tools::document::validate_sandboxed_path;
use crate::domain::errors::DocumentError;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde_json::json;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

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
    sandbox_root: PathBuf,
    allowed_extensions: HashSet<String>,
    policy: PermissionPolicy,
}

impl GrepSearchTool {
    /// Creates a new `GrepSearchTool` restricted to `sandbox_root` and the given extension allowlist.
    pub fn new(
        sandbox_root: impl Into<PathBuf>,
        allowed_extensions: HashSet<String>,
        policy: PermissionPolicy,
    ) -> Self {
        Self {
            sandbox_root: sandbox_root.into(),
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
                "Search for a pattern/substring in workspace text files. Supports extensions: {supported}. Paths are relative to sandbox root."
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
        if !self.policy.evaluate(Self::NAME, &description).await {
            return Err(DocumentError::PermissionDenied(description));
        }

        let path = validate_sandboxed_path(&self.sandbox_root, Path::new(&relative_path))?;

        let case_sensitive = args.case_sensitive.unwrap_or(false);
        let max_results = 100;
        let mut results = Vec::new();

        search_recursive(
            &path,
            &args.query,
            case_sensitive,
            &self.allowed_extensions,
            &self.sandbox_root,
            &mut results,
            max_results,
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
fn search_recursive(
    target: &Path,
    query: &str,
    case_sensitive: bool,
    allowed_extensions: &HashSet<String>,
    sandbox_root: &Path,
    results: &mut Vec<String>,
    max_results: usize,
) -> Result<(), std::io::Error> {
    if results.len() >= max_results {
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
                    sandbox_root,
                    results,
                    max_results,
                )?;
            } else {
                search_file(
                    &path,
                    query,
                    case_sensitive,
                    allowed_extensions,
                    sandbox_root,
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
            sandbox_root,
            results,
            max_results,
        )?;
    }

    Ok(())
}
// TODO: Refactor: Reduce nesting for this function
/// Searches a single file for `query`, appending matched lines to `results`.
fn search_file(
    file_path: &Path,
    query: &str,
    case_sensitive: bool,
    allowed_extensions: &HashSet<String>,
    sandbox_root: &Path,
    results: &mut Vec<String>,
    max_results: usize,
) -> Result<(), std::io::Error> {
    if results.len() >= max_results {
        return Ok(());
    }

    if let Some(ext) = file_path.extension().and_then(|e| e.to_str()) {
        if allowed_extensions.contains(ext) {
            if let Ok(content) = fs::read_to_string(file_path) {
                let relative_path = file_path
                    .strip_prefix(sandbox_root)
                    .unwrap_or(file_path)
                    .to_string_lossy()
                    .into_owned();

                let query_lower = if !case_sensitive {
                    Some(query.to_lowercase())
                } else {
                    None
                };

                for (line_num, line) in content.lines().enumerate() {
                    let matches = if case_sensitive {
                        line.contains(query)
                    } else {
                        line.to_lowercase().contains(query_lower.as_ref().unwrap())
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
            }
        }
    }

    Ok(())
}
