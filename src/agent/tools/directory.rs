use crate::agent::permission::PermissionPolicy;
use crate::domain::errors::DocumentError;
use crate::security::{SandboxConfig, validate_sandboxed_path};
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde_json::json;
use std::fs;
use std::path::Path;

/// Arguments for the `list_directory` tool.
#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct ListDirectoryArgs {
    /// Relative path to list (defaults to sandbox root).
    pub path: Option<String>,
}

/// Lists the contents of a directory within the sandbox root.
///
/// Directories are listed first (prefix `[DIR]`), followed by files
/// (prefix `[FILE]` with byte size). Entries are sorted case-insensitively
/// within each group.
#[derive(Debug, Clone)]
pub struct ListDirectoryTool {
    sandbox: SandboxConfig,
    policy: PermissionPolicy,
}

impl ListDirectoryTool {
    /// Creates a new `ListDirectoryTool` restricted to `sandbox_root`.
    pub fn new(sandbox: SandboxConfig, policy: PermissionPolicy) -> Self {
        Self { sandbox, policy }
    }
}

impl Tool for ListDirectoryTool {
    const NAME: &'static str = "list_directory";

    type Error = DocumentError;
    type Args = ListDirectoryArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "List the contents of a directory relative to the sandbox root."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to list (relative to the sandbox root, defaults to '.' if not provided)"
                    }
                }
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let relative_path = args.path.clone().unwrap_or_else(|| ".".to_string());
        let description = format!("Wants to list directory content at [{relative_path}]");
        if !self.policy.evaluate(Self::NAME, &description).await {
            return Err(DocumentError::PermissionDenied(description));
        }

        let path = validate_sandboxed_path(&self.sandbox, Path::new(&relative_path))?;

        if !path.is_dir() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Path is not a directory: {}", relative_path),
            )
            .into());
        }

        let mut entries = Vec::new();
        for entry in fs::read_dir(&path)? {
            let entry = entry?;
            let file_name = entry.file_name().to_string_lossy().into_owned();
            let metadata = entry.metadata()?;
            let is_dir = metadata.is_dir();

            let entry_str = if is_dir {
                format!("[DIR]  {}", file_name)
            } else {
                format!("[FILE] {} ({} bytes)", file_name, metadata.len())
            };
            entries.push((is_dir, file_name, entry_str));
        }

        // Sort: directories first, then files alphabetically (case-insensitive)
        entries.sort_by(|a, b| match (a.0, b.0) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.1.to_lowercase().cmp(&b.1.to_lowercase()),
        });

        if entries.is_empty() {
            Ok("Directory is empty.".to_string())
        } else {
            let formatted = entries
                .into_iter()
                .map(|(_, _, s)| s)
                .collect::<Vec<_>>()
                .join("\n");
            Ok(formatted)
        }
    }
}
