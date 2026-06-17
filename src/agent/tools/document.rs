use crate::agent::permission::PermissionPolicy;
use crate::domain::errors::DocumentError;
use crate::security::SharedSandbox;
use anyhow::{Context, Result};
use pdf_extract::extract_text;
use rig_core::completion::ToolDefinition;
use rig_core::tool::Tool;
use serde_json::json;
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct ReadDocumentArgs {
    pub path: String,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct WriteDocumentArgs {
    pub path: String,
    pub content: String,
    pub append: Option<bool>,
}

/// Tool for reading document or text files within the sandbox.
#[derive(Debug, Clone)]
pub struct ReadDocumentTool {
    sandbox: Arc<SharedSandbox>,
    allowed_extensions: HashSet<String>,
    policy: PermissionPolicy,
}

impl ReadDocumentTool {
    /// Creates a new `ReadDocumentTool` restricted to the given sandbox and file extensions.
    ///
    /// # Arguments
    ///
    /// * `sandbox` - The sandbox configuration containing allowed roots.
    /// * `allowed_extensions` - The set of allowed file extensions (without leading dots).
    /// * `policy` - The permission policy to evaluate before reading files.
    ///
    /// # Returns
    ///
    /// Returns the initialized `ReadDocumentTool`.
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

impl Tool for ReadDocumentTool {
    const NAME: &'static str = "read_document";

    type Error = DocumentError;
    type Args = ReadDocumentArgs;
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
                "Read the content of a document or text file. Supports: {supported}. Paths are resolved within the configured sandbox root(s)."
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to read (relative to the sandbox root)"
                    }
                },
                "required": ["path"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let description = format!("Wants to read file asset at [{}]", args.path);
        let path = self
            .sandbox
            .resolve_path_with_permission(&self.policy, Self::NAME, &description, Path::new(&args.path))
            .await?;
        let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");

        if !self.allowed_extensions.contains(extension) {
            return Err(DocumentError::UnsupportedExtension(extension.to_string()));
        }

        match extension {
            "pdf" => extract_pdf_text(&path).map_err(|e| DocumentError::Pdf(e.to_string())),
            _ => {
                let content = fs::read_to_string(&path)?;
                Ok(content)
            }
        }
    }
}

/// Tool for writing or editing document/text files within the sandbox.
#[derive(Debug, Clone)]
pub struct WriteDocumentTool {
    sandbox: Arc<SharedSandbox>,
    allowed_extensions: HashSet<String>,
    policy: PermissionPolicy,
}

impl WriteDocumentTool {
    /// Creates a new `WriteDocumentTool` restricted to the given sandbox and file extensions.
    ///
    /// # Arguments
    ///
    /// * `sandbox` - The sandbox configuration containing allowed roots.
    /// * `allowed_extensions` - The set of allowed file extensions (without leading dots).
    /// * `policy` - The permission policy to evaluate before writing files.
    ///
    /// # Returns
    ///
    /// Returns the initialized `WriteDocumentTool`.
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

impl Tool for WriteDocumentTool {
    const NAME: &'static str = "write_document";

    type Error = DocumentError;
    type Args = WriteDocumentArgs;
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
                "Write or edit a document or text file. Supports: {supported}. Paths are resolved within the configured sandbox root(s)."
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to write (relative to the sandbox root)"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write"
                    },
                    "append": {
                        "type": "boolean",
                        "description": "Whether to append to the file instead of overwriting"
                    }
                },
                "required": ["path", "content"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let description = format!("Wants to modify/write file asset at [{}]", args.path);
        let path = self
            .sandbox
            .resolve_path_with_permission(&self.policy, Self::NAME, &description, Path::new(&args.path))
            .await?;
        let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");

        if !self.allowed_extensions.contains(extension) {
            return Err(DocumentError::UnsupportedExtension(extension.to_string()));
        }

        if let Some(true) = args.append {
            let mut options = fs::OpenOptions::new();
            options.append(true).create(true);
            let mut file = options.open(&path)?;
            use std::io::Write;
            file.write_all(args.content.as_bytes())?;
        } else {
            fs::write(&path, args.content)?;
        }

        Ok(format!("Successfully wrote to {}", args.path))
    }
}

/// Extract text content from a PDF file.
///
/// # Arguments
///
/// * `path` - The file path to the PDF document.
///
/// # Returns
///
/// Returns the extracted text as a `Result<String>`.
///
/// # Errors
///
/// Returns an error if the PDF file cannot be parsed or read.
pub fn extract_pdf_text<P: AsRef<Path>>(path: P) -> Result<String> {
    extract_text(path).context("Failed to extract text from PDF")
}
