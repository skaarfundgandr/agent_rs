use crate::agent::rag::extract_pdf_text;
use crate::domain::errors::DocumentError;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde_json::json;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

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

#[derive(Debug, Clone)]
pub struct ReadDocumentTool {
    sandbox_root: PathBuf,
    allowed_extensions: HashSet<String>,
}

impl ReadDocumentTool {
    pub fn new(sandbox_root: impl Into<PathBuf>, allowed_extensions: HashSet<String>) -> Self {
        Self {
            sandbox_root: sandbox_root.into(),
            allowed_extensions,
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
                "Read the content of a document or text file. Supports: {supported}. Paths are resolved within the configured sandbox root."
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
        let path = validate_sandboxed_path(&self.sandbox_root, Path::new(&args.path))?;
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

#[derive(Debug, Clone)]
pub struct WriteDocumentTool {
    sandbox_root: PathBuf,
    allowed_extensions: HashSet<String>,
}

impl WriteDocumentTool {
    pub fn new(sandbox_root: impl Into<PathBuf>, allowed_extensions: HashSet<String>) -> Self {
        Self {
            sandbox_root: sandbox_root.into(),
            allowed_extensions,
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
                "Write or edit a document or text file. Supports: {supported}. Paths are resolved within the configured sandbox root."
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
        let path = validate_sandboxed_path(&self.sandbox_root, Path::new(&args.path))?;
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

pub(crate) fn validate_sandboxed_path(
    sandbox_root: &Path,
    user_path: &Path,
) -> Result<PathBuf, DocumentError> {
    let canonical_root = sandbox_root
        .canonicalize()
        .map_err(|e| DocumentError::Io(e))?;

    let target = sandbox_root.join(user_path);

    let canonical_target = if target.exists() {
        target.canonicalize().map_err(|e| DocumentError::Io(e))?
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

        let mut canonical_path = existing_parent
            .canonicalize()
            .map_err(|e| DocumentError::Io(e))?;
        for comp in components_to_append.into_iter().rev() {
            canonical_path.push(comp);
        }
        canonical_path
    };

    if !canonical_target.starts_with(&canonical_root) {
        return Err(DocumentError::SandboxEscape(format!(
            "Access denied: Path escapes sandbox: {}",
            user_path.display()
        )));
    }

    Ok(canonical_target)
}
