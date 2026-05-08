use crate::agent::rag::extract_pdf_text;
use crate::domain::errors::DocumentError;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde_json::json;
use std::fs;
use std::path::Path;

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

pub struct ReadDocumentTool;

impl Tool for ReadDocumentTool {
    const NAME: &'static str = "read_document";

    type Error = DocumentError;
    type Args = ReadDocumentArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description:
                "Read the content of a document or text file. Supports .txt, .md, and .pdf."
                    .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to read"
                    }
                },
                "required": ["path"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let path = Path::new(&args.path);
        let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");

        match extension {
            "txt" | "md" => {
                let content = fs::read_to_string(path)?;
                Ok(content)
            }
            "pdf" => extract_pdf_text(path).map_err(|e| DocumentError::Pdf(e.to_string())),
            _ => Err(DocumentError::UnsupportedExtension(extension.to_string())),
        }
    }
}

pub struct WriteDocumentTool;

impl Tool for WriteDocumentTool {
    const NAME: &'static str = "write_document";

    type Error = DocumentError;
    type Args = WriteDocumentArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Write or edit a document or text file.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to write"
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
        let path = Path::new(&args.path);

        if let Some(true) = args.append {
            let mut options = fs::OpenOptions::new();
            options.append(true).create(true);
            let mut file = options.open(path)?;
            use std::io::Write;
            file.write_all(args.content.as_bytes())?;
        } else {
            fs::write(path, args.content)?;
        }

        Ok(format!("Successfully wrote to {}", args.path))
    }
}
