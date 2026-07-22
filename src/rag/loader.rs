//! Document loaders: turn a file path into a `Document`.

use crate::agent::tools::document::extract_pdf_text;
use crate::rag::Document;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;

/// File extensions indexed by [`crate::rag::RagPipeline`] when no explicit
/// extension set is supplied to `open_or_create`. Lowercase, no leading dot.
pub const DEFAULT_EXTENSIONS: &[&str] = &["txt", "md", "pdf"];

/// Trait for loading documents from the file system.
#[async_trait::async_trait]
pub trait DocumentLoader {
    /// Loads a file and returns a `Document`.
    async fn load(&self, path: &Path) -> Result<Document>;
}

/// Loader for PDF documents.
#[derive(Default, Clone, Copy, Debug)]
pub struct PdfLoader;

impl PdfLoader {
    /// Creates a new `PdfLoader`.
    ///
    /// # Returns
    ///
    /// Returns a new `PdfLoader` instance.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl DocumentLoader for PdfLoader {
    async fn load(&self, path: &Path) -> Result<Document> {
        let path_owned = path.to_path_buf();
        let text = tokio::task::spawn_blocking(move || extract_pdf_text(&path_owned))
            .await
            .map_err(|e| anyhow::anyhow!("spawn_blocking failed: {e}"))??;
        let source_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown")
            .to_string();

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), source_name);
        metadata.insert("file_type".to_string(), "pdf".to_string());

        Ok(Document {
            content: text,
            metadata,
        })
    }
}

/// Loader for plain text and Markdown documents.
#[derive(Default, Clone, Copy, Debug)]
pub struct TextLoader;

impl TextLoader {
    /// Creates a new `TextLoader`.
    ///
    /// # Returns
    ///
    /// Returns a new `TextLoader` instance.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl DocumentLoader for TextLoader {
    async fn load(&self, path: &Path) -> Result<Document> {
        let content = tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("Failed to read text file: {:?}", path))?;

        let source_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown")
            .to_string();

        let file_type = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("txt")
            .to_string();

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), source_name);
        metadata.insert("file_type".to_string(), file_type);

        Ok(Document { content, metadata })
    }
}
