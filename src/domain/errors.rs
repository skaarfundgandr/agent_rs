use thiserror::Error;

/// Errors that can occur during document I/O, sandbox validation, or RAG operations.
#[derive(Debug, Error)]
pub enum DocumentError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("PDF extraction error: {0}")]
    Pdf(String),
    #[error("Unsupported file extension: {0}")]
    UnsupportedExtension(String),
    #[error("Sandbox escape error: {0}")]
    SandboxEscape(String),
    #[error("Sandbox error: {0}")]
    Sandbox(String),
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    #[error("RAG error: {0}")]
    Rag(String),
}

/// Errors that can occur during conversation history compaction.
#[derive(Debug, Error)]
pub enum CompactError {
    #[error("Model error: {0}")]
    Model(String),
}
