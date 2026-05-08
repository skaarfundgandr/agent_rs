use thiserror::Error;

#[derive(Debug, Error)]
pub enum DocumentError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("PDF extraction error: {0}")]
    Pdf(String),
    #[error("Unsupported file extension: {0}")]
    UnsupportedExtension(String),
}

#[derive(Debug, Error)]
pub enum CompactError {
    #[error("Model error: {0}")]
    Model(String),
}
