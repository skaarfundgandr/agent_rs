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

/// Errors that can occur during a ReAct loop execution.
#[derive(Debug, Error)]
pub enum ReActError {
    #[error("ReAct loop exceeded max_cycles ({cycles}) without a final answer")]
    MaxCyclesExceeded { cycles: usize },
    #[error("Tool execution error for '{tool}': {source}")]
    ToolExecution {
        tool: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("Model error: {0}")]
    Model(String),
    #[error("Model returned neither a tool call nor a final answer in cycle {cycle}")]
    NoToolCallsAndNoFinalAnswer { cycle: usize },
}
