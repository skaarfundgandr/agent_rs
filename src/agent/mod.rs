pub mod agents;
#[cfg(feature = "rag")]
pub mod embeddings;
pub mod memory;
pub mod model;
pub mod permission;
pub mod react;
pub mod tools;

pub use agents::{AgentContextExt, ContextManagedAgent, strip_reasoning_from_history};
#[cfg(feature = "rag")]
pub use embeddings::EmbeddingService;
pub use permission::{PermissionGate, PermissionPolicy};
pub use react::{REACT_PREAMBLE, ReActExt, ReActLoop, ReActSpanEmitter};

pub use tools::{
    CompactTool, GlobSearchTool, GrepSearchTool, ListDirectoryTool, ReadDocumentTool,
    WriteDocumentTool,
};
#[cfg(feature = "rag")]
pub use tools::{ManageRagTool, RagSourceRegistry};
