pub mod agents;
pub mod dispatch;
#[cfg(feature = "rag")]
pub mod embeddings;
pub mod invalid_tool;
pub mod memory;
pub mod model;
pub mod permission;
pub mod react;
pub(crate) mod retry;
pub mod state;
pub mod tools;

pub use agents::strip_reasoning_from_history;
pub use invalid_tool::{InvalidToolPolicy, InvalidToolRecoveryHook, invalid_tool_feedback};

mod managed;
#[cfg(feature = "rag")]
pub use embeddings::EmbeddingService;
pub use managed::{BuiltManagedAgent, ManagedBuilder, ManagedExt, ManagedStream};
pub use permission::{PermissionGate, PermissionPolicy};
pub use react::{
    BuiltReAct, CompactionConfig, NoCompaction, REACT_PREAMBLE, ReActBuilder, ReActExt,
    ReActSpanEmitter,
};

pub use tools::{
    CompactTool, GlobSearchTool, GrepSearchTool, ListDirectoryTool, ReadDocumentTool,
    ThinkTool, WriteDocumentTool,
};
#[cfg(feature = "rag")]
pub use tools::{ManageRagTool, RagSourceRegistry};
