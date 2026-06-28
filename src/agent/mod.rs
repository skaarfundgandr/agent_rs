pub mod agents;
#[cfg(feature = "rag")]
pub mod embeddings;
pub mod memory;
pub mod model;
pub mod permission;
pub mod react;
pub mod state;
pub mod tools;
pub(crate) mod utils;

pub mod dispatch;

pub use agents::strip_reasoning_from_history;

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
    WriteDocumentTool,
};
#[cfg(feature = "rag")]
pub use tools::{ManageRagTool, RagSourceRegistry};
