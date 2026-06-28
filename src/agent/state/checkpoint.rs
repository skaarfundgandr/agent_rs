use std::collections::HashMap;

use rig_core::message::Message;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Current checkpoint schema version. Load rejects versions that don't match.
pub const CURRENT_SCHEMA_VERSION: u32 = 1;

/// Serializable conversation checkpoint for long-running tasks.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AgentCheckpoint {
    /// Full conversation history at checkpoint time.
    pub history: Vec<Message>,
    /// Compacted context produced by a [`ContextManager`](crate::agent::memory::ContextManager), if any.
    pub compacted_context: Option<String>,
    /// Application-defined phase label (e.g. "research", "draft", "review").
    pub phase: String,
    /// Free-form partial results accumulated by the application.
    pub partial_results: HashMap<String, Value>,
    /// Checkpoint metadata.
    pub metadata: CheckpointMetadata,
}

/// Metadata describing when and how a checkpoint was created.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CheckpointMetadata {
    /// Seconds since UNIX epoch, formatted as an RFC3339-like string.
    pub created_at: String,
    /// Number of ReAct cycles completed at checkpoint time.
    pub cycles_completed: usize,
    /// Schema version for forward/backward compatibility.
    pub schema_version: u32,
}

impl Default for AgentCheckpoint {
    fn default() -> Self {
        Self {
            history: Vec::new(),
            compacted_context: None,
            phase: String::new(),
            partial_results: HashMap::new(),
            metadata: CheckpointMetadata {
                created_at: now_timestamp(),
                cycles_completed: 0,
                schema_version: CURRENT_SCHEMA_VERSION,
            },
        }
    }
}

/// Format the current wall-clock time as a seconds-since-epoch string.
///
/// This is deliberately a lightweight, zero-dependency alternative to a full
/// RFC3339 formatter. The value is metadata-only and not precision-critical.
/// Format: `"<seconds_since_epoch>s"`.
pub(crate) fn now_timestamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}s")
}
