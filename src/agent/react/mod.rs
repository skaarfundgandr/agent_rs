//! ReAct (Reasoning + Acting) loop implementation.
//!
//! The loop drives a per-cycle single-turn completion pattern: the model
//! reasons, optionally emits tool calls, the loop executes them via the
//! agent's tool server, feeds observations back, and repeats until the
//! model produces a "Final Answer:" sentinel or `max_cycles` is reached.

use std::pin::Pin;

use rig_core::completion::PromptError;
use rig_core::message::Message;

pub mod assistant_parse;
mod builder;
mod built;
mod built_compaction;
mod built_methods;
mod built_no_compaction;
mod callbacks;
mod constants;
pub mod cycle_compaction;
mod emitter;
mod ext;
mod helpers;
pub mod model_call;
pub mod stream_loop;
pub mod stream_process;
pub mod streaming;
pub mod tool_dispatch;

pub use builder::{CompactionConfig, NoCompaction, ReActBuilder};
pub use built::BuiltReAct;
#[doc(hidden)]
pub use built::emit_internal_tool_callbacks;
pub use callbacks::{ActionCb, ErrorCb, FinalCb, ObservationCb, ThoughtCb};
pub use constants::REACT_PREAMBLE;
pub use emitter::ReActSpanEmitter;
pub use ext::ReActExt;
pub use helpers::{detect_final_answer, recover_turn_limit_history};

/// Type-erased interface for automatic context compaction used by both the
/// synchronous and streaming ReAct loops.
pub(crate) trait Compact: Send + Sync {
    fn compact<'a>(
        &'a self,
        history: &'a mut Vec<Message>,
        prompt: &'a str,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<bool, PromptError>> + Send + 'a>>;
}

/// Errors that can occur during a ReAct loop execution.
///
/// Re-exported here for convenience; the canonical definition lives in
/// [`crate::domain::errors::ReActError`].
pub use crate::domain::errors::ReActError;
