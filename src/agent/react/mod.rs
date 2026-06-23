//! ReAct (Reasoning + Acting) loop implementation.
//!
//! The loop drives a per-cycle single-turn completion pattern: the model
//! reasons, optionally emits tool calls, the loop executes them via the
//! agent's tool server, feeds observations back, and repeats until the
//! model produces a "Final Answer:" sentinel or `max_cycles` is reached.

mod callbacks;
mod emitter;
mod ext;
mod helpers;
mod react_loop;

pub use callbacks::{ActionCb, ErrorCb, FinalCb, ObservationCb, ThoughtCb};
pub use emitter::ReActSpanEmitter;
pub use ext::ReActExt;
pub use helpers::detect_final_answer;
pub use react_loop::{REACT_PREAMBLE, ReActLoop};

/// Errors that can occur during a ReAct loop execution.
///
/// Re-exported here for convenience; the canonical definition lives in
/// [`crate::domain::errors::ReActError`].
pub use crate::domain::errors::ReActError;
