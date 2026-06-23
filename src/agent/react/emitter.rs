use crate::domain::agent::{Action, Observation, ReActTrace, Thought};

/// Trait for observing per-cycle lifecycle events of a ReAct loop.
///
/// Implement this to wire up OpenTelemetry spans, logging, or any other
/// observability backend. All methods have no-op defaults.
pub trait ReActSpanEmitter: Send + Sync {
    /// Called when the model emits a reasoning thought.
    fn emit_thought(&self, _thought: &Thought) {}
    /// Called at the start of a new cycle, before the prompt is sent.
    fn emit_cycle_start(&self, _cycle: usize) {}
    /// Called at the end of a cycle, after all observations have been recorded.
    fn emit_cycle_end(&self, _cycle: usize, _trace_so_far: &ReActTrace) {}
    /// Called when the model emits a tool call (action).
    fn emit_action(&self, _action: &Action) {}
    /// Called after a tool has been executed and the observation recorded.
    fn emit_observation(&self, _observation: &Observation) {}
}

/// Default no-op span emitter used when none is supplied.
pub(crate) struct NoopSpanEmitter;

impl ReActSpanEmitter for NoopSpanEmitter {}
