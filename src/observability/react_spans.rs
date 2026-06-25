//! OTel span emission for ReAct-loop cycles.
//!
//! Since the emitter methods are called from `BuiltReAct::prompt()` / `BuiltReAct::chat()` (which is
//! async), we cannot borrow a `tracing::Span` across an await point. Instead,
//! we **augment the current span** via `Span::current().record(...)` — the
//! rig-emitted `chat` / `execute_tool` span is the parent context, and it is
//! already being exported to OTel by the subscriber layer installed in
//! [`super::langsmith::init_tracing`].

use crate::agent::react::ReActSpanEmitter;
use crate::domain::agent::{Action, Observation, ReActTrace, Thought};
use crate::observability::conventions::*;

/// OTel-aware [`ReActSpanEmitter`] that records LangSmith run-typing
/// attributes on the current `tracing` span.
#[derive(Debug, Default, Clone)]
pub struct LangSmithReActEmitter;

impl ReActSpanEmitter for LangSmithReActEmitter {
    fn emit_thought(&self, thought: &Thought) {
        tracing::info!(
            cycle = thought.cycle,
            thought = %thought.reasoning,
            "react reasoning thought"
        );
    }

    fn emit_cycle_start(&self, cycle: usize) {
        let span = tracing::Span::current();
        span.record(LANGSMITH_SPAN_KIND, KIND_CHAIN);
        span.record(OPENINFERENCE_SPAN_KIND, "CHAIN");
        span.record(GEN_AI_OPERATION_NAME, "react_cycle");
        span.record("react.cycle", cycle as u64);
    }

    fn emit_cycle_end(&self, cycle: usize, trace_so_far: &ReActTrace) {
        let serialized = serde_json::to_string(trace_so_far).unwrap_or_else(|_| String::new());
        tracing::info!(
            cycle,
            trace = %serialized,
            "react cycle complete"
        );
    }

    fn emit_action(&self, action: &Action) {
        let span = tracing::Span::current();
        span.record(LANGSMITH_SPAN_KIND, KIND_AGENT);
        span.record(OPENINFERENCE_SPAN_KIND, "AGENT");
        span.record(GEN_AI_OPERATION_NAME, "react_action");
        span.record(GEN_AI_TOOL_NAME, action.tool_name.as_str());
        span.record(INPUT_VALUE, action.args.as_str());
        span.record("react.cycle", action.cycle as u64);
    }

    fn emit_observation(&self, observation: &Observation) {
        let span = tracing::Span::current();
        span.record(LANGSMITH_SPAN_KIND, KIND_TOOL);
        span.record(OPENINFERENCE_SPAN_KIND, "TOOL");
        span.record(GEN_AI_TOOL_NAME, observation.tool_name.as_str());
        span.record(OUTPUT_VALUE, observation.result.as_str());
        span.record("react.is_error", observation.is_error);
        span.record("react.duration_ms", observation.duration.as_millis() as u64);
    }
}
