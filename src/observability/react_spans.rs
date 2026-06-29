//! OTel span emission for ReAct-loop cycles.
//!
//! Each emitter method creates a **dedicated child span** for the ReAct event
//! (cycle, thought, action, observation). This keeps the top-level agent span
//! — created by the caller with `input.value` = the user's prompt — untouched:
//! tool args/results go on their own TOOL child spans, not the parent.
//!
//! `emit_error` is the exception: it marks the current (parent) span as
//! errored, which is the desired behaviour for a failed ReAct run.

use crate::agent::react::ReActSpanEmitter;
use crate::domain::agent::{Action, Observation, ReActTrace, Thought};
use crate::domain::errors::ReActError;
use crate::observability::conventions::*;
use opentelemetry::trace::Status;
use tracing_opentelemetry::OpenTelemetrySpanExt;

/// OTel-aware [`ReActSpanEmitter`] that emits LangSmith-typed child spans for
/// each ReAct lifecycle event.
#[derive(Debug, Default, Clone)]
pub struct LangSmithReActEmitter;

impl ReActSpanEmitter for LangSmithReActEmitter {
    fn emit_thought(&self, thought: &Thought) {
        let span = tracing::info_span!(
            "react_thought",
            "langsmith.span.kind" = KIND_CHAIN,
            "openinference.span.kind" = "CHAIN",
            "gen_ai.operation.name" = "reasoning",
            "gen_ai.content.reasoning" = %thought.reasoning,
            "react.cycle" = thought.cycle as i64,
        );
        let _enter = span.enter();

        tracing::info!(
            cycle = thought.cycle,
            thought = %thought.reasoning,
            "react reasoning thought"
        );
    }

    fn emit_cycle_start(&self, cycle: usize) {
        let span = tracing::info_span!(
            "react_cycle",
            "langsmith.span.kind" = KIND_CHAIN,
            "openinference.span.kind" = "CHAIN",
            "gen_ai.operation.name" = "react_cycle",
            "react.cycle" = cycle as i64,
        );
        let _enter = span.enter();

        tracing::info!(cycle, "react cycle start");
    }

    fn emit_cycle_end(&self, cycle: usize, trace_so_far: &ReActTrace) {
        if tracing::enabled!(tracing::Level::INFO) {
            let serialized = serde_json::to_string(trace_so_far).unwrap_or_else(|_| String::new());
            tracing::info!(
                cycle,
                trace = %serialized,
                "react cycle complete"
            );
        }
    }

    fn emit_action(&self, action: &Action) {
        let span = tracing::info_span!(
            "react_action",
            "langsmith.span.kind" = KIND_AGENT,
            "openinference.span.kind" = "AGENT",
            "gen_ai.operation.name" = "react_action",
            "gen_ai.tool.name" = %action.tool_name,
            "input.value" = %action.args,
            "react.cycle" = action.cycle as i64,
        );
        let _enter = span.enter();

        tracing::info!(
            cycle = action.cycle,
            tool_name = %action.tool_name,
            "react tool action"
        );
    }

    fn emit_observation(&self, observation: &Observation) {
        let span = tracing::info_span!(
            "react_observation",
            "langsmith.span.kind" = KIND_TOOL,
            "openinference.span.kind" = "TOOL",
            "gen_ai.tool.name" = %observation.tool_name,
            "output.value" = %observation.result,
            "react.is_error" = observation.is_error,
            "react.duration_ms" = observation.duration.as_millis() as u64,
            "react.cycle" = observation.cycle as i64,
        );
        let _enter = span.enter();

        if observation.is_error {
            span.set_status(Status::error(format!(
                "Tool '{}' execution failed: {}",
                observation.tool_name, observation.result
            )));
            tracing::error!(
                tool_name = %observation.tool_name,
                error = %observation.result,
                "react tool call error"
            );
        }
    }

    fn emit_error(&self, err: &ReActError) {
        let span = tracing::Span::current();
        span.record("react.is_error", true);
        span.set_status(Status::error(err.to_string()));

        tracing::error!(
            error = %err,
            "react loop execution error"
        );
    }
}
