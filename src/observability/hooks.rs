//! OTel attribute enrichment for rig's `AgentHook<M>` lifecycle.
//!
//! Records LangSmith run-typing attributes (`langsmith.span.kind`,
//! `openinference.span.kind`) and input/output messages onto rig's
//! already-open spans via `tracing::Span::current().record(...)`.
//!
//! rig 0.40 natively emits `gen_ai.operation.name`, `gen_ai.usage.*`, and
//! `gen_ai.tool.name` — these are NOT recorded here to avoid duplication
//! and the `gen_ai.operation.name` overwrite bug (where the hook would
//! overwrite the `invoke_agent` span's correct value with "chat").

use rig_core::agent::{AgentHook, Flow, HookContext, StepEvent};
use rig_core::completion::CompletionModel;
use rig_core::message::Message;

use crate::observability::conventions::*;

/// An [`AgentHook`] impl that enriches rig's GenAI spans with LangSmith
/// run-typing attributes and input/output messages.
///
/// rig 0.40 natively emits `gen_ai.operation.name`, `gen_ai.usage.*`, and
/// `gen_ai.tool.name`. This hook only adds LangSmith/OpenInference-specific
/// attributes (`langsmith.span.kind`, `openinference.span.kind`, `input.value`,
/// `output.value`) and fills `gen_ai.input.messages`/`gen_ai.output.messages`
/// which rig declares but does not populate.
#[derive(Debug, Default, Clone, Copy)]
pub struct LangSmithAgentHook;

impl<M> AgentHook<M> for LangSmithAgentHook
where
    M: CompletionModel,
{
    async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
        match event {
            StepEvent::CompletionCall { prompt, history, .. } => {
                let span = tracing::Span::current();
                span.record(LANGSMITH_SPAN_KIND, KIND_LLM);
                span.record(OPENINFERENCE_SPAN_KIND, "LLM");
                let mut messages = history.to_vec();
                messages.push(prompt.clone());
                if let Ok(serialized) = serde_json::to_string(&messages) {
                    span.record("gen_ai.input.messages", serialized.as_str());
                }
                Flow::cont()
            }
            StepEvent::CompletionResponse { response, .. } => {
                let span = tracing::Span::current();
                let assistant_msg = Message::Assistant {
                    id: response.message_id.clone(),
                    content: response.choice.clone(),
                };
                let output_messages = vec![assistant_msg];
                if let Ok(serialized) = serde_json::to_string(&output_messages) {
                    span.record("gen_ai.output.messages", serialized.as_str());
                }
                Flow::cont()
            }
            StepEvent::ToolCall { tool_name: _, args, .. } => {
                let span = tracing::Span::current();
                span.record(LANGSMITH_SPAN_KIND, KIND_TOOL);
                span.record(OPENINFERENCE_SPAN_KIND, "TOOL");
                span.record(INPUT_VALUE, args);
                Flow::cont()
            }
            StepEvent::ToolResult { result, .. } => {
                let span = tracing::Span::current();
                span.record(OUTPUT_VALUE, result);
                Flow::cont()
            }
            _ => Flow::cont(),
        }
    }
}
