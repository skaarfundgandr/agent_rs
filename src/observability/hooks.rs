//! OTel attribute enrichment for rig's `PromptHook<M>` lifecycle.
//!
//! Records LangSmith run-typing attributes (`langsmith.span.kind`,
//! `openinference.span.kind`) and input/output messages onto rig's
//! already-open spans via `tracing::Span::current().record(...)`.
//!
//! rig 0.39 natively emits `gen_ai.operation.name`, `gen_ai.usage.*`, and
//! `gen_ai.tool.name` — these are NOT recorded here to avoid duplication
//! and the `gen_ai.operation.name` overwrite bug (where the hook would
//! overwrite the `invoke_agent` span's correct value with "chat").

use rig_core::agent::{HookAction, PromptHook, ToolCallHookAction};
use rig_core::completion::{CompletionModel, CompletionResponse};
use rig_core::message::Message;

use crate::observability::conventions::*;

/// A [`PromptHook`] impl that enriches rig's GenAI spans with LangSmith
/// run-typing attributes and input/output messages.
///
/// rig 0.39 natively emits `gen_ai.operation.name`, `gen_ai.usage.*`, and
/// `gen_ai.tool.name`. This hook only adds LangSmith/OpenInference-specific
/// attributes (`langsmith.span.kind`, `openinference.span.kind`, `input.value`,
/// `output.value`) and fills `gen_ai.input.messages`/`gen_ai.output.messages`
/// which rig declares but does not populate.
#[derive(Debug, Default, Clone, Copy)]
pub struct LangSmithAgentHook;

impl<M> PromptHook<M> for LangSmithAgentHook
where
    M: CompletionModel,
{
    async fn on_completion_call(&self, prompt: &Message, history: &[Message]) -> HookAction {
        let span = tracing::Span::current();
        // LangSmith/OpenInference run-typing (NOT emitted by rig)
        span.record(LANGSMITH_SPAN_KIND, KIND_LLM);
        span.record(OPENINFERENCE_SPAN_KIND, "LLM");
        // rig declares gen_ai.input.messages on the chat span but never fills it.
        // Record it here on the invoke_agent span (what Span::current() sees).
        let mut messages = history.to_vec();
        messages.push(prompt.clone());
        if let Ok(serialized) = serde_json::to_string(&messages) {
            span.record("gen_ai.input.messages", serialized.as_str());
        }
        HookAction::cont()
    }

    async fn on_completion_response(
        &self,
        _prompt: &Message,
        resp: &CompletionResponse<M::Response>,
    ) -> HookAction {
        let span = tracing::Span::current();
        // rig declares gen_ai.output.messages on the chat span but never fills it.
        let assistant_msg = Message::Assistant {
            id: resp.message_id.clone(),
            content: resp.choice.clone(),
        };
        let output_messages = vec![assistant_msg];
        if let Ok(serialized) = serde_json::to_string(&output_messages) {
            span.record("gen_ai.output.messages", serialized.as_str());
        }
        HookAction::cont()
    }

    async fn on_tool_call(
        &self,
        _name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        args: &str,
    ) -> ToolCallHookAction {
        let span = tracing::Span::current();
        // LangSmith/OpenInference run-typing (NOT emitted by rig)
        span.record(LANGSMITH_SPAN_KIND, KIND_TOOL);
        span.record(OPENINFERENCE_SPAN_KIND, "TOOL");
        span.record(INPUT_VALUE, args);
        // rig records gen_ai.tool.name natively on the execute_tool span
        ToolCallHookAction::cont()
    }

    async fn on_tool_result(
        &self,
        _name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        _args: &str,
        result: &str,
    ) -> HookAction {
        let span = tracing::Span::current();
        span.record(OUTPUT_VALUE, result);
        HookAction::cont()
    }
}
