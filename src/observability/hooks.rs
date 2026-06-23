//! OTel attribute enrichment for rig's `PromptHook<M>` lifecycle.
//!
//! Records LangSmith run-typing attributes (`langsmith.span.kind`,
//! `openinference.span.kind`) and token-usage counters onto rig's
//! already-open `chat` / `execute_tool` spans via
//! `tracing::Span::current().record(...)`.

use rig_core::agent::{HookAction, PromptHook, ToolCallHookAction};
use rig_core::completion::{CompletionModel, CompletionResponse};
use rig_core::message::Message;

use crate::observability::conventions::*;

/// A [`PromptHook`] impl that enriches rig's GenAI spans with LangSmith
/// run-typing attributes and token-usage counters.
#[derive(Debug, Default, Clone, Copy)]
pub struct LangSmithAgentHook;

impl<M> PromptHook<M> for LangSmithAgentHook
where
    M: CompletionModel,
{
    async fn on_completion_call(&self, prompt: &Message, history: &[Message]) -> HookAction {
        let span = tracing::Span::current();
        span.record(LANGSMITH_SPAN_KIND, KIND_LLM);
        span.record(OPENINFERENCE_SPAN_KIND, "LLM");
        span.record(GEN_AI_OPERATION_NAME, "chat");

        // Record the prompt and history messages to gen_ai.input.messages
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
        span.record(GEN_AI_USAGE_INPUT_TOKENS, resp.usage.input_tokens);
        span.record(GEN_AI_USAGE_OUTPUT_TOKENS, resp.usage.output_tokens);

        // Record the assistant's output message to gen_ai.output.messages
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
        name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        args: &str,
    ) -> ToolCallHookAction {
        let span = tracing::Span::current();
        span.record(LANGSMITH_SPAN_KIND, KIND_TOOL);
        span.record(OPENINFERENCE_SPAN_KIND, "TOOL");
        span.record(GEN_AI_TOOL_NAME, name);
        span.record(INPUT_VALUE, args);
        ToolCallHookAction::cont()
    }

    async fn on_tool_result(
        &self,
        name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        _args: &str,
        result: &str,
    ) -> HookAction {
        let span = tracing::Span::current();
        span.record(GEN_AI_TOOL_NAME, name);
        span.record(OUTPUT_VALUE, result);
        HookAction::cont()
    }
}
