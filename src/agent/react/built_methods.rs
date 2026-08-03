use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use rig_core::completion::{Prompt, PromptError};
use rig_core::message::{AssistantContent, Message, ToolResultContent, UserContent};
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

use crate::agent::invalid_tool::InvalidToolPolicy;
use crate::agent::memory::ContextManager;
use crate::agent::react::Compact;
use crate::agent::telemetry::TelemetryAccum;
use crate::domain::agent::{Action, DetailsState, Observation, ReActStep, ReActTrace};
use crate::domain::errors::ReActError;

use super::built::{BuiltReAct, run_loop};
use super::callbacks::{ActionCb, ObservationCb};
use super::emitter::ReActSpanEmitter;

pub(crate) fn effective_prompt(preamble: &Option<String>, prompt: &str) -> String {
    match preamble {
        Some(p) => format!("{p}\n\n{prompt}"),
        None => prompt.to_string(),
    }
}

impl<C> Compact for ContextManager<C>
where
    C: Prompt + WasmCompatSend + WasmCompatSync + 'static,
{
    fn compact<'a>(
        &'a self,
        history: &'a mut Vec<Message>,
        prompt: &'a str,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<bool, PromptError>> + Send + 'a>> {
        Box::pin(async move { self.compact_history_if_needed(history, prompt).await })
    }
}

impl<M, C, S> BuiltReAct<M, C, S>
where
    M: rig_core::completion::CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    S: DetailsState,
{
    pub fn max_cycles(&self) -> usize {
        self.max_cycles
    }

    pub fn max_retries(&self) -> u32 {
        self.max_retries
    }

    pub fn react_preamble(&self) -> Option<&str> {
        self.react_preamble.as_deref()
    }

    pub fn invalid_tool_policy(&self) -> InvalidToolPolicy {
        self.invalid_tool_policy
    }

    pub fn max_invalid_tool_call_retries(&self) -> u32 {
        self.max_invalid_tool_call_retries
    }
}

/// Emit `on_action` / `on_observation` callbacks for tool calls and tool
/// results that rig-core executed internally before returning from
/// `agent.prompt()`. The existing `run_loop` logic only inspects the *last*
/// assistant message, so intermediate tool turns are otherwise lost.
#[doc(hidden)]
pub fn emit_internal_tool_callbacks(
    messages: &[Message],
    cycle: usize,
    on_action: &Option<ActionCb>,
    on_observation: &Option<ObservationCb>,
    span_emitter: &Arc<dyn ReActSpanEmitter>,
    trace: &mut ReActTrace,
) {
    let Some(last_assistant_idx) = messages
        .iter()
        .rposition(|msg| matches!(msg, Message::Assistant { .. }))
    else {
        return;
    };

    // Map tool_call_id → tool name so observations are attributed to the
    // correct action even when a provider returns tool results out of order.
    let mut pending_tool_names: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    // Skip the first message: it is the prompt passed to `agent.prompt()`.
    // In later ReAct cycles that prompt is the previous cycle's last tool
    // result, whose action/observation were already emitted.
    for msg in &messages[1..last_assistant_idx] {
        match msg {
            Message::Assistant { content, .. } => {
                for item in content.iter() {
                    if let AssistantContent::ToolCall(tc) = item {
                        pending_tool_names.insert(tc.id.clone(), tc.function.name.clone());
                        let action = Action {
                            tool_name: tc.function.name.clone(),
                            args: tc.function.arguments.to_string(),
                            tool_call_id: Some(tc.id.clone()),
                            cycle,
                        };
                        if let Some(cb) = on_action {
                            cb(&action);
                        }
                        span_emitter.emit_action(&action);
                        trace.steps.push(ReActStep::Action(action));
                    }
                }
            }
            Message::User { content } => {
                for item in content.iter() {
                    if let UserContent::ToolResult(tr) = item {
                        let result_text = tr
                            .content
                            .iter()
                            .filter_map(|c| match c {
                                ToolResultContent::Text(t) => Some(t.text.as_str()),
                                _ => None,
                            })
                            .collect::<String>();
                        let tool_name = pending_tool_names
                            .remove(&tr.id)
                            .unwrap_or_else(|| "unknown".to_string());
                        let observation = Observation {
                            tool_name,
                            result: result_text,
                            is_error: false,
                            cycle,
                            duration: Duration::from_secs(0),
                        };
                        if let Some(cb) = on_observation {
                            cb(&observation);
                        }
                        span_emitter.emit_observation(&observation);
                        trace.steps.push(ReActStep::Observation(observation));
                    }
                }
            }
            _ => {}
        }
    }
}

impl<M, C, S> BuiltReAct<M, C, S>
where
    M: rig_core::completion::CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    S: DetailsState,
{
    pub(crate) async fn run_prompt_impl(
        &self,
        msg: String,
        details: Option<&mut TelemetryAccum>,
    ) -> Result<ReActTrace, ReActError> {
        let (trace, _) = run_loop(
            &self.agent,
            &msg,
            &[],
            self.max_cycles,
            self.max_retries,
            self.max_invalid_tool_call_retries,
            self.tool_timeout_secs,
            &self.react_preamble,
            &self.span_emitter,
            &self.on_thought,
            &self.on_action,
            &self.on_observation,
            &self.on_final,
            &self.on_error,
            self.context_manager.as_deref(),
            &self.cycle_limit_reminder_msg,
            details,
        )
        .await?;
        Ok(trace)
    }

    pub(crate) async fn run_chat_impl(
        &self,
        msg: String,
        history: &[Message],
        details: Option<&mut TelemetryAccum>,
    ) -> Result<(String, Vec<Message>), ReActError> {
        let (trace, working) = run_loop(
            &self.agent,
            &msg,
            history,
            self.max_cycles,
            self.max_retries,
            self.max_invalid_tool_call_retries,
            self.tool_timeout_secs,
            &self.react_preamble,
            &self.span_emitter,
            &self.on_thought,
            &self.on_action,
            &self.on_observation,
            &self.on_final,
            &self.on_error,
            self.context_manager.as_deref(),
            &self.cycle_limit_reminder_msg,
            details,
        )
        .await?;
        Ok((
            trace.final_answer.map(|fa| fa.text).unwrap_or_default(),
            working,
        ))
    }
}

impl<M, C, S> BuiltReAct<M, C, S>
where
    M: rig_core::completion::CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    M::StreamingResponse: rig_core::completion::GetTokenUsage + Send,
    C: Send + Sync + 'static,
    S: DetailsState,
{
    pub(crate) fn make_stream_shared(&self) -> Arc<super::streaming::StreamShared<M, C>> {
        Arc::new(super::streaming::StreamShared {
            agent: self.agent.clone(),
            max_invalid_tool_call_retries: self.max_invalid_tool_call_retries,
            tool_timeout_secs: self.tool_timeout_secs,
            on_thought: self.on_thought.as_ref().map(Arc::clone),
            on_action: self.on_action.as_ref().map(Arc::clone),
            on_observation: self.on_observation.as_ref().map(Arc::clone),
            on_final: self.on_final.as_ref().map(Arc::clone),
            on_error: self.on_error.as_ref().map(Arc::clone),
            context_manager: self.context_manager.clone(),
            _compaction: PhantomData,
        })
    }

    pub(crate) fn run_stream_impl<'h>(
        &self,
        msg: String,
    ) -> Result<super::streaming::ReActStream<'h, M, C>, ReActError> {
        Ok(super::streaming::ReActStream::new(
            self.make_stream_shared(),
            Vec::new(),
            self.max_cycles,
            self.max_retries,
            self.react_preamble.clone(),
            Arc::clone(&self.span_emitter),
            msg,
            None,
        ))
    }
}
