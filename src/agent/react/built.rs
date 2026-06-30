use std::sync::Arc;

use rig_core::agent::{Agent, PromptHook};
use rig_core::completion::CompletionModel;
use rig_core::message::{Message, UserContent};
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

use crate::agent::react::Compact;
use crate::domain::agent::ReActTrace;
use crate::domain::errors::ReActError;

use super::callbacks::{ErrorCb, FinalCb, ThoughtCb};
use super::emitter::ReActSpanEmitter;

pub use super::built_methods::emit_internal_tool_callbacks;

/// A fully configured ReAct agent, ready to run prompts and chats.
///
/// Constructed by calling [`.build()`](super::ReActBuilder::build) on a
/// [`ReActBuilder`](super::ReActBuilder).
pub struct BuiltReAct<M, P, C = ()>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    pub(crate) agent: Agent<M, P>,
    pub(crate) max_cycles: usize,
    pub(crate) max_retries: u32,
    pub(crate) react_preamble: Option<String>,
    pub(crate) span_emitter: Arc<dyn ReActSpanEmitter>,
    pub(crate) on_thought: Option<ThoughtCb>,
    pub(crate) on_action: Option<super::callbacks::ActionCb>,
    pub(crate) on_observation: Option<super::callbacks::ObservationCb>,
    pub(crate) on_final: Option<FinalCb>,
    pub(crate) on_error: Option<ErrorCb>,
    pub(crate) context_manager: Option<Arc<dyn Compact + Send + Sync>>,
    pub(crate) tool_timeout_secs: u64,
    pub(crate) _compaction: std::marker::PhantomData<C>,
}

/// Standalone ReAct loop that works on a local `Vec<Message>` clone.
/// Returns the trace and the final working history on success.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_loop<M, P>(
    agent: &Agent<M, P>,
    prompt: &str,
    history_snapshot: &[Message],
    max_cycles: usize,
    max_retries: u32,
    tool_timeout_secs: u64,
    react_preamble: &Option<String>,
    span_emitter: &Arc<dyn ReActSpanEmitter>,
    on_thought: &Option<ThoughtCb>,
    on_action: &Option<super::callbacks::ActionCb>,
    on_observation: &Option<super::callbacks::ObservationCb>,
    on_final: &Option<FinalCb>,
    on_error: &Option<ErrorCb>,
    context_manager: Option<&(dyn Compact + Send + Sync)>,
) -> Result<(ReActTrace, Vec<Message>), ReActError>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    let mut trace = ReActTrace {
        prompt: prompt.to_string(),
        steps: Vec::new(),
        final_answer: None,
    };

    let effective_prompt = match react_preamble {
        Some(preamble) => format!("{preamble}\n\n{prompt}"),
        None => prompt.to_string(),
    };

    let mut working_history: Vec<Message> = history_snapshot.to_vec();
    let mut current_prompt = Message::User {
        content: rig_core::OneOrMany::one(UserContent::text(effective_prompt)),
    };
    let mut no_assistant_retried = false;
    let mut empty_output_retried = false;

    for cycle in 0..max_cycles {
        span_emitter.emit_cycle_start(cycle);

        super::cycle_compaction::maybe_compact_history(
            context_manager,
            &mut working_history,
            &current_prompt,
            prompt,
        )
        .await?;

        let response = match super::model_call::execute_model_call(
            agent,
            &current_prompt,
            &working_history,
            max_retries,
            cycle,
            span_emitter,
            on_error,
            &trace,
        )
        .await
        {
            super::model_call::ModelCallResult::Ok(resp) => resp,
            super::model_call::ModelCallResult::TurnLimitRecovery {
                recovered_history,
                recovered_prompt,
            } => {
                working_history = recovered_history;
                current_prompt = recovered_prompt;
                continue;
            }
            super::model_call::ModelCallResult::Err(err) => return Err(err),
        };

        if let Some(messages) = response.messages {
            emit_internal_tool_callbacks(
                &messages,
                cycle,
                on_action,
                on_observation,
                span_emitter,
                &mut trace,
            );
            working_history.extend(messages);
        }

        let assistant_content =
            match super::assistant_parse::find_assistant_content(&working_history) {
                Some(content) => content,
                None => {
                    if !no_assistant_retried {
                        no_assistant_retried = true;
                        continue;
                    }
                    let err = ReActError::NoToolCallsAndNoFinalAnswer { cycle };
                    if let Some(cb) = on_error {
                        cb(&err);
                    }
                    span_emitter.emit_error(&err);
                    span_emitter.emit_cycle_end(cycle, &trace);
                    return Err(err);
                }
            };

        let parsed = super::assistant_parse::classify_assistant_content(&assistant_content);

        if parsed.tool_calls.is_empty() {
            let text = response.output.clone();
            if text.is_empty() {
                if !empty_output_retried {
                    empty_output_retried = true;
                    continue;
                }
                let err = ReActError::NoToolCallsAndNoFinalAnswer { cycle };
                if let Some(cb) = on_error {
                    cb(&err);
                }
                span_emitter.emit_error(&err);
                span_emitter.emit_cycle_end(cycle, &trace);
                return Err(err);
            }
            let _fa = super::assistant_parse::emit_final_answer_from_output(
                text,
                cycle,
                &mut trace,
                on_final,
                span_emitter,
            );
            return Ok((trace, working_history));
        }

        let thought_text = parsed.reasoning_texts.join("\n").trim().to_string();
        if !thought_text.is_empty() {
            let thought = crate::domain::agent::Thought {
                reasoning: thought_text,
                cycle,
            };
            if let Some(cb) = on_thought {
                cb(&thought);
            }
            span_emitter.emit_thought(&thought);
            trace
                .steps
                .push(crate::domain::agent::ReActStep::Thought(thought));
        }

        if let Some(text) = super::assistant_parse::try_detect_final_answer(&parsed.trailing_texts)
        {
            let _fa = super::assistant_parse::emit_final_answer_from_output(
                text,
                cycle,
                &mut trace,
                on_final,
                span_emitter,
            );
            return Ok((trace, working_history));
        }

        let dispatch_result = super::tool_dispatch::dispatch_tool_calls(
            agent,
            &parsed.tool_calls,
            cycle,
            tool_timeout_secs,
            on_action,
            on_observation,
            span_emitter,
            &mut trace,
        )
        .await?;

        working_history.extend(dispatch_result.history_extensions);
        if let Some(msg) = dispatch_result.next_prompt {
            current_prompt = msg;
        }

        span_emitter.emit_cycle_end(cycle, &trace);
    }

    let err = ReActError::MaxCyclesExceeded { cycles: max_cycles };
    if let Some(cb) = on_error {
        cb(&err);
    }
    span_emitter.emit_error(&err);
    span_emitter.emit_cycle_end(max_cycles.saturating_sub(1), &trace);
    Err(err)
}
