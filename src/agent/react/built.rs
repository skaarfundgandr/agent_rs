use std::any::Any;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Instant;

use rig_core::agent::{Agent, PromptHook};
use rig_core::completion::{CompletionModel, Prompt};
use rig_core::message::{AssistantContent, Message, ToolCall, ToolResultContent, UserContent};
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

use crate::agent::memory::ContextManager;
use crate::agent::utils::{Mutex, lock_mutex};
use crate::domain::agent::{Action, FinalAnswer, Observation, ReActStep, ReActTrace, Thought};
use crate::domain::errors::ReActError;

use super::callbacks::{ActionCb, ErrorCb, FinalCb, ObservationCb, ThoughtCb};
use super::emitter::ReActSpanEmitter;
use super::helpers::{detect_final_answer, tool_error_to_string};

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
    pub(crate) history: Arc<Mutex<Vec<Message>>>,
    pub(crate) max_cycles: usize,
    pub(crate) react_preamble: Option<String>,
    pub(crate) span_emitter: Arc<dyn ReActSpanEmitter>,
    pub(crate) on_thought: Option<ThoughtCb>,
    pub(crate) on_action: Option<ActionCb>,
    pub(crate) on_observation: Option<ObservationCb>,
    pub(crate) on_final: Option<FinalCb>,
    pub(crate) on_error: Option<ErrorCb>,
    pub(crate) context_manager: Option<Arc<dyn Any + Send + Sync>>,
    pub(crate) _compaction: PhantomData<C>,
}

impl<M, P, C> BuiltReAct<M, P, C>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Return a snapshot of the current conversation history.
    pub fn history(&self) -> Vec<Message> {
        lock_mutex(&self.history).clone()
    }

    /// Return the configured `max_cycles` limit.
    pub fn max_cycles(&self) -> usize {
        self.max_cycles
    }
}

/// Standalone ReAct loop that works on a local `Vec<Message>` clone.
/// On success when `append_to_shared_history` is true, appends to the
/// shared history via the Mutex.
#[allow(clippy::too_many_arguments)]
async fn run_loop<M, P>(
    agent: &Agent<M, P>,
    prompt: &str,
    history_snapshot: &[Message],
    max_cycles: usize,
    react_preamble: &Option<String>,
    span_emitter: &Arc<dyn ReActSpanEmitter>,
    on_thought: &Option<ThoughtCb>,
    on_action: &Option<ActionCb>,
    on_observation: &Option<ObservationCb>,
    on_final: &Option<FinalCb>,
    on_error: &Option<ErrorCb>,
    shared_history: &Arc<Mutex<Vec<Message>>>,
    append_to_shared_history: bool,
) -> Result<ReActTrace, ReActError>
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

    for cycle in 0..max_cycles {
        span_emitter.emit_cycle_start(cycle);

        let response = agent
            .prompt(current_prompt.clone())
            .with_history(working_history.iter().cloned())
            .extended_details()
            .await
            .map_err(|e| ReActError::Model(e.to_string()))?;

        if let Some(messages) = response.messages {
            working_history.extend(messages);
        }

        let assistant_content = match working_history.iter().rev().find_map(|msg| match msg {
            Message::Assistant { content, .. } => Some(content),
            _ => None,
        }) {
            Some(content) => content.clone(),
            None => {
                let text = response.output.clone();
                let fa = FinalAnswer {
                    text,
                    cycles: cycle + 1,
                };
                trace.steps.push(ReActStep::FinalAnswer(fa.clone()));
                trace.final_answer = Some(fa.clone());
                if let Some(cb) = on_final {
                    cb(&fa);
                }
                span_emitter.emit_cycle_end(cycle, &trace);
                return Err(ReActError::NoToolCallsAndNoFinalAnswer { cycle });
            }
        };

        let mut reasoning_texts: Vec<String> = Vec::new();
        let mut tool_calls: Vec<&ToolCall> = Vec::new();
        let mut trailing_texts: Vec<String> = Vec::new();
        let mut seen_tool_call = false;

        for item in assistant_content.iter() {
            match item {
                AssistantContent::Reasoning(r) => {
                    let text = r.display_text();
                    if !text.is_empty() {
                        reasoning_texts.push(text);
                    }
                }
                AssistantContent::Text(t) => {
                    if seen_tool_call {
                        trailing_texts.push(t.text.clone());
                    } else {
                        reasoning_texts.push(t.text.clone());
                    }
                }
                AssistantContent::ToolCall(tc) => {
                    seen_tool_call = true;
                    tool_calls.push(tc);
                }
                AssistantContent::Image(_) => {}
            }
        }

        if tool_calls.is_empty() {
            let text = response.output.clone();
            if text.is_empty() {
                let err = ReActError::NoToolCallsAndNoFinalAnswer { cycle };
                if let Some(cb) = on_error {
                    cb(&err);
                }
                span_emitter.emit_cycle_end(cycle, &trace);
                return Err(err);
            }
            let fa = FinalAnswer {
                text,
                cycles: cycle + 1,
            };
            trace.steps.push(ReActStep::FinalAnswer(fa.clone()));
            trace.final_answer = Some(fa.clone());
            if let Some(cb) = on_final {
                cb(&fa);
            }
            span_emitter.emit_cycle_end(cycle, &trace);
            if append_to_shared_history {
                *lock_mutex(shared_history) = working_history;
            }
            return Ok(trace);
        }

        let thought_text = reasoning_texts.join("\n").trim().to_string();
        if !thought_text.is_empty() {
            let thought = Thought {
                reasoning: thought_text,
                cycle,
            };
            if let Some(cb) = on_thought {
                cb(&thought);
            }
            span_emitter.emit_thought(&thought);
            trace.steps.push(ReActStep::Thought(thought));
        }

        let full_trailing = trailing_texts.join("").trim().to_string();
        let final_answer = detect_final_answer(&full_trailing);

        if let Some(text) = final_answer {
            let fa = FinalAnswer {
                text,
                cycles: cycle + 1,
            };
            trace.steps.push(ReActStep::FinalAnswer(fa.clone()));
            trace.final_answer = Some(fa.clone());
            if let Some(cb) = on_final {
                cb(&fa);
            }
            span_emitter.emit_cycle_end(cycle, &trace);
            if append_to_shared_history {
                *lock_mutex(shared_history) = working_history;
            }
            return Ok(trace);
        }

        let mut next_prompt = None;
        let num_tool_calls = tool_calls.len();

        for (i, tc) in tool_calls.iter().enumerate() {
            let args_str = tc.function.arguments.to_string();
            let action = Action {
                tool_name: tc.function.name.clone(),
                args: args_str.clone(),
                tool_call_id: Some(tc.id.clone()),
                cycle,
            };
            if let Some(cb) = on_action {
                cb(&action);
            }
            span_emitter.emit_action(&action);
            trace.steps.push(ReActStep::Action(action.clone()));

            let start = Instant::now();
            let result = agent
                .tool_server_handle
                .call_tool(&tc.function.name, &args_str)
                .await;
            let duration = start.elapsed();

            let observation = match result {
                Ok(s) => Observation {
                    tool_name: tc.function.name.clone(),
                    result: s,
                    is_error: false,
                    cycle,
                    duration,
                },
                Err(e) => Observation {
                    tool_name: tc.function.name.clone(),
                    result: tool_error_to_string(&e),
                    is_error: true,
                    cycle,
                    duration,
                },
            };

            if let Some(cb) = on_observation {
                cb(&observation);
            }
            span_emitter.emit_observation(&observation);
            trace
                .steps
                .push(ReActStep::Observation(observation.clone()));

            let call_id = tc
                .call_id
                .clone()
                .unwrap_or_else(|| format!("react-cycle-{cycle}"));
            let tool_result_content = ToolResultContent::text(observation.result.clone());
            let user_content = UserContent::tool_result_with_call_id(
                tc.id.clone(),
                call_id,
                rig_core::OneOrMany::one(tool_result_content),
            );
            let msg = Message::User {
                content: rig_core::OneOrMany::one(user_content),
            };

            if i == num_tool_calls - 1 {
                next_prompt = Some(msg);
            } else {
                working_history.push(msg);
            }
        }

        if let Some(msg) = next_prompt {
            current_prompt = msg;
        }

        span_emitter.emit_cycle_end(cycle, &trace);
    }

    let err = ReActError::MaxCyclesExceeded { cycles: max_cycles };
    if let Some(cb) = on_error {
        cb(&err);
    }
    Err(err)
}

// ── No-compaction methods ────────────────────────────────────────────────

impl<M, P> BuiltReAct<M, P, ()>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Execute a ReAct prompt **without** mutating shared history.
    pub async fn prompt(&self, msg: impl Into<String>) -> Result<ReActTrace, ReActError> {
        let msg = msg.into();
        let snapshot = lock_mutex(&self.history).clone();
        run_loop(
            &self.agent,
            &msg,
            &snapshot,
            self.max_cycles,
            &self.react_preamble,
            &self.span_emitter,
            &self.on_thought,
            &self.on_action,
            &self.on_observation,
            &self.on_final,
            &self.on_error,
            &self.history,
            false,
        )
        .await
    }

    /// Execute a ReAct chat **with** history mutation on success.
    pub async fn chat(&self, msg: impl Into<String>) -> Result<String, ReActError> {
        let msg = msg.into();
        let snapshot = lock_mutex(&self.history).clone();
        let trace = run_loop(
            &self.agent,
            &msg,
            &snapshot,
            self.max_cycles,
            &self.react_preamble,
            &self.span_emitter,
            &self.on_thought,
            &self.on_action,
            &self.on_observation,
            &self.on_final,
            &self.on_error,
            &self.history,
            true,
        )
        .await?;
        Ok(trace.final_answer.map(|fa| fa.text).unwrap_or_default())
    }
}

// ── No-compaction streaming methods ──────────────────────────────────────

impl<M, P> BuiltReAct<M, P, ()>
where
    M: CompletionModel
        + rig_core::streaming::StreamingChat<M, M::StreamingResponse>
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
    P: PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
    M::StreamingResponse: rig_core::completion::GetTokenUsage + Send,
{
    /// Stream a ReAct prompt. Does **not** mutate shared history.
    pub fn stream_prompt(
        &self,
        msg: impl Into<String>,
    ) -> Result<super::streaming::ReActStream<M, P, ()>, ReActError> {
        let msg = msg.into();
        let snapshot = lock_mutex(&self.history).clone();
        Ok(super::streaming::ReActStream::new(
            Arc::new(super::streaming::StreamShared {
                agent: self.agent.clone(),
                history: Arc::clone(&self.history),
                _compaction: std::marker::PhantomData,
            }),
            snapshot,
            self.max_cycles,
            self.react_preamble.clone(),
            Arc::clone(&self.span_emitter),
            false,
            msg,
        ))
    }

    /// Stream a ReAct chat. Mutates shared history on completion.
    pub fn stream_chat(
        &self,
        msg: impl Into<String>,
    ) -> Result<super::streaming::ReActStream<M, P, ()>, ReActError> {
        let msg = msg.into();
        let snapshot = lock_mutex(&self.history).clone();
        Ok(super::streaming::ReActStream::new(
            Arc::new(super::streaming::StreamShared {
                agent: self.agent.clone(),
                history: Arc::clone(&self.history),
                _compaction: std::marker::PhantomData,
            }),
            snapshot,
            self.max_cycles,
            self.react_preamble.clone(),
            Arc::clone(&self.span_emitter),
            true,
            msg,
        ))
    }
}

// ── With-compaction methods ──────────────────────────────────────────────

impl<M, P, C> BuiltReAct<M, P, C>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
    C: Prompt + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Downcast the type-erased context manager back to `&ContextManager<C>`.
    fn context_manager(&self) -> Option<&ContextManager<C>> {
        self.context_manager
            .as_ref()
            .and_then(|arc| arc.downcast_ref::<ContextManager<C>>())
    }

    /// Execute a ReAct prompt with automatic compaction, **without** mutating
    /// shared history.
    pub async fn prompt_compact(&self, msg: impl Into<String>) -> Result<ReActTrace, ReActError> {
        let msg = msg.into();
        let mut snapshot = lock_mutex(&self.history).clone();
        if let Some(cm) = self.context_manager() {
            cm.compact_history_if_needed(&mut snapshot, &msg)
                .await
                .map_err(|e| ReActError::Model(e.to_string()))?;
        }
        run_loop(
            &self.agent,
            &msg,
            &snapshot,
            self.max_cycles,
            &self.react_preamble,
            &self.span_emitter,
            &self.on_thought,
            &self.on_action,
            &self.on_observation,
            &self.on_final,
            &self.on_error,
            &self.history,
            false,
        )
        .await
    }

    /// Execute a ReAct chat with automatic compaction, **with** history
    /// mutation on success.
    pub async fn chat_compact(&self, msg: impl Into<String>) -> Result<String, ReActError> {
        let msg = msg.into();
        let mut snapshot = lock_mutex(&self.history).clone();
        if let Some(cm) = self.context_manager() {
            cm.compact_history_if_needed(&mut snapshot, &msg)
                .await
                .map_err(|e| ReActError::Model(e.to_string()))?;
        }
        let trace = run_loop(
            &self.agent,
            &msg,
            &snapshot,
            self.max_cycles,
            &self.react_preamble,
            &self.span_emitter,
            &self.on_thought,
            &self.on_action,
            &self.on_observation,
            &self.on_final,
            &self.on_error,
            &self.history,
            true,
        )
        .await?;
        Ok(trace.final_answer.map(|fa| fa.text).unwrap_or_default())
    }
}
