use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rig_core::agent::{Agent, PromptHook};
use rig_core::completion::{CompletionError, CompletionModel, Prompt, PromptError};
use rig_core::message::{AssistantContent, Message, ToolCall, ToolResultContent, UserContent};
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

use crate::agent::memory::ContextManager;
use crate::agent::react::Compact;
use crate::agent::utils::{Mutex, lock_mutex};

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
use crate::domain::agent::{Action, FinalAnswer, Observation, ReActStep, ReActTrace, Thought};
use crate::domain::errors::ReActError;

use super::callbacks::{ActionCb, ErrorCb, FinalCb, ObservationCb, ThoughtCb};
use super::emitter::ReActSpanEmitter;
use super::helpers::{detect_final_answer, recover_turn_limit_history, tool_error_to_string};

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
    pub(crate) max_retries: u32,
    pub(crate) react_preamble: Option<String>,
    pub(crate) span_emitter: Arc<dyn ReActSpanEmitter>,
    pub(crate) on_thought: Option<ThoughtCb>,
    pub(crate) on_action: Option<ActionCb>,
    pub(crate) on_observation: Option<ObservationCb>,
    pub(crate) on_final: Option<FinalCb>,
    pub(crate) on_error: Option<ErrorCb>,
    pub(crate) context_manager: Option<Arc<dyn Compact + Send + Sync>>,
    pub(crate) tool_timeout_secs: u64,
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

    /// Return the configured `max_retries` limit.
    pub fn max_retries(&self) -> u32 {
        self.max_retries
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

    // Pending tool names, in emission order. Tool calls and their results are
    // interleaved by rig-core, so FIFO pairing is robust when provider IDs
    // between the call and the result do not match exactly.
    let mut pending_tool_names: std::collections::VecDeque<String> =
        std::collections::VecDeque::new();

    // Skip the first message: it is the prompt passed to `agent.prompt()`.
    // In later ReAct cycles that prompt is the previous cycle's last tool
    // result, whose action/observation were already emitted.
    for msg in &messages[1..last_assistant_idx] {
        match msg {
            Message::Assistant { content, .. } => {
                for item in content.iter() {
                    if let AssistantContent::ToolCall(tc) = item {
                        pending_tool_names.push_back(tc.function.name.clone());
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
                            .pop_front()
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

/// Standalone ReAct loop that works on a local `Vec<Message>` clone.
/// On success when `append_to_shared_history` is true, appends to the
/// shared history via the Mutex.
#[allow(clippy::too_many_arguments)]
async fn run_loop<M, P>(
    agent: &Agent<M, P>,
    prompt: &str,
    history_snapshot: &[Message],
    max_cycles: usize,
    max_retries: u32,
    tool_timeout_secs: u64,
    react_preamble: &Option<String>,
    span_emitter: &Arc<dyn ReActSpanEmitter>,
    on_thought: &Option<ThoughtCb>,
    on_action: &Option<ActionCb>,
    on_observation: &Option<ObservationCb>,
    on_final: &Option<FinalCb>,
    on_error: &Option<ErrorCb>,
    shared_history: &Arc<Mutex<Vec<Message>>>,
    append_to_shared_history: bool,
    context_manager: Option<&(dyn Compact + Send + Sync)>,
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
    let mut no_assistant_retried = false;
    let mut empty_output_retried = false;

    for cycle in 0..max_cycles {
        span_emitter.emit_cycle_start(cycle);

        if let Some(cm) = context_manager {
            let prompt_text = match &current_prompt {
                Message::User { content } => content
                    .iter()
                    .find_map(|c| match c {
                        UserContent::Text(t) => Some(t.text.as_str()),
                        _ => None,
                    })
                    .unwrap_or(prompt),
                _ => prompt,
            };
            cm.compact(&mut working_history, prompt_text)
                .await
                .map_err(|e| ReActError::Model(e.to_string()))?;
        }

        let mut attempt = 0;
        let response = loop {
            attempt += 1;
            match agent
                .prompt(current_prompt.clone())
                .with_history(working_history.iter().cloned())
                .extended_details()
                .await
            {
                Ok(resp) => break resp,
                Err(e) => {
                    let is_transient = matches!(
                        &e,
                        PromptError::CompletionError(
                            CompletionError::HttpError(_) | CompletionError::ProviderError(_)
                        )
                    );
                    let is_turn_limit = matches!(&e, PromptError::MaxTurnsError { .. });
                    if is_transient && attempt < max_retries {
                        let delay = std::time::Duration::from_millis(500 * 2u64.pow(attempt - 1));
                        tokio::time::sleep(delay).await;
                        continue;
                    }
                    // Turn limit (rig-core's `default_max_turns`) is a
                    // recoverable condition at the ReAct cycle level: the
                    // model hit its per-`agent.prompt()` tool-call budget
                    // mid-cycle. Surface the error to the `on_error`
                    // callback, but let the outer `for cycle` loop move on
                    // and try a fresh ReAct cycle, as long as the cycle
                    // budget hasn't been exhausted.
                    //
                    // Critical: rig-core carries the *full* accumulated
                    // history (snapshot + this cycle's prompt + every
                    // assistant turn and tool result gathered before the
                    // limit) inside `MaxTurnsError`. Recover it into
                    // `working_history`/`current_prompt` so the next cycle
                    // continues from where the inner loop left off. Without
                    // this recovery, the next cycle re-sends an identical
                    // request and reproduces the same turn-limit error until
                    // `max_cycles` is exhausted — the "hard stuck" failure
                    // mode. With recovery, the next cycle starts with a fresh
                    // turn budget and the accumulated context, so the model
                    // can reason (Chain-of-Thought, via the existing ReAct
                    // preamble) over what it has gathered instead of redoing
                    // the same tool calls.
                    if is_turn_limit {
                        let err = ReActError::Model(e.to_string());
                        if let Some(cb) = on_error {
                            cb(&err);
                        }
                        span_emitter.emit_error(&err);
                        span_emitter.emit_cycle_end(cycle, &trace);
                        if let Some(mut recovered) = recover_turn_limit_history(&e)
                            && let Some(last) = recovered.pop()
                        {
                            // `recovered` is snapshot + cycle prompt +
                            // progress (all but the final pending message);
                            // `last` is that pending message (the tool
                            // result rig-core was about to act on).
                            working_history = recovered;
                            current_prompt = last;
                        }
                        continue;
                    }
                    let err = ReActError::Model(e.to_string());
                    if let Some(cb) = on_error {
                        cb(&err);
                    }
                    span_emitter.emit_error(&err);
                    span_emitter.emit_cycle_end(cycle, &trace);
                    return Err(err);
                }
            }
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

        let assistant_content = match working_history.iter().rev().find_map(|msg| match msg {
            Message::Assistant { content, .. } => Some(content),
            _ => None,
        }) {
            Some(content) => content.clone(),
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
                let mut h = lock_mutex(shared_history);
                h.push(Message::User {
                    content: rig_core::OneOrMany::one(UserContent::text(prompt)),
                });
                h.push(Message::assistant(&fa.text));
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
                let mut h = lock_mutex(shared_history);
                h.push(Message::User {
                    content: rig_core::OneOrMany::one(UserContent::text(prompt)),
                });
                h.push(Message::assistant(&fa.text));
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
            let result = tokio::time::timeout(
                Duration::from_secs(tool_timeout_secs),
                agent
                    .tool_server_handle
                    .call_tool(&tc.function.name, &args_str),
            )
            .await;
            let duration = start.elapsed();

            let observation = match result {
                Ok(Ok(s)) => Observation {
                    tool_name: tc.function.name.clone(),
                    result: s,
                    is_error: false,
                    cycle,
                    duration,
                },
                Ok(Err(e)) => Observation {
                    tool_name: tc.function.name.clone(),
                    result: tool_error_to_string(&e),
                    is_error: true,
                    cycle,
                    duration,
                },
                Err(_elapsed) => Observation {
                    tool_name: tc.function.name.clone(),
                    result: format!(
                        "Tool '{}' timed out after {}s",
                        tc.function.name, tool_timeout_secs
                    ),
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
    span_emitter.emit_error(&err);
    span_emitter.emit_cycle_end(max_cycles.saturating_sub(1), &trace);
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
            self.max_retries,
            self.tool_timeout_secs,
            &self.react_preamble,
            &self.span_emitter,
            &self.on_thought,
            &self.on_action,
            &self.on_observation,
            &self.on_final,
            &self.on_error,
            &self.history,
            false,
            None,
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
            self.max_retries,
            self.tool_timeout_secs,
            &self.react_preamble,
            &self.span_emitter,
            &self.on_thought,
            &self.on_action,
            &self.on_observation,
            &self.on_final,
            &self.on_error,
            &self.history,
            true,
            None,
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
                tool_timeout_secs: self.tool_timeout_secs,
                on_thought: self.on_thought.as_ref().map(Arc::clone),
                on_action: self.on_action.as_ref().map(Arc::clone),
                on_observation: self.on_observation.as_ref().map(Arc::clone),
                on_final: self.on_final.as_ref().map(Arc::clone),
                on_error: self.on_error.as_ref().map(Arc::clone),
                context_manager: None,
                _compaction: PhantomData,
            }),
            snapshot,
            self.max_cycles,
            self.max_retries,
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
                tool_timeout_secs: self.tool_timeout_secs,
                on_thought: self.on_thought.as_ref().map(Arc::clone),
                on_action: self.on_action.as_ref().map(Arc::clone),
                on_observation: self.on_observation.as_ref().map(Arc::clone),
                on_final: self.on_final.as_ref().map(Arc::clone),
                on_error: self.on_error.as_ref().map(Arc::clone),
                context_manager: None,
                _compaction: PhantomData,
            }),
            snapshot,
            self.max_cycles,
            self.max_retries,
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
    /// Execute a ReAct prompt with automatic compaction, **without** mutating
    /// shared history.
    pub async fn prompt_compact(&self, msg: impl Into<String>) -> Result<ReActTrace, ReActError> {
        let msg = msg.into();
        let mut snapshot = lock_mutex(&self.history).clone();
        if let Some(cm) = self.context_manager.as_deref() {
            cm.compact(&mut snapshot, &msg)
                .await
                .map_err(|e| ReActError::Model(e.to_string()))?;
        }
        run_loop(
            &self.agent,
            &msg,
            &snapshot,
            self.max_cycles,
            self.max_retries,
            self.tool_timeout_secs,
            &self.react_preamble,
            &self.span_emitter,
            &self.on_thought,
            &self.on_action,
            &self.on_observation,
            &self.on_final,
            &self.on_error,
            &self.history,
            false,
            self.context_manager.as_deref(),
        )
        .await
    }

    /// Execute a ReAct chat with automatic compaction, **with** history
    /// mutation on success.
    pub async fn chat_compact(&self, msg: impl Into<String>) -> Result<String, ReActError> {
        let msg = msg.into();
        let mut snapshot = lock_mutex(&self.history).clone();
        if let Some(cm) = self.context_manager.as_deref() {
            cm.compact(&mut snapshot, &msg)
                .await
                .map_err(|e| ReActError::Model(e.to_string()))?;
        }
        let trace = run_loop(
            &self.agent,
            &msg,
            &snapshot,
            self.max_cycles,
            self.max_retries,
            self.tool_timeout_secs,
            &self.react_preamble,
            &self.span_emitter,
            &self.on_thought,
            &self.on_action,
            &self.on_observation,
            &self.on_final,
            &self.on_error,
            &self.history,
            true,
            self.context_manager.as_deref(),
        )
        .await?;
        Ok(trace.final_answer.map(|fa| fa.text).unwrap_or_default())
    }
}

// ── Streaming-compaction methods ──────────────────────────────────────────

impl<M, P, C> BuiltReAct<M, P, C>
where
    M: CompletionModel
        + rig_core::streaming::StreamingChat<M, M::StreamingResponse>
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
    P: PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
    M::StreamingResponse: rig_core::completion::GetTokenUsage + Send,
    C: Prompt + WasmCompatSend + WasmCompatSync + 'static,
{
    fn make_stream_shared(&self) -> Arc<super::streaming::StreamShared<M, P, C>> {
        Arc::new(super::streaming::StreamShared {
            agent: self.agent.clone(),
            history: Arc::clone(&self.history),
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

    /// Stream a ReAct prompt with automatic compaction. Does **not** mutate shared history.
    pub async fn stream_prompt_compact(
        &self,
        msg: impl Into<String>,
    ) -> Result<super::streaming::ReActStream<M, P, C>, ReActError> {
        let msg = msg.into();
        let mut snapshot = lock_mutex(&self.history).clone();
        if let Some(cm) = self.context_manager.as_deref() {
            cm.compact(&mut snapshot, &msg)
                .await
                .map_err(|e| ReActError::Model(e.to_string()))?;
        }
        Ok(super::streaming::ReActStream::new(
            self.make_stream_shared(),
            snapshot,
            self.max_cycles,
            self.max_retries,
            self.react_preamble.clone(),
            Arc::clone(&self.span_emitter),
            false,
            msg,
        ))
    }

    /// Stream a ReAct chat with automatic compaction. Mutates shared history on completion.
    pub async fn stream_chat_compact(
        &self,
        msg: impl Into<String>,
    ) -> Result<super::streaming::ReActStream<M, P, C>, ReActError> {
        let msg = msg.into();
        let mut snapshot = lock_mutex(&self.history).clone();
        if let Some(cm) = self.context_manager.as_deref() {
            cm.compact(&mut snapshot, &msg)
                .await
                .map_err(|e| ReActError::Model(e.to_string()))?;
        }
        Ok(super::streaming::ReActStream::new(
            self.make_stream_shared(),
            snapshot,
            self.max_cycles,
            self.max_retries,
            self.react_preamble.clone(),
            Arc::clone(&self.span_emitter),
            true,
            msg,
        ))
    }
}
