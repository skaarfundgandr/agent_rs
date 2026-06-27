use std::collections::HashMap;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use futures::Stream;
use rig_core::agent::{Agent, MultiTurnStreamItem, PromptHook};
use rig_core::completion::{CompletionError, CompletionModel, PromptError};
use rig_core::message::Message;
use rig_core::streaming::{StreamedAssistantContent, StreamedUserContent, StreamingChat};
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

use crate::agent::react::Compact;
use crate::agent::utils::{Mutex, lock_mutex};
use crate::domain::agent::ReActStreamItem;
use crate::domain::agent::{Action, FinalAnswer, Observation, ReActStep, ReActTrace, Thought};
use crate::domain::errors::ReActError;

use super::callbacks::{ActionCb, ErrorCb, FinalCb, ObservationCb, ThoughtCb};
use super::emitter::ReActSpanEmitter;
use super::helpers::recover_turn_limit_history;

pub(crate) struct StreamShared<M, P, C>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    pub(crate) agent: Agent<M, P>,
    pub(crate) history: Arc<Mutex<Vec<Message>>>,
    pub(crate) tool_timeout_secs: u64,
    pub(crate) on_thought: Option<ThoughtCb>,
    pub(crate) on_action: Option<ActionCb>,
    pub(crate) on_observation: Option<ObservationCb>,
    pub(crate) on_final: Option<FinalCb>,
    pub(crate) on_error: Option<ErrorCb>,
    pub(crate) context_manager: Option<Arc<dyn Compact + Send + Sync>>,
    pub(crate) _compaction: PhantomData<fn(M, P, C)>,
}

pub struct ReActStream<M, P, C = ()>
where
    M: CompletionModel
        + rig_core::streaming::StreamingChat<M, M::StreamingResponse>
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
    P: PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
    M::StreamingResponse: rig_core::completion::GetTokenUsage + Send,
{
    rx: tokio::sync::mpsc::Receiver<ReActStreamItem>,
    is_finished: bool,
    _phantom: PhantomData<fn(M, P, C)>,
}

#[allow(dead_code)]
fn is_retryable_streaming_error(e: &rig_core::agent::StreamingError) -> bool {
    match e {
        rig_core::agent::StreamingError::Completion(
            CompletionError::HttpError(_) | CompletionError::ProviderError(_),
        ) => true,
        rig_core::agent::StreamingError::Prompt(err) => matches!(
            err.as_ref(),
            PromptError::CompletionError(
                CompletionError::HttpError(_) | CompletionError::ProviderError(_)
            )
        ),
        _ => false,
    }
}

fn prompt_error_from_streaming(e: &rig_core::agent::StreamingError) -> Option<&PromptError> {
    match e {
        rig_core::agent::StreamingError::Prompt(err) => Some(err.as_ref()),
        _ => None,
    }
}

impl<M, P, C> ReActStream<M, P, C>
where
    M: CompletionModel
        + rig_core::streaming::StreamingChat<M, M::StreamingResponse>
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
    P: PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
    M::StreamingResponse: rig_core::completion::GetTokenUsage + Send,
    C: Send + Sync + 'static,
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        shared: Arc<StreamShared<M, P, C>>,
        history_snapshot: Vec<Message>,
        max_cycles: usize,
        max_retries: u32,
        react_preamble: Option<String>,
        span_emitter: Arc<dyn ReActSpanEmitter>,
        append_on_complete: bool,
        prompt_text: String,
    ) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let agent = shared.agent.clone();
        let history_clone = Arc::clone(&shared.history);

        let on_thought_cb = shared.on_thought.clone();
        let on_action_cb = shared.on_action.clone();
        let on_observation_cb = shared.on_observation.clone();
        let on_final_cb = shared.on_final.clone();
        let on_error_cb = shared.on_error.clone();

        tokio::spawn(async move {
            let effective_prompt = match &react_preamble {
                Some(preamble) => format!("{preamble}\n\n{prompt_text}"),
                None => prompt_text.clone(),
            };

            let mut trace = ReActTrace {
                prompt: prompt_text.clone(),
                steps: Vec::new(),
                final_answer: None,
            };

            let mut working_history = history_snapshot.clone();
            let mut current_prompt = Message::User {
                content: rig_core::OneOrMany::one(rig_core::message::UserContent::text(
                    effective_prompt.clone(),
                )),
            };

            let stream_timeout = Duration::from_secs(shared.tool_timeout_secs * 2);
            let mut current_cycle: usize = 0;
            let mut loop_continue = true;
            let mut error_emitted = false;
            let mut final_answer_buffer = String::new();
            let mut pending_tool_calls: HashMap<String, (String, Instant)> = HashMap::new();

            while loop_continue && current_cycle < max_cycles {
                span_emitter.emit_cycle_start(current_cycle);

                // Compact working history if a context manager is configured.
                if let Some(cm) = shared.context_manager.as_deref() {
                    let prompt_str = extract_prompt_text(&current_prompt, &effective_prompt);
                    if let Err(e) = cm.compact(&mut working_history, prompt_str).await {
                        let re_err = ReActError::Model(e.to_string());
                        if let Some(cb) = &on_error_cb {
                            cb(&re_err);
                        }
                        span_emitter.emit_error(&re_err);
                        let _ = tx
                            .send(ReActStreamItem::Error {
                                error: e.to_string(),
                            })
                            .await;
                        error_emitted = true;
                        break;
                    }
                }

                let stream_result = {
                    let mut stream_opt = None;
                    let mut init_attempt = 0u32;
                    loop {
                        init_attempt += 1;
                        let prompt_str = extract_prompt_text(&current_prompt, &effective_prompt);
                        let result = tokio::time::timeout(
                            stream_timeout,
                            agent
                                .stream_chat(prompt_str, working_history.clone())
                                .multi_turn(20),
                        )
                        .await;

                        match result {
                            Ok(s) => {
                                stream_opt = Some(s);
                                break;
                            }
                            Err(_elapsed) => {
                                if init_attempt < max_retries {
                                    let delay =
                                        Duration::from_millis(500 * 2u64.pow(init_attempt - 1));
                                    tokio::time::sleep(delay).await;
                                    continue;
                                }
                                let _ = tx
                                    .send(ReActStreamItem::Error {
                                        error: format!(
                                            "stream initialization timed out after {}s",
                                            stream_timeout.as_secs()
                                        ),
                                    })
                                    .await;
                                error_emitted = true;
                                loop_continue = false;
                                break;
                            }
                        }
                    }
                    stream_opt
                };

                let Some(mut stream) = stream_result else {
                    break;
                };

                let mut has_tool_calls = false;

                loop {
                    let item = match tokio::time::timeout(
                        stream_timeout,
                        futures::StreamExt::next(&mut stream),
                    )
                    .await
                    {
                        Ok(Some(Ok(item))) => item,
                        Ok(Some(Err(e))) => {
                            if let Some(prompt_err) = prompt_error_from_streaming(&e)
                                && matches!(prompt_err, PromptError::MaxTurnsError { .. })
                            {
                                let re_err = ReActError::Model(e.to_string());
                                if let Some(cb) = &on_error_cb {
                                    cb(&re_err);
                                }
                                span_emitter.emit_error(&re_err);
                                span_emitter.emit_cycle_end(current_cycle, &trace);

                                if let Some(mut recovered) = recover_turn_limit_history(prompt_err)
                                    && let Some(last) = recovered.pop()
                                {
                                    working_history = recovered;
                                    current_prompt = last;
                                    current_cycle += 1;
                                    break;
                                } else {
                                    let _ = tx
                                        .send(ReActStreamItem::Error {
                                            error: e.to_string(),
                                        })
                                        .await;
                                    error_emitted = true;
                                    loop_continue = false;
                                    break;
                                }
                            }

                            let re_err = ReActError::Model(e.to_string());
                            if let Some(cb) = &on_error_cb {
                                cb(&re_err);
                            }
                            span_emitter.emit_error(&re_err);
                            let _ = tx
                                .send(ReActStreamItem::Error {
                                    error: e.to_string(),
                                })
                                .await;
                            error_emitted = true;
                            loop_continue = false;
                            break;
                        }
                        Ok(None) => {
                            let _ = tx
                                .send(ReActStreamItem::Error {
                                    error: "stream ended unexpectedly".to_string(),
                                })
                                .await;
                            error_emitted = true;
                            loop_continue = false;
                            break;
                        }
                        Err(_elapsed) => {
                            let _ = tx
                                .send(ReActStreamItem::Error {
                                    error: format!(
                                        "stream item timed out after {}s",
                                        shared.tool_timeout_secs * 2
                                    ),
                                })
                                .await;
                            error_emitted = true;
                            loop_continue = false;
                            break;
                        }
                    };

                    match item {
                        MultiTurnStreamItem::StreamAssistantItem(assistant_item) => {
                            match assistant_item {
                                StreamedAssistantContent::Text(text) => {
                                    if has_tool_calls {
                                        final_answer_buffer.push_str(&text.text);
                                        if send_or_break(
                                            &tx,
                                            ReActStreamItem::FinalAnswerDelta {
                                                delta: text.text,
                                                cycle: current_cycle,
                                            },
                                        )
                                        .await
                                        {
                                            loop_continue = false;
                                            break;
                                        }
                                    } else {
                                        if send_or_break(
                                            &tx,
                                            ReActStreamItem::ThoughtDelta {
                                                delta: text.text,
                                                cycle: current_cycle,
                                            },
                                        )
                                        .await
                                        {
                                            loop_continue = false;
                                            break;
                                        }
                                    }
                                }
                                StreamedAssistantContent::ToolCall {
                                    tool_call,
                                    internal_call_id: _,
                                } => {
                                    let action = Action {
                                        tool_name: tool_call.function.name.clone(),
                                        args: tool_call.function.arguments.to_string(),
                                        tool_call_id: Some(tool_call.id.clone()),
                                        cycle: current_cycle,
                                    };
                                    pending_tool_calls.insert(
                                        tool_call.id.clone(),
                                        (tool_call.function.name.clone(), Instant::now()),
                                    );
                                    if let Some(cb) = &on_action_cb {
                                        cb(&action);
                                    }
                                    span_emitter.emit_action(&action);
                                    trace.steps.push(ReActStep::Action(action.clone()));
                                    has_tool_calls = true;
                                    if send_or_break(
                                        &tx,
                                        ReActStreamItem::Action {
                                            tool_name: action.tool_name,
                                            args: action.args,
                                            tool_call_id: action.tool_call_id,
                                            cycle: action.cycle,
                                        },
                                    )
                                    .await
                                    {
                                        loop_continue = false;
                                        break;
                                    }
                                }
                                StreamedAssistantContent::Reasoning(reasoning) => {
                                    let text = reasoning.display_text();
                                    if !text.is_empty() {
                                        let thought = Thought {
                                            reasoning: text.clone(),
                                            cycle: current_cycle,
                                        };
                                        if let Some(cb) = &on_thought_cb {
                                            cb(&thought);
                                        }
                                        span_emitter.emit_thought(&thought);
                                        trace.steps.push(ReActStep::Thought(thought));
                                        if send_or_break(
                                            &tx,
                                            ReActStreamItem::ThoughtDelta {
                                                delta: text,
                                                cycle: current_cycle,
                                            },
                                        )
                                        .await
                                        {
                                            loop_continue = false;
                                            break;
                                        }
                                    }
                                }
                                StreamedAssistantContent::ReasoningDelta { reasoning, .. }
                                    if !reasoning.is_empty() =>
                                {
                                    if send_or_break(
                                        &tx,
                                        ReActStreamItem::ThoughtDelta {
                                            delta: reasoning,
                                            cycle: current_cycle,
                                        },
                                    )
                                    .await
                                    {
                                        loop_continue = false;
                                        break;
                                    }
                                }
                                StreamedAssistantContent::ReasoningDelta { .. } => {}
                                _ => {}
                            }
                        }
                        MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                            tool_result,
                            internal_call_id: _,
                        }) => {
                            let result_text = tool_result
                                .content
                                .iter()
                                .filter_map(|c| match c {
                                    rig_core::message::ToolResultContent::Text(t) => {
                                        Some(t.text.as_str())
                                    }
                                    _ => None,
                                })
                                .collect::<String>();
                            let (tool_name, start) = pending_tool_calls
                                .remove(&tool_result.id)
                                .unwrap_or_else(|| ("unknown".to_string(), Instant::now()));
                            let observation = Observation {
                                tool_name,
                                result: result_text,
                                is_error: false,
                                cycle: current_cycle,
                                duration: start.elapsed(),
                            };
                            if let Some(cb) = &on_observation_cb {
                                cb(&observation);
                            }
                            span_emitter.emit_observation(&observation);
                            trace
                                .steps
                                .push(ReActStep::Observation(observation.clone()));
                            if send_or_break(
                                &tx,
                                ReActStreamItem::Observation {
                                    tool_name: observation.tool_name,
                                    result: observation.result,
                                    is_error: observation.is_error,
                                    cycle: observation.cycle,
                                    duration: observation.duration,
                                },
                            )
                            .await
                            {
                                loop_continue = false;
                                break;
                            }
                        }
                        MultiTurnStreamItem::FinalResponse(final_resp) => {
                            let last_text = final_resp.response().to_string();
                            let final_text = if last_text.is_empty() {
                                std::mem::take(&mut final_answer_buffer)
                            } else {
                                final_answer_buffer.clear();
                                last_text
                            };

                            if !final_text.is_empty() {
                                let fa = FinalAnswer {
                                    text: final_text,
                                    cycles: current_cycle + 1,
                                };
                                trace.steps.push(ReActStep::FinalAnswer(fa.clone()));
                                trace.final_answer = Some(fa.clone());
                                if let Some(cb) = &on_final_cb {
                                    cb(&fa);
                                }
                            }

                            if append_on_complete {
                                let mut h = lock_mutex(&history_clone);
                                h.push(Message::User {
                                    content: rig_core::OneOrMany::one(
                                        rig_core::message::UserContent::text(&prompt_text),
                                    ),
                                });
                                if let Some(fa) = &trace.final_answer {
                                    h.push(Message::assistant(&fa.text));
                                }
                            }

                            span_emitter.emit_cycle_end(current_cycle, &trace);
                            loop_continue = false;
                            break;
                        }
                        _ => {}
                    }
                }
            }

            if !error_emitted {
                if trace.final_answer.is_some() {
                    let _ = tx.send(ReActStreamItem::Completed { trace }).await;
                } else {
                    let err = ReActError::MaxCyclesExceeded { cycles: max_cycles };
                    if let Some(cb) = &on_error_cb {
                        cb(&err);
                    }
                    span_emitter.emit_error(&err);
                    let _ = tx
                        .send(ReActStreamItem::Error {
                            error: err.to_string(),
                        })
                        .await;
                }
            }
        });

        Self {
            rx,
            is_finished: false,
            _phantom: PhantomData,
        }
    }
}

fn extract_prompt_text<'a>(prompt: &'a Message, fallback: &'a str) -> &'a str {
    match prompt {
        Message::User { content } => content.iter().find_map(|c| match c {
            rig_core::message::UserContent::Text(t) => Some(t.text.as_str()),
            _ => None,
        }),
        _ => None,
    }
    .unwrap_or(fallback)
}

async fn send_or_break(
    tx: &tokio::sync::mpsc::Sender<ReActStreamItem>,
    item: ReActStreamItem,
) -> bool {
    tx.send(item).await.is_err()
}

impl<M, P, C> Stream for ReActStream<M, P, C>
where
    M: CompletionModel
        + rig_core::streaming::StreamingChat<M, M::StreamingResponse>
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
    P: PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
    M::StreamingResponse: rig_core::completion::GetTokenUsage + Send,
    C: Send + Sync + 'static,
{
    type Item = ReActStreamItem;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.is_finished {
            return Poll::Ready(None);
        }

        match Pin::new(&mut self.rx).poll_recv(cx) {
            Poll::Ready(Some(item)) => {
                if matches!(
                    &item,
                    ReActStreamItem::Completed { .. } | ReActStreamItem::Error { .. }
                ) {
                    self.is_finished = true;
                }
                Poll::Ready(Some(item))
            }
            Poll::Ready(None) => {
                self.is_finished = true;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}
