use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::Stream;
use rig_core::agent::{Agent, MultiTurnStreamItem, PromptHook};
use rig_core::completion::CompletionModel;
use rig_core::message::Message;
use rig_core::streaming::{StreamedAssistantContent, StreamingChat};
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

use crate::agent::utils::{Mutex, lock_mutex};
use crate::domain::agent::ReActStreamItem;
use crate::domain::agent::{Action, FinalAnswer, ReActStep, ReActTrace, Thought};

use super::emitter::ReActSpanEmitter;

/// Shared state between `BuiltReAct` and `ReActStream`.
pub(crate) struct StreamShared<M, P, C>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    pub(crate) agent: Agent<M, P>,
    pub(crate) history: Arc<Mutex<Vec<Message>>>,
    pub(crate) _compaction: PhantomData<fn(M, P, C)>,
}

/// A streaming ReAct loop. Implements [`Stream`] yielding [`ReActStreamItem`].
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
    pub(crate) fn new(
        shared: Arc<StreamShared<M, P, C>>,
        history_snapshot: Vec<Message>,
        max_cycles: usize,
        react_preamble: Option<String>,
        span_emitter: Arc<dyn ReActSpanEmitter>,
        append_on_complete: bool,
        prompt_text: String,
    ) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let agent = shared.agent.clone();
        let shared_clone = Arc::clone(&shared);

        tokio::spawn(async move {
            let effective_prompt = match &react_preamble {
                Some(preamble) => format!("{preamble}\n\n{prompt_text}"),
                None => prompt_text.clone(),
            };

            let mut trace = ReActTrace {
                prompt: prompt_text,
                steps: Vec::new(),
                final_answer: None,
            };

            let mut current_cycle: usize = 0;
            let mut has_tool_calls = false;

            let stream_result = agent
                .stream_chat(&effective_prompt, history_snapshot)
                .multi_turn(max_cycles)
                .await;

            let mut stream = stream_result;

            loop {
                let item = match futures::StreamExt::next(&mut stream).await {
                    Some(Ok(item)) => item,
                    Some(Err(e)) => {
                        let _ = tx
                            .send(ReActStreamItem::Error {
                                error: e.to_string(),
                            })
                            .await;
                        break;
                    }
                    None => {
                        let _ = tx
                            .send(ReActStreamItem::Error {
                                error: "stream ended unexpectedly".to_string(),
                            })
                            .await;
                        break;
                    }
                };

                match item {
                    MultiTurnStreamItem::StreamAssistantItem(assistant_item) => {
                        match assistant_item {
                            StreamedAssistantContent::Text(text) => {
                                if has_tool_calls {
                                    let fa = FinalAnswer {
                                        text: text.text.clone(),
                                        cycles: current_cycle + 1,
                                    };
                                    trace.steps.push(ReActStep::FinalAnswer(fa));
                                    let _ = tx
                                        .send(ReActStreamItem::FinalAnswerDelta {
                                            delta: text.text,
                                        })
                                        .await;
                                } else {
                                    let _ = tx
                                        .send(ReActStreamItem::ThoughtDelta {
                                            delta: text.text,
                                            cycle: current_cycle,
                                        })
                                        .await;
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
                                let item = ReActStreamItem::Action {
                                    tool_name: action.tool_name.clone(),
                                    args: action.args.clone(),
                                    tool_call_id: action.tool_call_id.clone(),
                                    cycle: action.cycle,
                                };
                                trace.steps.push(ReActStep::Action(action));
                                has_tool_calls = true;
                                let _ = tx.send(item).await;
                            }
                            StreamedAssistantContent::Reasoning(reasoning) => {
                                let text = reasoning.display_text();
                                if !text.is_empty() {
                                    let thought = Thought {
                                        reasoning: text.clone(),
                                        cycle: current_cycle,
                                    };
                                    span_emitter.emit_thought(&thought);
                                    trace.steps.push(ReActStep::Thought(thought));
                                    let _ = tx
                                        .send(ReActStreamItem::ThoughtDelta {
                                            delta: text,
                                            cycle: current_cycle,
                                        })
                                        .await;
                                }
                            }
                            StreamedAssistantContent::ReasoningDelta { reasoning, .. }
                                if !reasoning.is_empty() =>
                            {
                                let _ = tx
                                    .send(ReActStreamItem::ThoughtDelta {
                                        delta: reasoning,
                                        cycle: current_cycle,
                                    })
                                    .await;
                            }
                            StreamedAssistantContent::ReasoningDelta { .. } => {}
                            _ => {}
                        }
                    }
                    MultiTurnStreamItem::FinalResponse(final_resp) => {
                        if let Some(history) = final_resp.history() {
                            let last_text = final_resp.response().to_string();
                            if !last_text.is_empty() && trace.final_answer.is_none() {
                                let fa = FinalAnswer {
                                    text: last_text,
                                    cycles: current_cycle + 1,
                                };
                                trace.steps.push(ReActStep::FinalAnswer(fa.clone()));
                                trace.final_answer = Some(fa);
                            }

                            if append_on_complete {
                                let new_messages = history.to_vec();
                                *lock_mutex(&shared_clone.history) = new_messages;
                            }
                        }

                        span_emitter.emit_cycle_end(current_cycle, &trace);

                        let _ = tx.send(ReActStreamItem::Completed { trace }).await;
                        break;
                    }
                    _ => {}
                }

                current_cycle += 1;
            }
        });

        Self {
            rx,
            is_finished: false,
            _phantom: PhantomData,
        }
    }
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
