use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rig_core::agent::Agent;
use rig_core::completion::CompletionModel;
use rig_core::message::Message;
use rig_core::streaming::StreamingChat;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

use crate::agent::react::Compact;
use crate::agent::retry::retry_with_backoff;
use crate::domain::agent::{FinalAnswer, ReActStep, ReActStreamItem, ReActTrace};
use crate::domain::errors::ReActError;

use super::built_methods::effective_prompt;
use super::callbacks::{ActionCb, ErrorCb, FinalCb, ObservationCb, ThoughtCb};
use super::emitter::ReActSpanEmitter;
use super::streaming::extract_prompt_text;

pub(crate) struct StreamingLoopContext<M, C>
where
    M: CompletionModel
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
    M::StreamingResponse: rig_core::completion::GetTokenUsage + Send,
    C: Send + Sync + 'static,
{
    pub agent: Agent<M>,
    pub on_thought_cb: Option<ThoughtCb>,
    pub on_action_cb: Option<ActionCb>,
    pub on_observation_cb: Option<ObservationCb>,
    pub on_final_cb: Option<FinalCb>,
    pub on_error_cb: Option<ErrorCb>,
    pub react_preamble: Option<String>,
    pub max_cycles: usize,
    pub max_retries: u32,
    pub span_emitter: Arc<dyn ReActSpanEmitter>,
    pub prompt_text: String,
    pub history_snapshot: Vec<Message>,
    pub context_manager: Option<Arc<dyn Compact + Send + Sync>>,
    pub tool_timeout_secs: u64,
    pub tx: tokio::sync::mpsc::Sender<ReActStreamItem>,
    pub _compaction: PhantomData<C>,
}

pub(crate) async fn run_streaming_loop<M, C>(ctx: StreamingLoopContext<M, C>)
where
    M: CompletionModel
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
    M::StreamingResponse: rig_core::completion::GetTokenUsage + Send,
    C: Send + Sync + 'static,
{
    let StreamingLoopContext {
        agent,
        on_thought_cb,
        on_action_cb,
        on_observation_cb,
        on_final_cb,
        on_error_cb,
        react_preamble,
        max_cycles,
        max_retries,
        span_emitter,
        prompt_text,
        history_snapshot,
        context_manager,
        tool_timeout_secs,
        tx,
        _compaction: _,
    } = ctx;

    let effective_prompt = effective_prompt(&react_preamble, &prompt_text);

    let mut trace = ReActTrace {
        prompt: prompt_text.clone(),
        steps: Vec::new(),
        final_answer: None,
    };

    let mut working_history: Vec<Message> = history_snapshot.clone();
    let mut current_prompt = Message::User {
        content: rig_core::OneOrMany::one(rig_core::message::UserContent::text(
            effective_prompt.clone(),
        )),
    };

    let stream_timeout = Duration::from_secs(tool_timeout_secs * 2);
    let mut current_cycle: usize = 0;
    let mut loop_continue = true;
    let mut error_emitted = false;
    let mut completed_sent = false;
    let mut final_answer_buffer = String::new();
    let mut final_history: Option<Vec<Message>> = None;
    let mut pending_tool_calls: HashMap<String, (String, Instant)> = HashMap::new();

    while loop_continue && current_cycle < max_cycles {
        span_emitter.emit_cycle_start(current_cycle);

        if let Some(cm) = context_manager.as_deref() {
            let prompt_str = extract_prompt_text(&current_prompt, &effective_prompt);
            if let Err(e) = cm.compact(&mut working_history, prompt_str).await {
                emit_stream_error(&e, &on_error_cb, &span_emitter);
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
            let prompt_str = extract_prompt_text(&current_prompt, &effective_prompt);
            retry_with_backoff(max_retries, || {
                let prompt_str = prompt_str.to_owned();
                let working = working_history.clone();
                let agent = agent.clone();
                async move {
                    tokio::time::timeout(
                        stream_timeout,
                        async {
                            agent.stream_chat(prompt_str, working)
                                .max_turns(20)
                                .await
                        },
                    )
                    .await
                    .map_err(|e| {
                        rig_core::completion::PromptError::CompletionError(
                            rig_core::completion::request::CompletionError::RequestError(Box::new(
                                e,
                            )),
                        )
                    })
                }
            })
            .await
            .ok()
        };

        let Some(mut stream) = stream_result else {
            let _ = tx
                .send(ReActStreamItem::Error {
                    error: format!(
                        "stream initialization timed out after {}s",
                        stream_timeout.as_secs()
                    ),
                })
                .await;
            error_emitted = true;
            break;
        };

        let mut has_tool_calls = false;

        loop {
            let item =
                match tokio::time::timeout(stream_timeout, futures::StreamExt::next(&mut stream))
                    .await
                {
                    Ok(Some(Ok(item))) => item,
                    Ok(Some(Err(e))) => {
                        if let Some(prompt_err) = super::streaming::prompt_error_from_streaming(&e)
                            && matches!(
                                prompt_err,
                                rig_core::completion::PromptError::MaxTurnsError { .. }
                            )
                        {
                            emit_stream_error(&e, &on_error_cb, &span_emitter);
                            span_emitter.emit_cycle_end(current_cycle, &trace);

                            if let Some(mut recovered) =
                                super::helpers::recover_turn_limit_history(prompt_err)
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

                        emit_stream_error(&e, &on_error_cb, &span_emitter);
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
                                    tool_timeout_secs * 2
                                ),
                            })
                            .await;
                        error_emitted = true;
                        loop_continue = false;
                        break;
                    }
                };

            use rig_core::agent::MultiTurnStreamItem;
            match item {
                MultiTurnStreamItem::StreamAssistantItem(assistant_item) => {
                    if super::stream_process::process_assistant_item(
                        assistant_item,
                        &tx,
                        &mut trace,
                        &mut has_tool_calls,
                        &mut final_answer_buffer,
                        &mut pending_tool_calls,
                        current_cycle,
                        &on_thought_cb,
                        &on_action_cb,
                        &span_emitter,
                    )
                    .await
                    {
                        loop_continue = false;
                        break;
                    }
                }
                MultiTurnStreamItem::StreamUserItem(
                    rig_core::streaming::StreamedUserContent::ToolResult { tool_result, .. },
                ) => {
                    if super::stream_process::process_tool_result(
                        tool_result,
                        &tx,
                        &mut trace,
                        &mut pending_tool_calls,
                        current_cycle,
                        &on_observation_cb,
                        &span_emitter,
                    )
                    .await
                    {
                        loop_continue = false;
                        break;
                    }
                }
                MultiTurnStreamItem::FinalResponse(final_resp) => {
                    let last_text = final_resp.output().to_string();
                    let final_text = if last_text.is_empty() {
                        std::mem::take(&mut final_answer_buffer)
                    } else {
                        final_answer_buffer.clear();
                        last_text
                    };

                    if !final_text.is_empty() {
                        let fa = FinalAnswer {
                            text: final_text.clone(),
                            cycles: current_cycle + 1,
                        };
                        trace.steps.push(ReActStep::FinalAnswer(fa.clone()));
                        trace.final_answer = Some(fa.clone());
                        if let Some(cb) = &on_final_cb {
                            cb(&fa);
                        }
                    }

                    let mut fh = history_snapshot.clone();
                    fh.push(Message::User {
                        content: rig_core::OneOrMany::one(rig_core::message::UserContent::text(
                            &prompt_text,
                        )),
                    });
                    if !final_text.is_empty() {
                        fh.push(Message::assistant(&final_text));
                    }
                    final_history = Some(fh);

                    span_emitter.emit_cycle_end(current_cycle, &trace);
                    completed_sent = true;
                    loop_continue = false;
                    break;
                }
                _ => {}
            }
        }
    }

    if !error_emitted && !completed_sent {
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
    } else if completed_sent {
        let fh = final_history.take().unwrap_or_default();
        let _ = tx
            .send(ReActStreamItem::Completed {
                trace,
                final_history: fh,
            })
            .await;
    }
}

fn emit_stream_error(
    e: &impl std::fmt::Display,
    on_error_cb: &Option<ErrorCb>,
    span_emitter: &Arc<dyn ReActSpanEmitter>,
) -> ReActError {
    let re_err = ReActError::Model(e.to_string());
    if let Some(cb) = on_error_cb {
        cb(&re_err);
    }
    span_emitter.emit_error(&re_err);
    re_err
}
