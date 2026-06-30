use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::Stream;
use rig_core::agent::{Agent, PromptHook};
use rig_core::completion::CompletionModel;
use rig_core::message::Message;
use rig_core::streaming::StreamingChat;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

use crate::agent::react::Compact;
use crate::domain::agent::ReActStreamItem;

pub(crate) struct StreamShared<M, P, C>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    pub(crate) agent: Agent<M, P>,
    pub(crate) tool_timeout_secs: u64,
    pub(crate) on_thought: Option<super::callbacks::ThoughtCb>,
    pub(crate) on_action: Option<super::callbacks::ActionCb>,
    pub(crate) on_observation: Option<super::callbacks::ObservationCb>,
    pub(crate) on_final: Option<super::callbacks::FinalCb>,
    pub(crate) on_error: Option<super::callbacks::ErrorCb>,
    pub(crate) context_manager: Option<Arc<dyn Compact + Send + Sync>>,
    pub(crate) _compaction: PhantomData<fn(M, P, C)>,
}

pub struct ReActStream<'h, M, P, C = ()>
where
    M: CompletionModel
        + StreamingChat<M, M::StreamingResponse>
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
    P: PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
    M::StreamingResponse: rig_core::completion::GetTokenUsage + Send,
{
    rx: tokio::sync::mpsc::Receiver<ReActStreamItem>,
    is_finished: bool,
    history_out: Option<&'h mut Vec<Message>>,
    _phantom: PhantomData<fn(M, P, C)>,
}

pub(crate) fn prompt_error_from_streaming(
    e: &rig_core::agent::StreamingError,
) -> Option<&rig_core::completion::PromptError> {
    match e {
        rig_core::agent::StreamingError::Prompt(err) => Some(err.as_ref()),
        _ => None,
    }
}

impl<'h, M, P, C> ReActStream<'h, M, P, C>
where
    M: CompletionModel
        + StreamingChat<M, M::StreamingResponse>
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
        span_emitter: Arc<dyn super::emitter::ReActSpanEmitter>,
        prompt_text: String,
        history_out: Option<&'h mut Vec<Message>>,
    ) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(64);

        let ctx = super::stream_loop::StreamingLoopContext {
            agent: shared.agent.clone(),
            on_thought_cb: shared.on_thought.clone(),
            on_action_cb: shared.on_action.clone(),
            on_observation_cb: shared.on_observation.clone(),
            on_final_cb: shared.on_final.clone(),
            on_error_cb: shared.on_error.clone(),
            react_preamble,
            max_cycles,
            max_retries,
            span_emitter,
            prompt_text,
            history_snapshot,
            context_manager: shared.context_manager.clone(),
            tool_timeout_secs: shared.tool_timeout_secs,
            tx,
            _compaction: PhantomData::<C>,
        };

        tokio::spawn(super::stream_loop::run_streaming_loop(ctx));

        Self {
            rx,
            is_finished: false,
            history_out,
            _phantom: PhantomData,
        }
    }
}

pub(crate) fn extract_prompt_text<'a>(prompt: &'a Message, fallback: &'a str) -> &'a str {
    match prompt {
        Message::User { content } => content.iter().find_map(|c| match c {
            rig_core::message::UserContent::Text(t) => Some(t.text.as_str()),
            _ => None,
        }),
        _ => None,
    }
    .unwrap_or(fallback)
}

pub(crate) async fn send_or_break(
    tx: &tokio::sync::mpsc::Sender<ReActStreamItem>,
    item: ReActStreamItem,
) -> bool {
    tx.send(item).await.is_err()
}

impl<'h, M, P, C> Stream for ReActStream<'h, M, P, C>
where
    M: CompletionModel
        + StreamingChat<M, M::StreamingResponse>
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
                if let ReActStreamItem::Completed {
                    trace: _,
                    ref final_history,
                } = item
                {
                    self.is_finished = true;
                    if let Some(h) = &mut self.history_out {
                        **h = final_history.clone();
                    }
                    return Poll::Ready(Some(item));
                }
                if matches!(&item, ReActStreamItem::Error { .. }) {
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
