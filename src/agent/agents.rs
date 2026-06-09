use futures::stream::Stream;
use rig::agent::{Agent, MultiTurnStreamItem, StreamingError};
use rig::completion::{CompletionModel, Prompt, PromptError};
use rig::message::Message;
use rig::streaming::StreamingChat;
use rig::wasm_compat::{WasmCompatSend, WasmCompatSync};
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::oneshot;
use tracing::Instrument;

use crate::agent::memory::context::ContextManager;

const DEFAULT_MAX_TURNS: usize = 20;

/// An agent wrapper that automatically compacts conversation history
/// when it exceeds a specified token threshold.
///
/// Under the hood, this uses a `ContextManager` which leverages the `cl100k_base`
/// BPE tokenizer to accurately count the tokens of the conversation history.
/// If the threshold is crossed, a compaction model summarizes the history.
pub struct ContextManagedAgent<M: CompletionModel, C: Prompt, P = ()>
where
    P: rig::agent::PromptHook<M>,
{
    inner: Agent<M, P>,
    context_manager: ContextManager<C>,
}

impl<
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    C: Prompt + WasmCompatSend + WasmCompatSync + 'static,
    P: rig::agent::PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
> ContextManagedAgent<M, C, P>
{
    /// Send a chat prompt and automatically manage the context history.
    ///
    /// The history is mutated in-place:
    /// - If the estimated token count of the history and prompt exceeds the threshold,
    ///   the history is compacted into a summary (represented as a single system message).
    /// - The new prompt and the agent's response are automatically appended to the history.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The user input prompt text.
    /// * `history` - A mutable reference to the conversation history vector.
    ///
    /// # Returns
    ///
    /// Returns the response text from the LLM.
    ///
    /// # Errors
    ///
    /// Returns `PromptError` if either context compaction or the chat turn fails.
    pub async fn chat(
        &self,
        prompt: &str,
        history: &mut Vec<Message>,
    ) -> Result<String, PromptError> {
        let chat_span = tracing::info_span!("agent_chat", prompt = prompt);

        async move {
            // 1. Compact if threshold is exceeded
            self.context_manager
                .compact_history_if_needed(history, prompt)
                .await?;

            // 2. Call the model
            let response =
                crate::agent::model::chat::execute_chat(&self.inner, prompt, history).await?;

            Ok(response)
        }
        .instrument(chat_span)
        .await
    }

    /// Send a chat prompt and automatically manage the context history using owned history.
    ///
    /// This method avoids mutating borrowed history and returns the updated history instead.
    /// - If the estimated token count exceeds the threshold, the history is compacted into a summary.
    /// - The new prompt and the agent's response are automatically appended to the returned history.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The user input prompt text.
    /// * `history` - The owned conversation history vector.
    ///
    /// # Returns
    ///
    /// Returns a tuple containing:
    /// 1. The response text from the LLM.
    /// 2. The updated conversation history vector.
    ///
    /// # Errors
    ///
    /// Returns `PromptError` if either context compaction or the chat turn fails.
    pub async fn chat_with_owned_history(
        &self,
        prompt: &str,
        mut history: Vec<Message>,
    ) -> Result<(String, Vec<Message>), PromptError> {
        let chat_span = tracing::info_span!("agent_chat_with_owned_history", prompt = prompt);

        async move {
            // 1. Compact if threshold is exceeded
            self.context_manager
                .compact_history_if_needed(&mut history, prompt)
                .await?;

            // 2. Call the model
            let response =
                crate::agent::model::chat::execute_chat(&self.inner, prompt, &mut history).await?;

            Ok((response, history))
        }
        .instrument(chat_span)
        .await
    }

    /// Stream a chat prompt and automatically manage the context history.
    ///
    /// Compacts history in-place if needed, then executes a streaming chat.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The user input prompt text.
    /// * `history` - A slice of current conversation messages.
    ///
    /// # Returns
    ///
    /// Returns a tuple containing:
    /// 1. A stream wrapper `ContextManagedChatStream` yielding chunks from the LLM.
    /// 2. A oneshot `Receiver` that resolves to the updated history vector once the stream is fully consumed.
    ///
    /// # Errors
    ///
    /// Returns `PromptError` if context compaction fails.
    pub async fn stream_chat(
        &self,
        prompt: &str,
        history: &[Message],
    ) -> Result<
        (
            ContextManagedChatStream<
                impl Stream<Item = Result<MultiTurnStreamItem<M::StreamingResponse>, StreamingError>>
                + Unpin,
                M::StreamingResponse,
            >,
            oneshot::Receiver<Vec<Message>>,
        ),
        PromptError,
    >
    where
        M::StreamingResponse: rig::completion::GetTokenUsage,
        Agent<M, P>: StreamingChat<M, M::StreamingResponse, Hook = P>,
    {
        let mut cloned_history = history.to_vec();
        self.context_manager
            .compact_history_if_needed(&mut cloned_history, prompt)
            .await?;

        // Snapshot the compacted history for later merging with current-turn messages
        let original_history = cloned_history.clone();

        let rig_stream =
            crate::agent::model::chat::execute_stream_chat(&self.inner, prompt, cloned_history)
                .multi_turn(self.inner.default_max_turns.unwrap_or(DEFAULT_MAX_TURNS))
                .await;

        let (tx, rx) = oneshot::channel();
        let stream = ContextManagedChatStream::new(rig_stream, tx, original_history);

        Ok((stream, rx))
    }

    /// Stream a chat prompt with owned history, automatically managing context history.
    ///
    /// Compacts history if needed, then executes a streaming chat.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The user input prompt text.
    /// * `history` - The owned conversation history vector.
    ///
    /// # Returns
    ///
    /// Returns a tuple containing:
    /// 1. A stream wrapper `ContextManagedChatStream` yielding chunks from the LLM.
    /// 2. A oneshot `Receiver` that resolves to the updated history vector once the stream is fully consumed.
    ///
    /// # Errors
    ///
    /// Returns `PromptError` if context compaction fails.
    pub async fn stream_chat_with_owned_history(
        &self,
        prompt: &str,
        mut history: Vec<Message>,
    ) -> Result<
        (
            ContextManagedChatStream<
                impl Stream<Item = Result<MultiTurnStreamItem<M::StreamingResponse>, StreamingError>>
                + Unpin,
                M::StreamingResponse,
            >,
            oneshot::Receiver<Vec<Message>>,
        ),
        PromptError,
    >
    where
        M::StreamingResponse: rig::completion::GetTokenUsage,
        Agent<M, P>: StreamingChat<M, M::StreamingResponse, Hook = P>,
    {
        self.context_manager
            .compact_history_if_needed(&mut history, prompt)
            .await?;

        // Snapshot the compacted history for later merging with current-turn messages
        let original_history = history.clone();

        let rig_stream =
            crate::agent::model::chat::execute_stream_chat(&self.inner, prompt, history)
                .multi_turn(self.inner.default_max_turns.unwrap_or(DEFAULT_MAX_TURNS))
                .await;

        let (tx, rx) = oneshot::channel();
        let stream = ContextManagedChatStream::new(rig_stream, tx, original_history);

        Ok((stream, rx))
    }

    /// Registers a custom token estimator callback.
    ///
    /// # Arguments
    ///
    /// * `estimator` - A function pointer that estimates the token count of a message slice.
    ///
    /// # Returns
    ///
    /// Returns `Self` with the updated estimator callback.
    pub fn with_token_estimator(mut self, estimator: fn(&[Message]) -> usize) -> Self {
        self.context_manager = self.context_manager.with_token_estimator(estimator);
        self
    }

    /// Registers a custom compaction prompt formatter callback.
    ///
    /// # Arguments
    ///
    /// * `formatter` - A function pointer that takes the history JSON text representation and returns the custom prompt for the compaction LLM.
    ///
    /// # Returns
    ///
    /// Returns `Self` with the updated prompt formatter callback.
    pub fn with_compaction_prompt_formatter(mut self, formatter: fn(&str) -> String) -> Self {
        self.context_manager = self.context_manager.with_compaction_prompt_formatter(formatter);
        self
    }


    /// Access the underlying standard Rig Agent.
    ///
    /// # Returns
    ///
    /// Returns a reference to the inner standard Rig `Agent` instance.
    pub fn agent(&self) -> &Agent<M, P> {
        &self.inner
    }
}

/// Removes `AssistantContent::Reasoning` blocks from the history.
///
/// Use this on history that will be persisted and fed back to the model
/// on subsequent turns. Reasoning blocks are ephemeral chain-of-thought
/// that waste tokens when persisted.
///
/// Reasoning is still yielded by the stream for real-time display —
/// this function only affects the history vector.
///
/// # Example
///
/// ```ignore
/// let clean = agent_rs::agent::strip_reasoning_from_history(history);
/// ```
pub fn strip_reasoning_from_history(history: Vec<Message>) -> Vec<Message> {
    history
        .into_iter()
        .filter_map(|msg| match msg {
            Message::Assistant { id, content } => {
                let filtered: Vec<_> = content
                    .into_iter()
                    .filter(|item| {
                        !matches!(item, rig::message::AssistantContent::Reasoning(_))
                    })
                    .collect();
                match rig::OneOrMany::many(filtered) {
                    Ok(new_content) => Some(Message::Assistant {
                        id,
                        content: new_content,
                    }),
                    Err(_) => None, // message was only reasoning
                }
            }
            other => Some(other),
        })
        .collect()
}

/// A stream wrapper for a context-managed agent chat session.
///
/// Once the stream finishes and yields the `FinalResponse`, the updated history
/// (original history + current-turn messages) is sent to the oneshot channel.
#[must_use = "streams must be polled to completion to update history"]
pub struct ContextManagedChatStream<S, R> {
    inner: S,
    history_tx: Option<oneshot::Sender<Vec<Message>>>,
    original_history: Vec<Message>,
    _phantom: std::marker::PhantomData<R>,
}

impl<S, R> ContextManagedChatStream<S, R> {
    /// Creates a new `ContextManagedChatStream`.
    ///
    /// # Arguments
    ///
    /// * `inner` - The underlying stream yielding standard Rig multi-turn items.
    /// * `history_tx` - A oneshot sender to transmit the updated history vector once the stream finishes.
    /// * `original_history` - The conversation history snapshot sent to the LLM for this turn.
    ///
    /// # Returns
    ///
    /// Returns a new instance of `ContextManagedChatStream`.
    pub fn new(
        inner: S,
        history_tx: oneshot::Sender<Vec<Message>>,
        original_history: Vec<Message>,
    ) -> Self {
        Self {
            inner,
            history_tx: Some(history_tx),
            original_history,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<S, R> Stream for ContextManagedChatStream<S, R>
where
    S: Stream<Item = Result<MultiTurnStreamItem<R>, StreamingError>> + Unpin,
    R: Unpin,
{
    type Item = Result<MultiTurnStreamItem<R>, StreamingError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let poll_res = Pin::new(&mut this.inner).poll_next(cx);
        if let Poll::Ready(Some(Ok(MultiTurnStreamItem::FinalResponse(final_res)))) = &poll_res
            && let Some(tx) = this.history_tx.take() {
                let current_turn = final_res.history().map(|h| h.to_vec()).unwrap_or_default();
                let mut full_history = this.original_history.clone();
                full_history.extend(current_turn);
                let _ = tx.send(full_history);
            }
        poll_res
    }
}

/// Extension trait to easily add context management to an existing rig Agent
pub trait AgentContextExt<M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static, P = ()>
where
    P: rig::agent::PromptHook<M>,
{
    /// Wraps the agent in a ContextManagedAgent that will automatically
    /// compact conversation history using the provided compaction model
    /// when the estimated token count exceeds the threshold.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The threshold token count above which conversation history is compacted.
    /// * `compaction_model` - The LLM/compactor model used to summarize history.
    ///
    /// # Returns
    ///
    /// Returns a `ContextManagedAgent` wrapping this agent.
    fn with_compaction<C: Prompt + WasmCompatSend + WasmCompatSync + 'static>(
        self,
        threshold: usize,
        compaction_model: C,
    ) -> ContextManagedAgent<M, C, P>;
}

impl<
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: rig::agent::PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
> AgentContextExt<M, P> for Agent<M, P>
{
    /// Wraps the agent in a ContextManagedAgent that will automatically
    /// compact conversation history using the provided compaction model
    /// when the estimated token count exceeds the threshold.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The threshold token count above which conversation history is compacted.
    /// * `compaction_model` - The LLM/compactor model used to summarize history.
    ///
    /// # Returns
    ///
    /// Returns a `ContextManagedAgent` wrapping this agent.
    fn with_compaction<C: Prompt + WasmCompatSend + WasmCompatSync + 'static>(
        self,
        threshold: usize,
        compaction_model: C,
    ) -> ContextManagedAgent<M, C, P> {
        ContextManagedAgent {
            inner: self,
            context_manager: ContextManager::new(threshold, compaction_model),
        }
    }
}
