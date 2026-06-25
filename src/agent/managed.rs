use std::any::Any;
use std::marker::PhantomData;
use std::sync::Arc;

use rig_core::agent::{Agent, MultiTurnStreamItem, StreamingError};
use rig_core::completion::{CompletionModel, Prompt, PromptError};
use rig_core::message::Message;
use rig_core::streaming::StreamingChat;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

use crate::agent::memory::ContextManager;
use crate::agent::model::chat::{execute_chat, execute_stream_chat};
use crate::agent::utils::{Mutex, lock_mutex};

use super::react::{CompactionConfig, NoCompaction};

/// Extension trait to build a managed agent with optional context compaction.
pub trait ManagedExt<M, P>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: rig_core::agent::PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Start building a managed agent with optional compaction.
    fn managed(&self) -> ManagedBuilder<'_, M, P, NoCompaction>;
}

impl<M, P> ManagedExt<M, P> for Agent<M, P>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: rig_core::agent::PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    fn managed(&self) -> ManagedBuilder<'_, M, P, NoCompaction> {
        ManagedBuilder {
            agent: self,
            initial_history: Vec::new(),
            compaction: NoCompaction,
        }
    }
}

// ── Builder ──────────────────────────────────────────────────────────────

/// Builder for a managed agent. Constructed via [`ManagedExt::managed`].
///
/// Chain configuration methods, then call [`.build()`](ManagedBuilder::build)
/// to obtain a [`BuiltManagedAgent`].
pub struct ManagedBuilder<'a, M, P, CompState = NoCompaction>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: rig_core::agent::PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    agent: &'a Agent<M, P>,
    initial_history: Vec<Message>,
    compaction: CompState,
}

// ── Blanket config methods (work for any CompState) ──────────────────────

impl<'a, M, P, CompState> ManagedBuilder<'a, M, P, CompState>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: rig_core::agent::PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Seed the initial conversation history.
    pub fn with_history(self, history: Vec<Message>) -> Self {
        Self {
            initial_history: history,
            ..self
        }
    }
}

// ── NoCompaction: .with_compaction() and .build() ────────────────────────

impl<'a, M, P> ManagedBuilder<'a, M, P, NoCompaction>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: rig_core::agent::PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Enable automatic context compaction.
    ///
    /// The compaction model defaults to a clone of the agent itself.
    /// Use [`.threshold()`](CompactionConfig), [`.compaction_model()`](CompactionConfig),
    /// [`.compaction_prompt()`](CompactionConfig), and [`.tokenizer()`](CompactionConfig)
    /// to customise.
    pub fn with_compaction(self) -> ManagedBuilder<'a, M, P, CompactionConfig<Agent<M, P>>>
    where
        Agent<M, P>: Clone,
    {
        ManagedBuilder {
            agent: self.agent,
            initial_history: self.initial_history,
            compaction: CompactionConfig {
                model: self.agent.clone(),
                threshold: 0,
                tokenizer: None,
                compaction_prompt: None,
            },
        }
    }

    /// Build the [`BuiltManagedAgent`] without context compaction.
    pub fn build(self) -> BuiltManagedAgent<M, P, ()> {
        BuiltManagedAgent {
            agent: self.agent.clone(),
            history: Arc::new(Mutex::new(self.initial_history)),
            context_manager: None,
            _compaction: PhantomData,
        }
    }
}

// ── CompactionConfig: threshold, model, tokenizer, prompt, build ──────────

impl<'a, M, P, C: Prompt> ManagedBuilder<'a, M, P, CompactionConfig<C>>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: rig_core::agent::PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Set the compaction threshold (must be > 0).
    pub fn threshold(self, n: usize) -> Self {
        assert!(n > 0, "threshold must be greater than 0");
        Self {
            compaction: CompactionConfig {
                threshold: n,
                ..self.compaction
            },
            ..self
        }
    }

    /// Replace the compaction model.
    pub fn compaction_model<NewC: Prompt>(
        self,
        model: NewC,
    ) -> ManagedBuilder<'a, M, P, CompactionConfig<NewC>> {
        ManagedBuilder {
            agent: self.agent,
            initial_history: self.initial_history,
            compaction: CompactionConfig {
                model,
                threshold: self.compaction.threshold,
                tokenizer: self.compaction.tokenizer,
                compaction_prompt: self.compaction.compaction_prompt,
            },
        }
    }

    /// Set a custom compaction prompt formatter.
    pub fn compaction_prompt(self, formatter: fn(&str) -> String) -> Self {
        Self {
            compaction: CompactionConfig {
                compaction_prompt: Some(formatter),
                ..self.compaction
            },
            ..self
        }
    }

    /// Set a custom token estimator for compaction threshold checks.
    pub fn tokenizer(self, estimator: fn(&[Message]) -> usize) -> Self {
        Self {
            compaction: CompactionConfig {
                tokenizer: Some(estimator),
                ..self.compaction
            },
            ..self
        }
    }

    /// Build the [`BuiltManagedAgent`] with context compaction.
    ///
    /// # Panics
    ///
    /// Panics if [`.threshold()`](Self::threshold) has not been called (threshold == 0).
    pub fn build(self) -> BuiltManagedAgent<M, P, C>
    where
        C: WasmCompatSend + WasmCompatSync + 'static,
    {
        assert!(
            self.compaction.threshold > 0,
            "threshold must be configured before build()"
        );

        let mut ctx = ContextManager::new(self.compaction.threshold, self.compaction.model);
        if let Some(estimator) = self.compaction.tokenizer {
            ctx = ctx.with_token_estimator(estimator);
        }
        if let Some(formatter) = self.compaction.compaction_prompt {
            ctx = ctx.with_compaction_prompt_formatter(formatter);
        }

        BuiltManagedAgent {
            agent: self.agent.clone(),
            history: Arc::new(Mutex::new(self.initial_history)),
            context_manager: Some(Arc::new(ctx)),
            _compaction: PhantomData,
        }
    }
}

// ── BuiltManagedAgent ────────────────────────────────────────────────────

/// A fully configured managed agent, ready to run prompts and chats.
///
/// Constructed by calling [`.build()`](ManagedBuilder::build) on a
/// [`ManagedBuilder`].
pub struct BuiltManagedAgent<M, P, C = ()>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: rig_core::agent::PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    agent: Agent<M, P>,
    history: Arc<Mutex<Vec<Message>>>,
    context_manager: Option<Arc<dyn Any + Send + Sync>>,
    _compaction: PhantomData<C>,
}

// ── Shared methods (all CompStates) ──────────────────────────────────────

impl<M, P, C> BuiltManagedAgent<M, P, C>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: rig_core::agent::PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Return a snapshot of the current conversation history.
    pub fn history(&self) -> Vec<Message> {
        lock_mutex(&self.history).clone()
    }
}

// ── No-compaction methods ────────────────────────────────────────────────

impl<M, P> BuiltManagedAgent<M, P, ()>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: rig_core::agent::PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Execute a chat prompt **without** mutating shared history.
    pub async fn prompt(&self, msg: impl Into<String>) -> Result<String, PromptError> {
        let msg = msg.into();
        let mut working = lock_mutex(&self.history).clone();
        execute_chat(&self.agent, &msg, &mut working).await
    }

    /// Execute a chat prompt **with** history mutation on success.
    ///
    /// On success, the shared history is replaced with the new working history.
    /// On error, the shared history is not modified.
    pub async fn chat(&self, msg: impl Into<String>) -> Result<String, PromptError> {
        let msg = msg.into();
        let mut working = lock_mutex(&self.history).clone();
        let result = execute_chat(&self.agent, &msg, &mut working).await;
        if result.is_ok() {
            *lock_mutex(&self.history) = working;
        }
        result
    }

    /// Stream a chat prompt **without** mutating shared history.
    pub async fn stream_prompt(
        &self,
        msg: impl Into<String>,
    ) -> Result<ManagedStream<M::StreamingResponse>, PromptError>
    where
        M: StreamingChat<M, M::StreamingResponse>,
        M::StreamingResponse: rig_core::completion::GetTokenUsage + Send + 'static,
        Agent<M, P>: StreamingChat<M, M::StreamingResponse, Hook = P>,
    {
        let msg = msg.into();
        let working = lock_mutex(&self.history).clone();
        let rig_stream = execute_stream_chat(&self.agent, &msg, working)
            .multi_turn(self.agent.default_max_turns.unwrap_or(20))
            .await;
        Ok(ManagedStream::new(rig_stream, None))
    }

    /// Stream a chat prompt **with** history mutation on completion.
    pub async fn stream_chat(
        &self,
        msg: impl Into<String>,
    ) -> Result<ManagedStream<M::StreamingResponse>, PromptError>
    where
        M: StreamingChat<M, M::StreamingResponse>,
        M::StreamingResponse: rig_core::completion::GetTokenUsage + Send + 'static,
        Agent<M, P>: StreamingChat<M, M::StreamingResponse, Hook = P>,
    {
        let msg = msg.into();
        let working = lock_mutex(&self.history).clone();
        let rig_stream = execute_stream_chat(&self.agent, &msg, working)
            .multi_turn(self.agent.default_max_turns.unwrap_or(20))
            .await;
        Ok(ManagedStream::new(
            rig_stream,
            Some(Arc::clone(&self.history)),
        ))
    }
}

// ── With-compaction methods ──────────────────────────────────────────────

impl<M, P, C> BuiltManagedAgent<M, P, C>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: rig_core::agent::PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
    C: Prompt + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Downcast the type-erased context manager back to `&ContextManager<C>`.
    fn context_manager(&self) -> Option<&ContextManager<C>> {
        self.context_manager
            .as_ref()
            .and_then(|arc| arc.downcast_ref::<ContextManager<C>>())
    }

    /// Execute a chat prompt **without** mutating shared history, with compaction.
    pub async fn prompt_compact(&self, msg: impl Into<String>) -> Result<String, PromptError> {
        let msg = msg.into();
        let mut working = lock_mutex(&self.history).clone();
        if let Some(cm) = self.context_manager() {
            cm.compact_history_if_needed(&mut working, &msg).await?;
        }
        execute_chat(&self.agent, &msg, &mut working).await
    }

    /// Execute a chat prompt **with** history mutation on success, with compaction.
    pub async fn chat_compact(&self, msg: impl Into<String>) -> Result<String, PromptError> {
        let msg = msg.into();
        let mut working = lock_mutex(&self.history).clone();
        if let Some(cm) = self.context_manager() {
            cm.compact_history_if_needed(&mut working, &msg).await?;
        }
        let result = execute_chat(&self.agent, &msg, &mut working).await;
        if result.is_ok() {
            *lock_mutex(&self.history) = working;
        }
        result
    }

    /// Stream a chat prompt **without** mutating shared history, with compaction.
    pub async fn stream_prompt_compact(
        &self,
        msg: impl Into<String>,
    ) -> Result<ManagedStream<M::StreamingResponse>, PromptError>
    where
        M: StreamingChat<M, M::StreamingResponse>,
        M::StreamingResponse: rig_core::completion::GetTokenUsage + Send + 'static,
        Agent<M, P>: StreamingChat<M, M::StreamingResponse, Hook = P>,
    {
        let msg = msg.into();
        let mut working = lock_mutex(&self.history).clone();
        if let Some(cm) = self.context_manager() {
            cm.compact_history_if_needed(&mut working, &msg).await?;
        }
        let rig_stream = execute_stream_chat(&self.agent, &msg, working)
            .multi_turn(self.agent.default_max_turns.unwrap_or(20))
            .await;
        Ok(ManagedStream::new(rig_stream, None))
    }

    /// Stream a chat prompt **with** history mutation on completion, with compaction.
    pub async fn stream_chat_compact(
        &self,
        msg: impl Into<String>,
    ) -> Result<ManagedStream<M::StreamingResponse>, PromptError>
    where
        M: StreamingChat<M, M::StreamingResponse>,
        M::StreamingResponse: rig_core::completion::GetTokenUsage + Send + 'static,
        Agent<M, P>: StreamingChat<M, M::StreamingResponse, Hook = P>,
    {
        let msg = msg.into();
        let mut working = lock_mutex(&self.history).clone();
        if let Some(cm) = self.context_manager() {
            cm.compact_history_if_needed(&mut working, &msg).await?;
        }
        let rig_stream = execute_stream_chat(&self.agent, &msg, working)
            .multi_turn(self.agent.default_max_turns.unwrap_or(20))
            .await;
        Ok(ManagedStream::new(
            rig_stream,
            Some(Arc::clone(&self.history)),
        ))
    }
}

// ── ManagedStream ────────────────────────────────────────────────────────

/// A stream wrapper for a managed agent chat session.
///
/// Once the stream finishes and yields the `FinalResponse`, the shared history
/// (if provided) is updated with the final accumulated messages.
#[must_use = "streams must be polled to completion to update history"]
pub struct ManagedStream<R: Send + 'static> {
    inner: tokio::sync::mpsc::Receiver<Result<MultiTurnStreamItem<R>, StreamingError>>,
    is_finished: bool,
}

impl<R: Send + 'static> ManagedStream<R> {
    pub fn new<S>(stream: S, shared_history: Option<Arc<Mutex<Vec<Message>>>>) -> Self
    where
        S: futures::Stream<Item = Result<MultiTurnStreamItem<R>, StreamingError>>
            + Send
            + Unpin
            + 'static,
    {
        let (tx, rx) = tokio::sync::mpsc::channel(64);

        tokio::spawn(async move {
            use futures::StreamExt;
            let mut stream = stream;

            while let Some(item) = stream.next().await {
                if let Ok(MultiTurnStreamItem::FinalResponse(ref final_res)) = item
                    && let Some(ref shared_history) = shared_history
                {
                    let final_history = final_res.history().map(|h| h.to_vec()).unwrap_or_default();
                    if !final_history.is_empty() {
                        *lock_mutex(shared_history) = final_history;
                    }
                }

                if tx.send(item).await.is_err() {
                    break;
                }
            }
        });

        Self {
            inner: rx,
            is_finished: false,
        }
    }
}

impl<R: Send + 'static> futures::Stream for ManagedStream<R> {
    type Item = Result<MultiTurnStreamItem<R>, StreamingError>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        if self.is_finished {
            return std::task::Poll::Ready(None);
        }

        match std::pin::Pin::new(&mut self.inner).poll_recv(cx) {
            std::task::Poll::Ready(Some(item)) => {
                if matches!(&item, Ok(MultiTurnStreamItem::FinalResponse(_))) {
                    self.is_finished = true;
                }
                std::task::Poll::Ready(Some(item))
            }
            std::task::Poll::Ready(None) => {
                self.is_finished = true;
                std::task::Poll::Ready(None)
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}
