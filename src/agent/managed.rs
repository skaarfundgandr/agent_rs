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
use crate::agent::retry::is_retryable;

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
            max_retries: 3,
            compaction: NoCompaction,
        }
    }
}

async fn retry_chat<M, P>(
    agent: &Agent<M, P>,
    msg: &str,
    working: &mut Vec<Message>,
    max_retries: u32,
) -> Result<String, PromptError>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: rig_core::agent::PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    let snapshot = working.clone();
    let mut attempt = 0u32;
    loop {
        attempt += 1;
        *working = snapshot.clone();
        match execute_chat(agent, msg, working).await {
            Ok(s) => return Ok(s),
            Err(e) if is_retryable(&e) && attempt < max_retries => {
                tokio::time::sleep(std::time::Duration::from_millis(
                    500 * 2u64.pow(attempt - 1),
                ))
                .await;
            }
            Err(e) => return Err(e),
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
    max_retries: u32,
    compaction: CompState,
}

// ── Blanket config methods (work for any CompState) ──────────────────────

impl<'a, M, P, CompState> ManagedBuilder<'a, M, P, CompState>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: rig_core::agent::PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Set the maximum number of retries for completion calls on transient errors.
    ///
    /// Retries occur with exponential backoff (500ms * 2^attempt) and are
    /// triggered on `HttpError` and `ProviderError`. Defaults to 3.
    pub fn max_retries(self, max_retries: u32) -> Self {
        Self {
            max_retries,
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
            max_retries: self.max_retries,
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
            max_retries: self.max_retries,
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
            max_retries: self.max_retries,
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
            max_retries: self.max_retries,
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
    max_retries: u32,
    context_manager: Option<Arc<dyn Any + Send + Sync>>,
    _compaction: PhantomData<C>,
}

// ── Shared methods (all CompStates) ──────────────────────────────────────

impl<M, P, C> BuiltManagedAgent<M, P, C>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: rig_core::agent::PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Return the configured `max_retries` limit.
    pub fn max_retries(&self) -> u32 {
        self.max_retries
    }
}

async fn build_stream<'a, M, P>(
    agent: &Agent<M, P>,
    msg: &str,
    working: Vec<Message>,
    working_for_restore: Option<Vec<Message>>,
    history_out: Option<&'a mut Vec<Message>>,
) -> Result<ManagedStream<'a, M::StreamingResponse>, PromptError>
where
    M: CompletionModel
        + StreamingChat<M, M::StreamingResponse>
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
    P: rig_core::agent::PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
    M::StreamingResponse: rig_core::completion::GetTokenUsage + Send + 'static,
    Agent<M, P>: StreamingChat<M, M::StreamingResponse, Hook = P>,
{
    let rig_stream = execute_stream_chat(agent, msg, working)
        .multi_turn(agent.default_max_turns.unwrap_or(20))
        .await;
    Ok(ManagedStream::new(
        rig_stream,
        history_out,
        msg.to_owned(),
        working_for_restore,
    ))
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
        let mut working = Vec::new();
        retry_chat(&self.agent, &msg, &mut working, self.max_retries).await
    }

    /// Execute a chat prompt **with** history mutation on success.
    ///
    /// On success, pushes the user message and assistant response to the
    /// caller's history. On error, the caller's history is not modified.
    pub async fn chat(
        &self,
        msg: impl Into<String>,
        history: &mut Vec<Message>,
    ) -> Result<String, PromptError> {
        let msg = msg.into();
        let mut working = history.clone();
        let result = retry_chat(&self.agent, &msg, &mut working, self.max_retries).await;
        if let Ok(final_text) = &result {
            history.push(Message::user(msg.as_str()));
            history.push(Message::assistant(final_text));
        }
        result
    }

    /// Stream a chat prompt **without** mutating shared history.
    pub async fn stream_prompt(
        &self,
        msg: impl Into<String>,
    ) -> Result<ManagedStream<'_, M::StreamingResponse>, PromptError>
    where
        M: StreamingChat<M, M::StreamingResponse>,
        M::StreamingResponse: rig_core::completion::GetTokenUsage + Send + 'static,
        Agent<M, P>: StreamingChat<M, M::StreamingResponse, Hook = P>,
    {
        let msg = msg.into();
        build_stream(&self.agent, &msg, Vec::new(), None, None).await
    }

    /// Stream a chat prompt **with** history mutation on completion.
    pub async fn stream_chat<'a>(
        &self,
        msg: impl Into<String>,
        history: &'a mut Vec<Message>,
    ) -> Result<ManagedStream<'a, M::StreamingResponse>, PromptError>
    where
        M: StreamingChat<M, M::StreamingResponse>,
        M::StreamingResponse: rig_core::completion::GetTokenUsage + Send + 'static,
        Agent<M, P>: StreamingChat<M, M::StreamingResponse, Hook = P>,
    {
        let msg = msg.into();
        let working = history.clone();
        build_stream(&self.agent, &msg, working, None, Some(history)).await
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
        let mut working = Vec::new();
        if let Some(cm) = self.context_manager() {
            cm.compact_history_if_needed(&mut working, &msg).await?;
        }
        retry_chat(&self.agent, &msg, &mut working, self.max_retries).await
    }

    /// Execute a chat prompt **with** history mutation on success, with compaction.
    pub async fn chat_compact(
        &self,
        msg: impl Into<String>,
        history: &mut Vec<Message>,
    ) -> Result<String, PromptError> {
        let msg = msg.into();
        let mut working = history.clone();
        if let Some(cm) = self.context_manager() {
            cm.compact_history_if_needed(&mut working, &msg).await?;
        }
        let result = retry_chat(&self.agent, &msg, &mut working, self.max_retries).await;
        if let Ok(final_text) = &result {
            *history = working;
            history.push(Message::user(msg.as_str()));
            history.push(Message::assistant(final_text));
        }
        result
    }

    /// Stream a chat prompt **without** mutating shared history, with compaction.
    pub async fn stream_prompt_compact(
        &self,
        msg: impl Into<String>,
    ) -> Result<ManagedStream<'_, M::StreamingResponse>, PromptError>
    where
        M: StreamingChat<M, M::StreamingResponse>,
        M::StreamingResponse: rig_core::completion::GetTokenUsage + Send + 'static,
        Agent<M, P>: StreamingChat<M, M::StreamingResponse, Hook = P>,
    {
        let msg = msg.into();
        let mut working = Vec::new();
        if let Some(cm) = self.context_manager() {
            cm.compact_history_if_needed(&mut working, &msg).await?;
        }
        build_stream(&self.agent, &msg, working, None, None).await
    }

    /// Stream a chat prompt **with** history mutation on completion, with compaction.
    pub async fn stream_chat_compact<'a>(
        &self,
        msg: impl Into<String>,
        history: &'a mut Vec<Message>,
    ) -> Result<ManagedStream<'a, M::StreamingResponse>, PromptError>
    where
        M: StreamingChat<M, M::StreamingResponse>,
        M::StreamingResponse: rig_core::completion::GetTokenUsage + Send + 'static,
        Agent<M, P>: StreamingChat<M, M::StreamingResponse, Hook = P>,
    {
        let msg = msg.into();
        let mut working = history.clone();
        if let Some(cm) = self.context_manager() {
            cm.compact_history_if_needed(&mut working, &msg).await?;
        }
        build_stream(
            &self.agent,
            &msg,
            working.clone(),
            Some(working),
            Some(history),
        )
        .await
    }
}

// ── ManagedStream ────────────────────────────────────────────────────────

/// A stream wrapper for a managed agent chat session.
///
/// Once the stream finishes and yields the `FinalResponse`, the optional
/// history output is updated with the user message and assistant response.
#[must_use = "streams must be polled to completion to update history"]
pub struct ManagedStream<'h, R: Send + 'static> {
    inner: tokio::sync::mpsc::Receiver<Result<MultiTurnStreamItem<R>, StreamingError>>,
    is_finished: bool,
    history_out: Option<&'h mut Vec<Message>>,
    prompt_text: String,
    history_appended: bool,
    replace_baseline: Option<Vec<Message>>,
}

impl<'h, R: Send + 'static> ManagedStream<'h, R> {
    pub fn new<S>(
        stream: S,
        history_out: Option<&'h mut Vec<Message>>,
        prompt_text: String,
        replace_baseline: Option<Vec<Message>>,
    ) -> Self
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
                if tx.send(item).await.is_err() {
                    break;
                }
            }
        });

        Self {
            inner: rx,
            is_finished: false,
            history_out,
            prompt_text,
            history_appended: false,
            replace_baseline,
        }
    }
}

impl<'h, R: Send + 'static> futures::Stream for ManagedStream<'h, R> {
    type Item = Result<MultiTurnStreamItem<R>, StreamingError>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        if self.is_finished {
            return std::task::Poll::Ready(None);
        }

        let this = &mut *self;

        match std::pin::Pin::new(&mut this.inner).poll_recv(cx) {
            std::task::Poll::Ready(Some(item)) => {
                if !this.history_appended
                    && let Ok(MultiTurnStreamItem::FinalResponse(ref final_res)) = item
                {
                    let prompt = this.prompt_text.clone();
                    let response = final_res.response().to_string();
                    if let Some(h) = &mut this.history_out {
                        if let Some(baseline) = this.replace_baseline.take() {
                            **h = baseline;
                        }
                        h.push(Message::user(&prompt));
                        h.push(Message::assistant(response));
                    }
                    this.history_appended = true;
                }
                if matches!(&item, Ok(MultiTurnStreamItem::FinalResponse(_))) {
                    this.is_finished = true;
                }
                std::task::Poll::Ready(Some(item))
            }
            std::task::Poll::Ready(None) => {
                this.is_finished = true;
                std::task::Poll::Ready(None)
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}
