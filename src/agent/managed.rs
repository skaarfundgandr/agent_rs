use std::any::Any;
use std::marker::PhantomData;
use std::sync::Arc;

use rig_core::agent::{Agent, MultiTurnStreamItem, StreamingError};
use rig_core::completion::{CompletionModel, Prompt, PromptError};
use rig_core::message::Message;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

use crate::agent::invalid_tool::{InvalidToolPolicy, InvalidToolRecoveryHook};
use crate::agent::memory::ContextManager;
use crate::agent::model::chat::{execute_chat, execute_stream_chat};
use crate::agent::retry::is_retryable;

use super::react::{CompactionConfig, NoCompaction};

/// Extension trait to build a managed agent with optional context compaction.
pub trait ManagedExt<M>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Start building a managed agent with optional compaction.
    fn managed(&self) -> ManagedBuilder<'_, M, NoCompaction>;
}

impl<M> ManagedExt<M> for Agent<M>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
{
    fn managed(&self) -> ManagedBuilder<'_, M, NoCompaction> {
        ManagedBuilder {
            agent: self,
            max_retries: 3,
            invalid_tool_policy: InvalidToolPolicy::Skip,
            max_invalid_tool_call_retries: 0,
            invalid_tool_retries_explicit: false,
            compaction: NoCompaction,
        }
    }
}

async fn retry_chat<M>(
    agent: &Agent<M>,
    msg: &str,
    working: &mut Vec<Message>,
    max_retries: u32,
    max_invalid_tool_call_retries: u32,
) -> Result<String, PromptError>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
{
    let snapshot = working.clone();
    let mut attempt = 0u32;
    loop {
        attempt += 1;
        *working = snapshot.clone();
        match execute_chat(agent, msg, working, max_invalid_tool_call_retries).await {
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
pub struct ManagedBuilder<'a, M, CompState = NoCompaction>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
{
    agent: &'a Agent<M>,
    max_retries: u32,
    invalid_tool_policy: InvalidToolPolicy,
    max_invalid_tool_call_retries: u32,
    invalid_tool_retries_explicit: bool,
    compaction: CompState,
}

// ── Blanket config methods (work for any CompState) ──────────────────────

impl<'a, M, CompState> ManagedBuilder<'a, M, CompState>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
{
    pub fn max_retries(self, max_retries: u32) -> Self {
        Self {
            max_retries,
            ..self
        }
    }

    pub fn invalid_tool_policy(mut self, policy: InvalidToolPolicy) -> Self {
        self.invalid_tool_policy = policy;
        if matches!(policy, InvalidToolPolicy::Retry) && !self.invalid_tool_retries_explicit {
            self.max_invalid_tool_call_retries = 2;
        }
        self
    }

    pub fn max_invalid_tool_call_retries(mut self, n: u32) -> Self {
        self.max_invalid_tool_call_retries = n;
        self.invalid_tool_retries_explicit = true;
        self
    }
}

// ── NoCompaction: .with_compaction() and .build() ────────────────────────

impl<'a, M> ManagedBuilder<'a, M, NoCompaction>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
{
    pub fn with_compaction(self) -> ManagedBuilder<'a, M, CompactionConfig<Agent<M>>>
    where
        Agent<M>: Clone,
    {
        ManagedBuilder {
            agent: self.agent,
            max_retries: self.max_retries,
            invalid_tool_policy: self.invalid_tool_policy,
            max_invalid_tool_call_retries: self.max_invalid_tool_call_retries,
            invalid_tool_retries_explicit: self.invalid_tool_retries_explicit,
            compaction: CompactionConfig {
                model: self.agent.clone(),
                threshold: 0,
                tokenizer: None,
                compaction_prompt: None,
            },
        }
    }

    pub fn build(self) -> BuiltManagedAgent<M, ()> {
        let mut agent = self.agent.clone();
        agent
            .hooks
            .push(InvalidToolRecoveryHook::new(self.invalid_tool_policy));
        BuiltManagedAgent {
            agent,
            max_retries: self.max_retries,
            invalid_tool_policy: self.invalid_tool_policy,
            max_invalid_tool_call_retries: self.max_invalid_tool_call_retries,
            context_manager: None,
            _compaction: PhantomData,
        }
    }
}

// ── CompactionConfig: threshold, model, tokenizer, prompt, build ──────────

impl<'a, M, C: Prompt> ManagedBuilder<'a, M, CompactionConfig<C>>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
{
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

    pub fn compaction_model<NewC: Prompt>(
        self,
        model: NewC,
    ) -> ManagedBuilder<'a, M, CompactionConfig<NewC>> {
        ManagedBuilder {
            agent: self.agent,
            max_retries: self.max_retries,
            invalid_tool_policy: self.invalid_tool_policy,
            max_invalid_tool_call_retries: self.max_invalid_tool_call_retries,
            invalid_tool_retries_explicit: self.invalid_tool_retries_explicit,
            compaction: CompactionConfig {
                model,
                threshold: self.compaction.threshold,
                tokenizer: self.compaction.tokenizer,
                compaction_prompt: self.compaction.compaction_prompt,
            },
        }
    }

    pub fn compaction_prompt(self, formatter: fn(&str) -> String) -> Self {
        Self {
            compaction: CompactionConfig {
                compaction_prompt: Some(formatter),
                ..self.compaction
            },
            ..self
        }
    }

    pub fn tokenizer(self, estimator: fn(&[Message]) -> usize) -> Self {
        Self {
            compaction: CompactionConfig {
                tokenizer: Some(estimator),
                ..self.compaction
            },
            ..self
        }
    }

    pub fn build(self) -> BuiltManagedAgent<M, C>
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

        let mut agent = self.agent.clone();
        agent
            .hooks
            .push(InvalidToolRecoveryHook::new(self.invalid_tool_policy));
        BuiltManagedAgent {
            agent,
            max_retries: self.max_retries,
            invalid_tool_policy: self.invalid_tool_policy,
            max_invalid_tool_call_retries: self.max_invalid_tool_call_retries,
            context_manager: Some(Arc::new(ctx)),
            _compaction: PhantomData,
        }
    }
}

// ── BuiltManagedAgent ────────────────────────────────────────────────────

/// A fully configured managed agent, ready to run prompts and chats.
pub struct BuiltManagedAgent<M, C = ()>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
{
    agent: Agent<M>,
    max_retries: u32,
    invalid_tool_policy: InvalidToolPolicy,
    max_invalid_tool_call_retries: u32,
    context_manager: Option<Arc<dyn Any + Send + Sync>>,
    _compaction: PhantomData<C>,
}

// ── Shared methods (all CompStates) ──────────────────────────────────────

impl<M, C> BuiltManagedAgent<M, C>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Return the configured `max_retries` limit.
    pub fn max_retries(&self) -> u32 {
        self.max_retries
    }

    /// Return the configured invalid tool policy.
    pub fn invalid_tool_policy(&self) -> InvalidToolPolicy {
        self.invalid_tool_policy
    }

    /// Return the configured max invalid tool call retries budget.
    pub fn max_invalid_tool_call_retries(&self) -> u32 {
        self.max_invalid_tool_call_retries
    }
}

async fn build_stream<'a, M>(
    agent: &Agent<M>,
    msg: &str,
    working: Vec<Message>,
    working_for_restore: Option<Vec<Message>>,
    history_out: Option<&'a mut Vec<Message>>,
    max_invalid_tool_call_retries: u32,
) -> Result<ManagedStream<'a, M::StreamingResponse>, PromptError>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    M::StreamingResponse: rig_core::completion::GetTokenUsage + Send + 'static,
{
    let mut rig_stream =
        execute_stream_chat(agent, msg, working).max_turns(agent.default_max_turns.unwrap_or(20));
    if max_invalid_tool_call_retries > 0 {
        rig_stream =
            rig_stream.max_invalid_tool_call_retries(max_invalid_tool_call_retries as usize);
    }
    let rig_stream = rig_stream.await;
    Ok(ManagedStream::new(
        rig_stream,
        history_out,
        msg.to_owned(),
        working_for_restore,
    ))
}

// ── No-compaction methods ────────────────────────────────────────────────

impl<M> BuiltManagedAgent<M, ()>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
{
    pub async fn prompt(&self, msg: impl Into<String>) -> Result<String, PromptError> {
        let msg = msg.into();
        let mut working = Vec::new();
        retry_chat(
            &self.agent,
            &msg,
            &mut working,
            self.max_retries,
            self.max_invalid_tool_call_retries,
        )
        .await
    }

    pub async fn chat(
        &self,
        msg: impl Into<String>,
        history: &mut Vec<Message>,
    ) -> Result<String, PromptError> {
        let msg = msg.into();
        let mut working = history.clone();
        let result = retry_chat(
            &self.agent,
            &msg,
            &mut working,
            self.max_retries,
            self.max_invalid_tool_call_retries,
        )
        .await;
        if let Ok(final_text) = &result {
            history.push(Message::user(msg.as_str()));
            history.push(Message::assistant(final_text));
        }
        result
    }

    pub async fn stream_prompt(
        &self,
        msg: impl Into<String>,
    ) -> Result<ManagedStream<'_, M::StreamingResponse>, PromptError>
    where
        M::StreamingResponse: rig_core::completion::GetTokenUsage + Send + 'static,
    {
        let msg = msg.into();
        build_stream(
            &self.agent,
            &msg,
            Vec::new(),
            None,
            None,
            self.max_invalid_tool_call_retries,
        )
        .await
    }

    pub async fn stream_chat<'a>(
        &self,
        msg: impl Into<String>,
        history: &'a mut Vec<Message>,
    ) -> Result<ManagedStream<'a, M::StreamingResponse>, PromptError>
    where
        M::StreamingResponse: rig_core::completion::GetTokenUsage + Send + 'static,
    {
        let msg = msg.into();
        let working = history.clone();
        build_stream(
            &self.agent,
            &msg,
            working,
            None,
            Some(history),
            self.max_invalid_tool_call_retries,
        )
        .await
    }
}

// ── With-compaction methods ──────────────────────────────────────────────

impl<M, C> BuiltManagedAgent<M, C>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    C: Prompt + WasmCompatSend + WasmCompatSync + 'static,
{
    fn context_manager(&self) -> Option<&ContextManager<C>> {
        self.context_manager
            .as_ref()
            .and_then(|arc| arc.downcast_ref::<ContextManager<C>>())
    }

    pub async fn prompt_compact(&self, msg: impl Into<String>) -> Result<String, PromptError> {
        let msg = msg.into();
        let mut working = Vec::new();
        if let Some(cm) = self.context_manager() {
            cm.compact_history_if_needed(&mut working, &msg).await?;
        }
        retry_chat(
            &self.agent,
            &msg,
            &mut working,
            self.max_retries,
            self.max_invalid_tool_call_retries,
        )
        .await
    }

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
        let result = retry_chat(
            &self.agent,
            &msg,
            &mut working,
            self.max_retries,
            self.max_invalid_tool_call_retries,
        )
        .await;
        if let Ok(final_text) = &result {
            *history = working;
            history.push(Message::user(msg.as_str()));
            history.push(Message::assistant(final_text));
        }
        result
    }

    pub async fn stream_prompt_compact(
        &self,
        msg: impl Into<String>,
    ) -> Result<ManagedStream<'_, M::StreamingResponse>, PromptError>
    where
        M::StreamingResponse: rig_core::completion::GetTokenUsage + Send + 'static,
    {
        let msg = msg.into();
        let mut working = Vec::new();
        if let Some(cm) = self.context_manager() {
            cm.compact_history_if_needed(&mut working, &msg).await?;
        }
        build_stream(
            &self.agent,
            &msg,
            working,
            None,
            None,
            self.max_invalid_tool_call_retries,
        )
        .await
    }

    pub async fn stream_chat_compact<'a>(
        &self,
        msg: impl Into<String>,
        history: &'a mut Vec<Message>,
    ) -> Result<ManagedStream<'a, M::StreamingResponse>, PromptError>
    where
        M::StreamingResponse: rig_core::completion::GetTokenUsage + Send + 'static,
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
            self.max_invalid_tool_call_retries,
        )
        .await
    }
}

// ── ManagedStream ────────────────────────────────────────────────────────

/// A stream wrapper for a managed agent chat session.
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
                    let response = final_res.to_string();
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
