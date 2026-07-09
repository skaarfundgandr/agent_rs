use std::marker::PhantomData;
use std::sync::Arc;

use rig_core::agent::{Agent, PromptHook};
use rig_core::completion::{CompletionModel, Prompt};
use rig_core::message::Message;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

use crate::agent::memory::ContextManager;

use super::built::BuiltReAct;
use super::callbacks::{ActionCb, ErrorCb, FinalCb, ObservationCb, ThoughtCb};
use super::emitter::ReActSpanEmitter;

/// Marker: no compaction configured.
pub struct NoCompaction;

/// Compaction configured, parameterized over the compaction model C.
pub struct CompactionConfig<C: Prompt> {
    pub(crate) model: C,
    pub(crate) threshold: usize, // 0 = unset sentinel
    pub(crate) tokenizer: Option<fn(&[Message]) -> usize>,
    pub(crate) compaction_prompt: Option<fn(&str) -> String>,
}

/// Builder for a ReAct loop. Constructed via [`ReActExt::react`](super::ReActExt::react).
///
/// Chain configuration methods, then call [`.build()`](ReActBuilder::build) to
/// obtain a [`BuiltReAct`] that can run prompts and chats.
pub struct ReActBuilder<'a, M, P, CompState = NoCompaction>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    pub agent: &'a Agent<M, P>,
    pub max_cycles: usize,
    pub max_retries: u32,
    pub react_preamble: Option<String>,
    pub span_emitter: Arc<dyn ReActSpanEmitter>,
    pub on_thought: Option<ThoughtCb>,
    pub on_action: Option<ActionCb>,
    pub on_observation: Option<ObservationCb>,
    pub on_final: Option<FinalCb>,
    pub on_error: Option<ErrorCb>,
    pub tool_timeout_secs: u64,
    pub compaction: CompState,
    pub cycle_limit_reminder_msg: Option<String>,
}

// ── Blanket config methods (work for any CompState) ──────────────────────

impl<'a, M, P, CompState> ReActBuilder<'a, M, P, CompState>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Set the maximum number of reasoning-action cycles before the loop
    /// returns [`ReActError::MaxCyclesExceeded`](crate::domain::errors::ReActError::MaxCyclesExceeded).
    ///
    /// # Panics
    ///
    /// Panics if `max_cycles` is 0.
    pub fn max_cycles(self, max_cycles: usize) -> Self {
        assert!(max_cycles > 0, "max_cycles must be at least 1");
        Self { max_cycles, ..self }
    }

    /// Set the maximum number of retries for completion calls within a single cycle.
    ///
    /// Retries occur with exponential backoff (500ms * 2^attempt) and are
    /// triggered on transient completion errors (`HttpError`, `ProviderError`)
    /// and on `MaxTurnsError`. Defaults to 3.
    pub fn max_retries(self, max_retries: u32) -> Self {
        Self {
            max_retries,
            ..self
        }
    }

    /// Set the timeout in seconds for individual tool executions.
    ///
    /// A tool call that exceeds this duration is cancelled and reported as
    /// a timed-out observation. Defaults to 60 seconds.
    ///
    /// In the streaming ReAct path this value is doubled and used as the
    /// timeout for stream initialization and for waiting for each streamed
    /// item from the model.
    pub fn tool_timeout_secs(self, secs: u64) -> Self {
        Self {
            tool_timeout_secs: secs,
            ..self
        }
    }

    /// Set a custom preamble that is prepended to the user prompt.
    ///
    /// Pass `None` to disable the preamble entirely.
    pub fn react_preamble(self, preamble: Option<String>) -> Self {
        Self {
            react_preamble: preamble,
            ..self
        }
    }

    /// Set a custom span emitter for observability integration.
    pub fn with_span_emitter(self, emitter: Arc<dyn ReActSpanEmitter>) -> Self {
        Self {
            span_emitter: emitter,
            ..self
        }
    }

    /// Register a callback invoked when the model emits a reasoning step.
    pub fn on_thought(
        self,
        cb: impl Fn(&crate::domain::agent::Thought) + Send + Sync + 'static,
    ) -> Self {
        Self {
            on_thought: Some(Arc::new(cb)),
            ..self
        }
    }

    /// Register a callback invoked when the model selects a tool call.
    pub fn on_action(
        self,
        cb: impl Fn(&crate::domain::agent::Action) + Send + Sync + 'static,
    ) -> Self {
        Self {
            on_action: Some(Arc::new(cb)),
            ..self
        }
    }

    /// Register a callback invoked after a tool has been executed.
    pub fn on_observation(
        self,
        cb: impl Fn(&crate::domain::agent::Observation) + Send + Sync + 'static,
    ) -> Self {
        Self {
            on_observation: Some(Arc::new(cb)),
            ..self
        }
    }

    /// Register a callback invoked when the loop terminates with a final answer.
    pub fn on_final(
        self,
        cb: impl Fn(&crate::domain::agent::FinalAnswer) + Send + Sync + 'static,
    ) -> Self {
        Self {
            on_final: Some(Arc::new(cb)),
            ..self
        }
    }

    /// Register a callback invoked when the loop terminates with an error.
    pub fn on_error(
        self,
        cb: impl Fn(&crate::domain::errors::ReActError) + Send + Sync + 'static,
    ) -> Self {
        Self {
            on_error: Some(Arc::new(cb)),
            ..self
        }
    }
    /// Set a custom message to be included in the context when the cycle limit is exceeded.
    pub fn set_cycle_limit_reminder_msg(self, msg: Option<String>) -> Self {
        Self {
            cycle_limit_reminder_msg: msg,
            ..self
        }
    }
}

// ── NoCompaction: .with_compaction() and .build() ────────────────────────

impl<'a, M, P> ReActBuilder<'a, M, P, NoCompaction>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Enable automatic context compaction.
    ///
    /// The compaction model defaults to a clone of the agent itself.
    /// Use [`.threshold()`](CompactionConfig) and
    /// [`.compaction_model()`](CompactionConfig) to customise.
    pub fn with_compaction(self) -> ReActBuilder<'a, M, P, CompactionConfig<Agent<M, P>>>
    where
        Agent<M, P>: Clone,
    {
        ReActBuilder {
            agent: self.agent,
            max_cycles: self.max_cycles,
            max_retries: self.max_retries,
            react_preamble: self.react_preamble,
            span_emitter: self.span_emitter,
            on_thought: self.on_thought,
            on_action: self.on_action,
            on_observation: self.on_observation,
            on_final: self.on_final,
            on_error: self.on_error,
            tool_timeout_secs: self.tool_timeout_secs,
            cycle_limit_reminder_msg: self.cycle_limit_reminder_msg,
            compaction: CompactionConfig {
                model: self.agent.clone(),
                threshold: 0,
                tokenizer: None,
                compaction_prompt: None,
            },
        }
    }

    /// Build the [`BuiltReAct`] without context compaction.
    pub fn build(self) -> BuiltReAct<M, P, ()> {
        BuiltReAct {
            agent: self.agent.clone(),
            max_cycles: self.max_cycles,
            max_retries: self.max_retries,
            react_preamble: self.react_preamble,
            span_emitter: self.span_emitter,
            on_thought: self.on_thought,
            on_action: self.on_action,
            on_observation: self.on_observation,
            on_final: self.on_final,
            on_error: self.on_error,
            context_manager: None,
            tool_timeout_secs: self.tool_timeout_secs,
            cycle_limit_reminder_msg: self.cycle_limit_reminder_msg,
            _compaction: PhantomData,
        }
    }
}

// ── CompactionConfig: threshold, model, build ────────────────────────────

impl<'a, M, P, C: Prompt> ReActBuilder<'a, M, P, CompactionConfig<C>>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Set the compaction threshold (must be > 0).
    ///
    /// When the combined token count of history + prompt exceeds this
    /// threshold, the history is compacted before the next cycle.
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
    ) -> ReActBuilder<'a, M, P, CompactionConfig<NewC>> {
        ReActBuilder {
            agent: self.agent,
            max_cycles: self.max_cycles,
            max_retries: self.max_retries,
            react_preamble: self.react_preamble,
            span_emitter: self.span_emitter,
            on_thought: self.on_thought,
            on_action: self.on_action,
            on_observation: self.on_observation,
            on_final: self.on_final,
            on_error: self.on_error,
            tool_timeout_secs: self.tool_timeout_secs,
            cycle_limit_reminder_msg: self.cycle_limit_reminder_msg,
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

    /// Build the [`BuiltReAct`] with context compaction.
    ///
    /// # Panics
    ///
    /// Panics if [`.threshold()`](Self::threshold) has not been called (threshold == 0).
    pub fn build(self) -> BuiltReAct<M, P, C>
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

        BuiltReAct {
            agent: self.agent.clone(),
            max_cycles: self.max_cycles,
            max_retries: self.max_retries,
            react_preamble: self.react_preamble,
            span_emitter: self.span_emitter,
            on_thought: self.on_thought,
            on_action: self.on_action,
            on_observation: self.on_observation,
            on_final: self.on_final,
            on_error: self.on_error,
            cycle_limit_reminder_msg: self.cycle_limit_reminder_msg,
            context_manager: Some(
                Arc::new(ctx) as Arc<dyn crate::agent::react::Compact + Send + Sync>
            ),
            tool_timeout_secs: self.tool_timeout_secs,
            _compaction: PhantomData,
        }
    }
}
