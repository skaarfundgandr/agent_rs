use std::marker::PhantomData;
use std::sync::Arc;

use rig_core::agent::Agent;
use rig_core::completion::{CompletionModel, Prompt};
use rig_core::message::Message;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

use crate::agent::invalid_tool::{InvalidToolPolicy, InvalidToolRecoveryHook};
use crate::agent::memory::ContextManager;

use super::built::BuiltReAct;
use super::callbacks::{ActionCb, ErrorCb, FinalCb, ObservationCb, ThoughtCb};
use super::emitter::ReActSpanEmitter;

/// Typestate marker for a ReAct builder without context compaction.
///
/// The default `CompState` of [`ReActBuilder`]; switch to
/// [`CompactionConfig`] via [`ReActBuilder::with_compaction`].
pub struct NoCompaction;

/// Typestate carrying compaction configuration: the compaction model, the
/// token threshold, and optional tokenizer / prompt-formatter overrides.
///
/// Not constructible directly; reached via [`ReActBuilder::with_compaction`].
pub struct CompactionConfig<C: Prompt> {
    pub(crate) model: C,
    pub(crate) threshold: usize,
    pub(crate) tokenizer: Option<fn(&[Message]) -> usize>,
    pub(crate) compaction_prompt: Option<fn(&str) -> String>,
}

/// Builder for a ReAct (Reasoning + Acting) loop, created via
/// [`ReActExt::react`](crate::agent::react::ReActExt::react).
///
/// `CompState` is a typestate: [`NoCompaction`] by default, switched to
/// [`CompactionConfig`] via [`Self::with_compaction`].
pub struct ReActBuilder<'a, M, CompState = NoCompaction>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
{
    pub agent: &'a Agent<M>,
    pub max_cycles: usize,
    pub max_retries: u32,
    pub invalid_tool_policy: InvalidToolPolicy,
    pub max_invalid_tool_call_retries: u32,
    pub invalid_tool_retries_explicit: bool,
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

impl<'a, M, CompState> ReActBuilder<'a, M, CompState>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Set the maximum number of ReAct cycles (default 20).
    ///
    /// # Panics
    ///
    /// Panics if `max_cycles` is 0.
    pub fn max_cycles(self, max_cycles: usize) -> Self {
        assert!(max_cycles > 0, "max_cycles must be at least 1");
        Self { max_cycles, ..self }
    }

    /// Set the number of retries per model call on retryable errors (default 3).
    pub fn max_retries(self, max_retries: u32) -> Self {
        Self {
            max_retries,
            ..self
        }
    }

    /// Set the recovery policy for invalid tool names (default
    /// [`InvalidToolPolicy::Skip`]). Selecting
    /// [`Retry`](InvalidToolPolicy::Retry) without an explicit
    /// [`Self::max_invalid_tool_call_retries`] raises the retry budget to 2.
    pub fn invalid_tool_policy(mut self, policy: InvalidToolPolicy) -> Self {
        self.invalid_tool_policy = policy;
        if matches!(policy, InvalidToolPolicy::Retry) && !self.invalid_tool_retries_explicit {
            self.max_invalid_tool_call_retries = 2;
        }
        self
    }

    /// Set the explicit budget of retries for invalid tool calls. Marks the
    /// budget as explicitly configured so a later
    /// [`Self::invalid_tool_policy`] change does not override it.
    pub fn max_invalid_tool_call_retries(mut self, n: u32) -> Self {
        self.max_invalid_tool_call_retries = n;
        self.invalid_tool_retries_explicit = true;
        self
    }

    /// Set the per-tool-call timeout in seconds (default 60); a tool call
    /// exceeding it is treated as failed.
    pub fn tool_timeout_secs(self, secs: u64) -> Self {
        Self {
            tool_timeout_secs: secs,
            ..self
        }
    }

    /// Set an optional preamble prepended to the user prompt on every cycle.
    pub fn react_preamble(self, preamble: Option<String>) -> Self {
        Self {
            react_preamble: preamble,
            ..self
        }
    }

    /// Replace the span emitter used for cycle/step telemetry (default
    /// [`NoopSpanEmitter`](crate::agent::react::NoopSpanEmitter)).
    pub fn with_span_emitter(self, emitter: Arc<dyn ReActSpanEmitter>) -> Self {
        Self {
            span_emitter: emitter,
            ..self
        }
    }

    /// Register a callback invoked for each reasoning/thought step.
    pub fn on_thought(
        self,
        cb: impl Fn(&crate::domain::agent::Thought) + Send + Sync + 'static,
    ) -> Self {
        Self {
            on_thought: Some(Arc::new(cb)),
            ..self
        }
    }

    /// Register a callback invoked for each tool action taken by the agent.
    pub fn on_action(
        self,
        cb: impl Fn(&crate::domain::agent::Action) + Send + Sync + 'static,
    ) -> Self {
        Self {
            on_action: Some(Arc::new(cb)),
            ..self
        }
    }

    /// Register a callback invoked for each tool result observed by the loop.
    pub fn on_observation(
        self,
        cb: impl Fn(&crate::domain::agent::Observation) + Send + Sync + 'static,
    ) -> Self {
        Self {
            on_observation: Some(Arc::new(cb)),
            ..self
        }
    }

    /// Register a callback invoked when the loop produces its final answer.
    pub fn on_final(
        self,
        cb: impl Fn(&crate::domain::agent::FinalAnswer) + Send + Sync + 'static,
    ) -> Self {
        Self {
            on_final: Some(Arc::new(cb)),
            ..self
        }
    }

    /// Register a callback invoked when the loop encounters a
    /// [`ReActError`](crate::domain::errors::ReActError).
    pub fn on_error(
        self,
        cb: impl Fn(&crate::domain::errors::ReActError) + Send + Sync + 'static,
    ) -> Self {
        Self {
            on_error: Some(Arc::new(cb)),
            ..self
        }
    }

    /// Set an optional reminder appended as a system message in the final two
    /// cycles before the cycle limit is hit.
    pub fn set_cycle_limit_reminder_msg(self, msg: Option<String>) -> Self {
        Self {
            cycle_limit_reminder_msg: msg,
            ..self
        }
    }
}

impl<'a, M> ReActBuilder<'a, M, NoCompaction>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Switch to the compaction-enabled builder state, using the agent's own
    /// model (when `Clone`) as the compaction model. Configure
    /// [`Self::threshold`] before [`build`](Self::build).
    pub fn with_compaction(self) -> ReActBuilder<'a, M, CompactionConfig<Agent<M>>>
    where
        Agent<M>: Clone,
    {
        ReActBuilder {
            agent: self.agent,
            max_cycles: self.max_cycles,
            max_retries: self.max_retries,
            invalid_tool_policy: self.invalid_tool_policy,
            max_invalid_tool_call_retries: self.max_invalid_tool_call_retries,
            invalid_tool_retries_explicit: self.invalid_tool_retries_explicit,
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

    /// Finalize the builder into a [`BuiltReAct`] without compaction,
    /// installing an [`InvalidToolRecoveryHook`] with the configured policy.
    pub fn build(self) -> BuiltReAct<M, ()> {
        let mut agent = self.agent.clone();
        agent
            .hooks
            .push(InvalidToolRecoveryHook::new(self.invalid_tool_policy));
        BuiltReAct {
            agent,
            max_cycles: self.max_cycles,
            max_retries: self.max_retries,
            invalid_tool_policy: self.invalid_tool_policy,
            max_invalid_tool_call_retries: self.max_invalid_tool_call_retries,
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

impl<'a, M, C: Prompt> ReActBuilder<'a, M, CompactionConfig<C>>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Set the history token threshold that triggers compaction.
    ///
    /// # Panics
    ///
    /// Panics if `n` is 0.
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

    /// Replace the model used to generate compaction summaries.
    pub fn compaction_model<NewC: Prompt>(
        self,
        model: NewC,
    ) -> ReActBuilder<'a, M, CompactionConfig<NewC>> {
        ReActBuilder {
            agent: self.agent,
            max_cycles: self.max_cycles,
            max_retries: self.max_retries,
            invalid_tool_policy: self.invalid_tool_policy,
            max_invalid_tool_call_retries: self.max_invalid_tool_call_retries,
            invalid_tool_retries_explicit: self.invalid_tool_retries_explicit,
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

    /// Set a custom formatter turning the serialized history into the
    /// compaction summary prompt (defaults to the crate's built-in prompt).
    pub fn compaction_prompt(self, formatter: fn(&str) -> String) -> Self {
        Self {
            compaction: CompactionConfig {
                compaction_prompt: Some(formatter),
                ..self.compaction
            },
            ..self
        }
    }

    /// Set a custom token estimator for the history; defaults to the crate's
    /// built-in character-based heuristic.
    pub fn tokenizer(self, estimator: fn(&[Message]) -> usize) -> Self {
        Self {
            compaction: CompactionConfig {
                tokenizer: Some(estimator),
                ..self.compaction
            },
            ..self
        }
    }

    /// Finalize the builder into a [`BuiltReAct`] with compaction enabled,
    /// wiring up a [`ContextManager`](crate::agent::memory::ContextManager)
    /// and an [`InvalidToolRecoveryHook`] with the configured policy.
    ///
    /// # Panics
    ///
    /// Panics if [`threshold`](Self::threshold) was not configured first.
    pub fn build(self) -> BuiltReAct<M, C>
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
        BuiltReAct {
            agent,
            max_cycles: self.max_cycles,
            max_retries: self.max_retries,
            invalid_tool_policy: self.invalid_tool_policy,
            max_invalid_tool_call_retries: self.max_invalid_tool_call_retries,
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
