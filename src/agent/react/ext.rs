use std::marker::PhantomData;
use std::sync::Arc;

use rig_core::agent::Agent;
use rig_core::completion::CompletionModel;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

use super::builder::{NoCompaction, ReActBuilder};
use super::emitter::NoopSpanEmitter;

/// Extension trait that adds a `.react()` method to rig [`Agent`]s.
pub trait ReActExt<M, P>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: rig_core::agent::PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Start building a ReAct loop for this agent.
    fn react(&self) -> ReActBuilder<'_, M, P, NoCompaction>;
}

impl<M, P> ReActExt<M, P> for Agent<M, P>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: rig_core::agent::PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    fn react(&self) -> ReActBuilder<'_, M, P, NoCompaction> {
        ReActBuilder {
            agent: self,
            max_cycles: 20,
            max_retries: 3,
            react_preamble: None,
            initial_history: Vec::new(),
            span_emitter: Arc::new(NoopSpanEmitter),
            on_thought: None,
            on_action: None,
            on_observation: None,
            on_final: None,
            on_error: None,
            tool_timeout_secs: 60,
            compaction: NoCompaction,
            _phantom: PhantomData,
        }
    }
}
