use rig_core::agent::Agent;
use rig_core::completion::CompletionModel;
use rig_core::message::Message;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

use super::react_loop::ReActLoop;

/// Extension trait that adds a `.react()` method to rig [`Agent`]s.
pub trait ReActExt<M, P>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: rig_core::agent::PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Start building a ReAct loop for this agent.
    fn react<'a>(
        &'a self,
        prompt: impl Into<String>,
        history: &'a mut Vec<Message>,
    ) -> ReActLoop<'a, M, P>;
}

impl<M, P> ReActExt<M, P> for Agent<M, P>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: rig_core::agent::PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    fn react<'a>(
        &'a self,
        prompt: impl Into<String>,
        history: &'a mut Vec<Message>,
    ) -> ReActLoop<'a, M, P> {
        ReActLoop::builder(self, prompt, history)
    }
}
