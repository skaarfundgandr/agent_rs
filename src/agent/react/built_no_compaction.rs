use rig_core::completion::CompletionModel;
use rig_core::message::Message;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

use crate::domain::errors::ReActError;

use super::built::BuiltReAct;

impl<M> BuiltReAct<M, ()>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
{
    pub async fn prompt(
        &self,
        msg: impl Into<String>,
    ) -> Result<crate::domain::agent::ReActTrace, ReActError> {
        self.run_prompt_impl(msg.into()).await
    }

    pub async fn chat(
        &self,
        msg: impl Into<String>,
        history: &mut Vec<Message>,
    ) -> Result<String, ReActError> {
        self.run_chat_impl(msg.into(), history).await
    }
}

impl<M> BuiltReAct<M, ()>
where
    M: CompletionModel
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
    M::StreamingResponse: rig_core::completion::GetTokenUsage + Send,
{
    pub fn stream_prompt<'h>(
        &self,
        msg: impl Into<String>,
    ) -> Result<super::streaming::ReActStream<'h, M, ()>, ReActError> {
        self.run_stream_impl(msg.into())
    }

    pub fn stream_chat<'h>(
        &self,
        msg: impl Into<String>,
        history: &'h mut Vec<Message>,
    ) -> Result<super::streaming::ReActStream<'h, M, ()>, ReActError> {
        let msg = msg.into();
        let snapshot = history.clone();
        Ok(super::streaming::ReActStream::new(
            self.make_stream_shared(),
            snapshot,
            self.max_cycles,
            self.max_retries,
            self.react_preamble.clone(),
            std::sync::Arc::clone(&self.span_emitter),
            msg,
            Some(history),
        ))
    }
}
