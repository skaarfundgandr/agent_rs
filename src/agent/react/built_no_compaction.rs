use rig_core::completion::CompletionModel;
use rig_core::message::Message;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

use crate::agent::telemetry::TelemetryAccum;
use crate::domain::agent::{
    DetailsState, Extended, ExtendedChatDetails, ExtendedReActTrace, Standard,
};
use crate::domain::errors::ReActError;

use super::built::BuiltReAct;

impl<M> BuiltReAct<M, (), Standard>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
{
    pub async fn prompt(
        &self,
        msg: impl Into<String>,
    ) -> Result<crate::domain::agent::ReActTrace, ReActError> {
        self.run_prompt_impl(msg.into(), None).await
    }

    pub async fn chat(
        &self,
        msg: impl Into<String>,
        history: &mut Vec<Message>,
    ) -> Result<String, ReActError> {
        let (text, working) = self.run_chat_impl(msg.into(), history, None).await?;
        *history = working;
        Ok(text)
    }
}

impl<M> BuiltReAct<M, (), Extended>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Prompt with extended telemetry details.
    pub async fn prompt(&self, msg: impl Into<String>) -> Result<ExtendedReActTrace, ReActError> {
        let mut accum = TelemetryAccum::new();
        let trace = self.run_prompt_impl(msg.into(), Some(&mut accum)).await?;
        let (usage, completion_calls, raw_responses) = accum.finish();
        Ok(ExtendedReActTrace {
            trace,
            usage,
            completion_calls,
            raw_responses,
        })
    }

    /// Chat with extended telemetry details.
    pub async fn chat(
        &self,
        msg: impl Into<String>,
        history: &mut Vec<Message>,
    ) -> Result<ExtendedChatDetails, ReActError> {
        let mut accum = TelemetryAccum::new();
        let (output, working) = self
            .run_chat_impl(msg.into(), history, Some(&mut accum))
            .await?;
        let (usage, completion_calls, raw_responses) = accum.finish();
        *history = working.clone();
        Ok(ExtendedChatDetails {
            output,
            usage,
            completion_calls,
            raw_responses,
            history: working,
        })
    }
}

impl<M, S> BuiltReAct<M, (), S>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    M::StreamingResponse: rig_core::completion::GetTokenUsage + Send,
    S: DetailsState,
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
