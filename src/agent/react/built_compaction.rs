use rig_core::completion::{CompletionModel, Prompt};
use rig_core::message::Message;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

use crate::agent::telemetry::TelemetryAccum;
use crate::domain::agent::{
    DetailsState, Extended, ExtendedChatDetails, ExtendedReActTrace, Standard,
};
use crate::domain::errors::ReActError;

use super::built::{BuiltReAct, run_loop};

impl<M, C> BuiltReAct<M, C, Standard>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    C: Prompt + WasmCompatSend + WasmCompatSync + 'static,
{
    pub async fn prompt_compact(
        &self,
        msg: impl Into<String>,
    ) -> Result<crate::domain::agent::ReActTrace, ReActError> {
        self.run_prompt_impl(msg.into(), None).await
    }

    pub async fn chat_compact(
        &self,
        msg: impl Into<String>,
        history: &mut Vec<Message>,
    ) -> Result<String, ReActError> {
        let msg = msg.into();
        let mut working = history.clone();
        if let Some(cm) = self.context_manager.as_deref() {
            cm.compact(&mut working, &msg)
                .await
                .map_err(|e| ReActError::Model(e.to_string()))?;
        }
        let (trace, final_working) = run_loop(
            &self.agent,
            &msg,
            &working,
            self.max_cycles,
            self.max_retries,
            self.max_invalid_tool_call_retries,
            self.tool_timeout_secs,
            &self.react_preamble,
            &self.span_emitter,
            &self.on_thought,
            &self.on_action,
            &self.on_observation,
            &self.on_final,
            &self.on_error,
            self.context_manager.as_deref(),
            &self.cycle_limit_reminder_msg,
            None,
        )
        .await?;
        *history = final_working;
        Ok(trace.final_answer.map(|fa| fa.text).unwrap_or_default())
    }
}

impl<M, C> BuiltReAct<M, C, Extended>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    C: Prompt + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Prompt with extended telemetry details.
    pub async fn prompt_compact(
        &self,
        msg: impl Into<String>,
    ) -> Result<ExtendedReActTrace, ReActError> {
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
    pub async fn chat_compact(
        &self,
        msg: impl Into<String>,
        history: &mut Vec<Message>,
    ) -> Result<ExtendedChatDetails, ReActError> {
        let msg = msg.into();
        let mut working = history.clone();
        if let Some(cm) = self.context_manager.as_deref() {
            cm.compact(&mut working, &msg)
                .await
                .map_err(|e| ReActError::Model(e.to_string()))?;
        }
        let mut accum = TelemetryAccum::new();
        let (trace, final_working) = run_loop(
            &self.agent,
            &msg,
            &working,
            self.max_cycles,
            self.max_retries,
            self.max_invalid_tool_call_retries,
            self.tool_timeout_secs,
            &self.react_preamble,
            &self.span_emitter,
            &self.on_thought,
            &self.on_action,
            &self.on_observation,
            &self.on_final,
            &self.on_error,
            self.context_manager.as_deref(),
            &self.cycle_limit_reminder_msg,
            Some(&mut accum),
        )
        .await?;
        let (usage, completion_calls, raw_responses) = accum.finish();
        *history = final_working.clone();
        Ok(ExtendedChatDetails {
            output: trace.final_answer.map(|fa| fa.text).unwrap_or_default(),
            usage,
            completion_calls,
            raw_responses,
            history: final_working,
        })
    }
}

impl<M, C, S> BuiltReAct<M, C, S>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    M::StreamingResponse: rig_core::completion::GetTokenUsage + Send,
    C: Prompt + WasmCompatSend + WasmCompatSync + 'static,
    S: DetailsState,
{
    pub async fn stream_prompt_compact<'h>(
        &self,
        msg: impl Into<String>,
    ) -> Result<super::streaming::ReActStream<'h, M, C>, ReActError> {
        self.run_stream_impl(msg.into())
    }

    pub async fn stream_chat_compact<'h>(
        &self,
        msg: impl Into<String>,
        history: &'h mut Vec<Message>,
    ) -> Result<super::streaming::ReActStream<'h, M, C>, ReActError> {
        let msg = msg.into();
        let snapshot = {
            let mut working = history.clone();
            if let Some(cm) = self.context_manager.as_deref() {
                cm.compact(&mut working, &msg)
                    .await
                    .map_err(|e| ReActError::Model(e.to_string()))?;
            }
            working
        };
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
