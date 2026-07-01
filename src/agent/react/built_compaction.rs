use rig_core::completion::{CompletionModel, Prompt};
use rig_core::message::Message;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

use crate::domain::errors::ReActError;

use super::built::{BuiltReAct, run_loop};

impl<M, P, C> BuiltReAct<M, P, C>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: rig_core::agent::PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
    C: Prompt + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Execute a ReAct prompt with automatic compaction, **without** mutating
    /// shared history.
    pub async fn prompt_compact(
        &self,
        msg: impl Into<String>,
    ) -> Result<crate::domain::agent::ReActTrace, ReActError> {
        self.run_prompt_impl(msg.into()).await
    }

    /// Execute a ReAct chat with automatic compaction, **with** caller-owned
    /// history mutation on success.
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
            self.tool_timeout_secs,
            &self.react_preamble,
            &self.span_emitter,
            &self.on_thought,
            &self.on_action,
            &self.on_observation,
            &self.on_final,
            &self.on_error,
            self.context_manager.as_deref(),
        )
        .await?;
        *history = final_working;
        Ok(trace.final_answer.map(|fa| fa.text).unwrap_or_default())
    }
}

impl<M, P, C> BuiltReAct<M, P, C>
where
    M: CompletionModel
        + rig_core::streaming::StreamingChat<M, M::StreamingResponse>
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
    P: rig_core::agent::PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
    M::StreamingResponse: rig_core::completion::GetTokenUsage + Send,
    C: Prompt + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Stream a ReAct prompt with automatic compaction. Does **not** mutate shared history.
    pub async fn stream_prompt_compact<'h>(
        &self,
        msg: impl Into<String>,
    ) -> Result<super::streaming::ReActStream<'h, M, P, C>, ReActError> {
        self.run_stream_impl(msg.into())
    }

    /// Stream a ReAct chat with automatic compaction. Caller-owned history is written back on completion.
    pub async fn stream_chat_compact<'h>(
        &self,
        msg: impl Into<String>,
        history: &'h mut Vec<Message>,
    ) -> Result<super::streaming::ReActStream<'h, M, P, C>, ReActError> {
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
