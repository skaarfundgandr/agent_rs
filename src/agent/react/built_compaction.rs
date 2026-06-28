use std::marker::PhantomData;
use std::sync::Arc;

use rig_core::completion::{CompletionModel, Prompt};
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

use crate::agent::utils::lock_mutex;
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
        let msg = msg.into();
        let mut snapshot = lock_mutex(&self.history).clone();
        if let Some(cm) = self.context_manager.as_deref() {
            cm.compact(&mut snapshot, &msg)
                .await
                .map_err(|e| ReActError::Model(e.to_string()))?;
        }
        run_loop(
            &self.agent,
            &msg,
            &snapshot,
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
            &self.history,
            false,
            self.context_manager.as_deref(),
        )
        .await
    }

    /// Execute a ReAct chat with automatic compaction, **with** history
    /// mutation on success.
    pub async fn chat_compact(&self, msg: impl Into<String>) -> Result<String, ReActError> {
        let msg = msg.into();
        let mut snapshot = lock_mutex(&self.history).clone();
        if let Some(cm) = self.context_manager.as_deref() {
            cm.compact(&mut snapshot, &msg)
                .await
                .map_err(|e| ReActError::Model(e.to_string()))?;
        }
        let trace = run_loop(
            &self.agent,
            &msg,
            &snapshot,
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
            &self.history,
            true,
            self.context_manager.as_deref(),
        )
        .await?;
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
    fn make_stream_shared(&self) -> Arc<super::streaming::StreamShared<M, P, C>> {
        Arc::new(super::streaming::StreamShared {
            agent: self.agent.clone(),
            history: Arc::clone(&self.history),
            tool_timeout_secs: self.tool_timeout_secs,
            on_thought: self.on_thought.as_ref().map(Arc::clone),
            on_action: self.on_action.as_ref().map(Arc::clone),
            on_observation: self.on_observation.as_ref().map(Arc::clone),
            on_final: self.on_final.as_ref().map(Arc::clone),
            on_error: self.on_error.as_ref().map(Arc::clone),
            context_manager: self.context_manager.clone(),
            _compaction: PhantomData,
        })
    }

    /// Stream a ReAct prompt with automatic compaction. Does **not** mutate shared history.
    pub async fn stream_prompt_compact(
        &self,
        msg: impl Into<String>,
    ) -> Result<super::streaming::ReActStream<M, P, C>, ReActError> {
        let msg = msg.into();
        let mut snapshot = lock_mutex(&self.history).clone();
        if let Some(cm) = self.context_manager.as_deref() {
            cm.compact(&mut snapshot, &msg)
                .await
                .map_err(|e| ReActError::Model(e.to_string()))?;
        }
        Ok(super::streaming::ReActStream::new(
            self.make_stream_shared(),
            snapshot,
            self.max_cycles,
            self.max_retries,
            self.react_preamble.clone(),
            Arc::clone(&self.span_emitter),
            false,
            msg,
        ))
    }

    /// Stream a ReAct chat with automatic compaction. Mutates shared history on completion.
    pub async fn stream_chat_compact(
        &self,
        msg: impl Into<String>,
    ) -> Result<super::streaming::ReActStream<M, P, C>, ReActError> {
        let msg = msg.into();
        let mut snapshot = lock_mutex(&self.history).clone();
        if let Some(cm) = self.context_manager.as_deref() {
            cm.compact(&mut snapshot, &msg)
                .await
                .map_err(|e| ReActError::Model(e.to_string()))?;
        }
        Ok(super::streaming::ReActStream::new(
            self.make_stream_shared(),
            snapshot,
            self.max_cycles,
            self.max_retries,
            self.react_preamble.clone(),
            Arc::clone(&self.span_emitter),
            true,
            msg,
        ))
    }
}
