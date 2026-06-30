use std::marker::PhantomData;
use std::sync::Arc;

use rig_core::completion::CompletionModel;
use rig_core::message::Message;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

use crate::domain::errors::ReActError;

use super::built::{BuiltReAct, run_loop};

impl<M, P> BuiltReAct<M, P, ()>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: rig_core::agent::PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Execute a ReAct prompt **without** mutating shared history.
    pub async fn prompt(
        &self,
        msg: impl Into<String>,
    ) -> Result<crate::domain::agent::ReActTrace, ReActError> {
        let msg = msg.into();
        let (trace, _) = run_loop(
            &self.agent,
            &msg,
            &[],
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
            None,
        )
        .await?;
        Ok(trace)
    }

    /// Execute a ReAct chat **with** caller-owned history mutation on success.
    pub async fn chat(
        &self,
        msg: impl Into<String>,
        history: &mut Vec<Message>,
    ) -> Result<String, ReActError> {
        let msg = msg.into();
        let (trace, working) = run_loop(
            &self.agent,
            &msg,
            history,
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
            None,
        )
        .await?;
        *history = working;
        Ok(trace.final_answer.map(|fa| fa.text).unwrap_or_default())
    }
}

impl<M, P> BuiltReAct<M, P, ()>
where
    M: CompletionModel
        + rig_core::streaming::StreamingChat<M, M::StreamingResponse>
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
    P: rig_core::agent::PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
    M::StreamingResponse: rig_core::completion::GetTokenUsage + Send,
{
    /// Stream a ReAct prompt. Does **not** mutate shared history.
    pub fn stream_prompt<'h>(
        &self,
        msg: impl Into<String>,
    ) -> Result<super::streaming::ReActStream<'h, M, P, ()>, ReActError> {
        let msg = msg.into();
        Ok(super::streaming::ReActStream::new(
            Arc::new(super::streaming::StreamShared {
                agent: self.agent.clone(),
                tool_timeout_secs: self.tool_timeout_secs,
                on_thought: self.on_thought.as_ref().map(Arc::clone),
                on_action: self.on_action.as_ref().map(Arc::clone),
                on_observation: self.on_observation.as_ref().map(Arc::clone),
                on_final: self.on_final.as_ref().map(Arc::clone),
                on_error: self.on_error.as_ref().map(Arc::clone),
                context_manager: None,
                _compaction: PhantomData,
            }),
            Vec::new(),
            self.max_cycles,
            self.max_retries,
            self.react_preamble.clone(),
            Arc::clone(&self.span_emitter),
            msg,
            None,
        ))
    }

    /// Stream a ReAct chat. Caller-owned history is written back on completion.
    pub fn stream_chat<'h>(
        &self,
        msg: impl Into<String>,
        history: &'h mut Vec<Message>,
    ) -> Result<super::streaming::ReActStream<'h, M, P, ()>, ReActError> {
        let msg = msg.into();
        let snapshot = history.clone();
        Ok(super::streaming::ReActStream::new(
            Arc::new(super::streaming::StreamShared {
                agent: self.agent.clone(),
                tool_timeout_secs: self.tool_timeout_secs,
                on_thought: self.on_thought.as_ref().map(Arc::clone),
                on_action: self.on_action.as_ref().map(Arc::clone),
                on_observation: self.on_observation.as_ref().map(Arc::clone),
                on_final: self.on_final.as_ref().map(Arc::clone),
                on_error: self.on_error.as_ref().map(Arc::clone),
                context_manager: None,
                _compaction: PhantomData,
            }),
            snapshot,
            self.max_cycles,
            self.max_retries,
            self.react_preamble.clone(),
            Arc::clone(&self.span_emitter),
            msg,
            Some(history),
        ))
    }
}
