//! Internal telemetry capture plumbing for the opt-in `.extended_details()` agent APIs.

use std::sync::{Arc, Mutex};

use rig_core::agent::{AgentHook, CompletionCall, Flow, HookContext, PromptResponse, StepEvent};
use rig_core::completion::{CompletionModel, Usage};
use serde_json::Value;

/// Accumulates usage, completion calls, and raw provider payloads for one run.
#[derive(Default)]
pub(crate) struct TelemetryAccum {
    usage: Usage,
    completion_calls: Vec<CompletionCall>,
    raw: Arc<Mutex<Vec<Value>>>,
}

impl TelemetryAccum {
    /// Creates an empty accumulator.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Folds a prompt response's usage and completion calls into the run totals.
    pub(crate) fn fold_response(&mut self, resp: &PromptResponse) {
        self.usage += resp.usage;
        let start = self.completion_calls.len();
        for call in &resp.completion_calls {
            self.completion_calls
                .push(CompletionCall::new(start + call.call_index, call.usage));
        }
    }

    /// Clones the shared raw-payload buffer handle for the capture hook.
    pub(crate) fn raw_handle(&self) -> Arc<Mutex<Vec<Value>>> {
        self.raw.clone()
    }

    /// Current length of the raw-payload buffer.
    pub(crate) fn raw_len(&self) -> usize {
        self.raw.lock().map_or(0, |raw| raw.len())
    }

    /// Truncates the raw-payload buffer back to `len`.
    pub(crate) fn truncate_raw(&mut self, len: usize) {
        if let Ok(mut raw) = self.raw.lock() {
            raw.truncate(len);
        }
    }

    /// Consumes the accumulator and returns usage, completion calls, and raw payloads.
    pub(crate) fn finish(self) -> (Usage, Vec<CompletionCall>, Vec<Value>) {
        let raw = self.raw.lock().map_or(Vec::new(), |raw| raw.clone());
        (self.usage, self.completion_calls, raw)
    }
}

/// AgentHook that captures provider-native completion payloads into a shared buffer.
pub(crate) struct CaptureTelemetryHook {
    raw: Arc<Mutex<Vec<Value>>>,
}

impl CaptureTelemetryHook {
    /// Wraps the shared raw-payload buffer.
    pub(crate) fn new(raw: Arc<Mutex<Vec<Value>>>) -> Self {
        Self { raw }
    }
}

impl<M: CompletionModel> AgentHook<M> for CaptureTelemetryHook {
    async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
        if let StepEvent::CompletionResponse { response, .. } = event
            && let Ok(value) = serde_json::to_value(&response.raw_response)
            && let Ok(mut raw) = self.raw.lock()
        {
            raw.push(value);
        }
        Flow::cont()
    }
}
