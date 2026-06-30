//! Pure data types for the ReAct (Reasoning + Acting) agent loop.
//!
//! This module is the **domain** counterpart to `src/agent/react.rs`, which
//! contains the loop's runtime behaviour. Types here carry no business logic
//! and are fully serde-serialisable so they can be persisted or streamed.

use rig_core::message::Message;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// One ReAct reasoning step emitted by the model in a given cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thought {
    /// Chain-of-thought reasoning text (may originate from a rig
    /// `Reasoning` block when the provider emits them, or inline text).
    pub reasoning: String,
    /// Cycle index within the ReAct loop (0-based).
    pub cycle: usize,
}

/// A tool invocation the model selected in a given cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub tool_name: String,
    /// Raw JSON arguments string as the model produced them.
    pub args: String,
    /// tool_call id if the provider assigned one.
    pub tool_call_id: Option<String>,
    pub cycle: usize,
}

/// The result of executing an `Action`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub tool_name: String,
    /// The serialized tool output string (ToolDyn::call returns String).
    pub result: String,
    /// `true` if the tool returned an error; `result` then carries the error text.
    pub is_error: bool,
    pub cycle: usize,
    pub duration: Duration,
}

/// Terminal outcome of a ReAct run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalAnswer {
    pub text: String,
    /// Total cycles executed before the final answer was produced.
    pub cycles: usize,
}

/// Full serializable trace for one `react()` invocation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReActTrace {
    pub prompt: String,
    pub steps: Vec<ReActStep>,
    pub final_answer: Option<FinalAnswer>,
}

/// One step in a `ReActTrace`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReActStep {
    Thought(Thought),
    Action(Action),
    Observation(Observation),
    FinalAnswer(FinalAnswer),
}

/// A single item yielded by a [`ReActStream`](crate::agent::react::streaming::ReActStream).
#[derive(Debug, Clone)]
pub enum ReActStreamItem {
    /// A new cycle is starting.
    CycleStart { cycle: usize },
    /// A streaming text delta from the model's reasoning/thinking.
    ThoughtDelta { delta: String, cycle: usize },
    /// The model has emitted a tool call.
    Action {
        tool_name: String,
        args: String,
        tool_call_id: Option<String>,
        cycle: usize,
    },
    /// A streaming text delta for tool call arguments.
    ActionArgsDelta {
        tool_name: String,
        delta: String,
        cycle: usize,
    },
    /// A tool has been executed and produced an observation.
    Observation {
        tool_name: String,
        result: String,
        is_error: bool,
        cycle: usize,
        duration: Duration,
    },
    /// A streaming text delta for the final answer.
    FinalAnswerDelta { delta: String, cycle: usize },
    /// The ReAct loop has completed successfully.
    Completed {
        trace: ReActTrace,
        final_history: Vec<Message>,
    },
    /// An error occurred during the ReAct loop.
    Error { error: String },
}
