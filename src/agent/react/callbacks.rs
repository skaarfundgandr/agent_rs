use crate::domain::agent::{Action, FinalAnswer, Observation, Thought};
use crate::domain::errors::ReActError;

/// Callback invoked when the model emits a reasoning step.
pub type ThoughtCb = Box<dyn Fn(&Thought) + Send + Sync>;
/// Callback invoked when the model selects a tool call.
pub type ActionCb = Box<dyn Fn(&Action) + Send + Sync>;
/// Callback invoked after a tool has been executed.
pub type ObservationCb = Box<dyn Fn(&Observation) + Send + Sync>;
/// Callback invoked when the loop terminates with a final answer.
pub type FinalCb = Box<dyn Fn(&FinalAnswer) + Send + Sync>;
/// Callback invoked when the loop terminates with an error.
pub type ErrorCb = Box<dyn Fn(&ReActError) + Send + Sync>;
