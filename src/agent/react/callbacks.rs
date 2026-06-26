use std::sync::Arc;

use crate::domain::agent::{Action, FinalAnswer, Observation, Thought};
use crate::domain::errors::ReActError;

/// Callback invoked when the model emits a reasoning step.
pub type ThoughtCb = Arc<dyn Fn(&Thought) + Send + Sync>;
/// Callback invoked when the model selects a tool call.
pub type ActionCb = Arc<dyn Fn(&Action) + Send + Sync>;
/// Callback invoked after a tool has been executed.
pub type ObservationCb = Arc<dyn Fn(&Observation) + Send + Sync>;
/// Callback invoked when the loop terminates with a final answer.
pub type FinalCb = Arc<dyn Fn(&FinalAnswer) + Send + Sync>;
/// Callback invoked when the loop terminates with an error.
pub type ErrorCb = Arc<dyn Fn(&ReActError) + Send + Sync>;
