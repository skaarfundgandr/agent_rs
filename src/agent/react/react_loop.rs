use std::sync::Arc;
use std::time::Instant;

use rig_core::agent::{Agent, PromptHook};
use rig_core::completion::{CompletionModel, Prompt};
use rig_core::message::{AssistantContent, Message, ToolCall, ToolResultContent, UserContent};
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

use crate::domain::agent::{Action, FinalAnswer, Observation, ReActStep, ReActTrace, Thought};
use crate::domain::errors::ReActError;

use super::callbacks::{ActionCb, ErrorCb, FinalCb, ObservationCb, ThoughtCb};
use super::emitter::{NoopSpanEmitter, ReActSpanEmitter};
use super::helpers::{detect_final_answer, tool_error_to_string};

/// Default preamble injected before the user prompt to instruct the model
/// to follow the ReAct pattern.
pub const REACT_PREAMBLE: &str = "\
You are an AI agent using the ReAct (Reasoning + Acting) pattern. For each turn:
1. Think step-by-step about what to do next. Emit your reasoning in a `Reasoning` block (or as plain text before any tool call).
2. If you need more information or to take an action, emit a tool call (using the available tools).
3. After receiving the observation, decide whether to take another action or finish.
4. When you are done, respond with plain text that starts with `Final Answer:` followed by your answer. Do NOT emit any tool calls after a Final Answer.

Do not repeat the same action with the same arguments if it has already produced an observation. If a tool returns an error, decide whether to retry with different arguments or to stop.";

/// Builder and executor for a ReAct (Reasoning + Acting) agent loop.
///
/// Construct via [`ReActExt::react`](super::ReActExt::react) or [`ReActLoop::builder`].
pub struct ReActLoop<'a, M, P>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    pub agent: &'a Agent<M, P>,
    pub prompt: String,
    pub history: &'a mut Vec<Message>,
    pub max_cycles: usize,
    pub react_preamble: Option<String>,
    pub on_thought: Option<ThoughtCb>,
    pub on_action: Option<ActionCb>,
    pub on_observation: Option<ObservationCb>,
    pub on_final: Option<FinalCb>,
    pub on_error: Option<ErrorCb>,
    pub span_emitter: Arc<dyn ReActSpanEmitter>,
}

impl<'a, M, P> ReActLoop<'a, M, P>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Create a new builder for a ReAct loop.
    pub fn builder(
        agent: &'a Agent<M, P>,
        prompt: impl Into<String>,
        history: &'a mut Vec<Message>,
    ) -> Self {
        Self {
            agent,
            prompt: prompt.into(),
            history,
            max_cycles: 20,
            react_preamble: None,
            on_thought: None,
            on_action: None,
            on_observation: None,
            on_final: None,
            on_error: None,
            span_emitter: Arc::new(NoopSpanEmitter),
        }
    }

    /// Set the maximum number of reasoning-action cycles before the loop
    /// returns [`ReActError::MaxCyclesExceeded`].
    pub fn max_cycles(mut self, max_cycles: usize) -> Self {
        self.max_cycles = max_cycles;
        self
    }

    /// Set a custom preamble that is prepended to the user prompt.
    ///
    /// Pass `None` to disable the preamble entirely.
    pub fn react_preamble(mut self, preamble: Option<String>) -> Self {
        self.react_preamble = preamble;
        self
    }

    /// Register a callback invoked when the model emits a reasoning step.
    pub fn on_thought(mut self, cb: impl Fn(&Thought) + Send + Sync + 'static) -> Self {
        self.on_thought = Some(Box::new(cb));
        self
    }

    /// Register a callback invoked when the model selects a tool call.
    pub fn on_action(mut self, cb: impl Fn(&Action) + Send + Sync + 'static) -> Self {
        self.on_action = Some(Box::new(cb));
        self
    }

    /// Register a callback invoked after a tool has been executed.
    pub fn on_observation(mut self, cb: impl Fn(&Observation) + Send + Sync + 'static) -> Self {
        self.on_observation = Some(Box::new(cb));
        self
    }

    /// Register a callback invoked when the loop terminates with a final answer.
    pub fn on_final(mut self, cb: impl Fn(&FinalAnswer) + Send + Sync + 'static) -> Self {
        self.on_final = Some(Box::new(cb));
        self
    }

    /// Register a callback invoked when the loop terminates with an error.
    pub fn on_error(mut self, cb: impl Fn(&ReActError) + Send + Sync + 'static) -> Self {
        self.on_error = Some(Box::new(cb));
        self
    }

    /// Set a custom span emitter for observability integration.
    pub fn with_span_emitter(mut self, emitter: Arc<dyn ReActSpanEmitter>) -> Self {
        self.span_emitter = emitter;
        self
    }

    /// Execute the ReAct loop.
    ///
    /// Returns the full [`ReActTrace`] on success, or a [`ReActError`] if
    /// the loop terminates due to max cycles being exceeded, a tool error,
    /// or the model returning an invalid response.
    pub async fn execute(self) -> Result<ReActTrace, ReActError> {
        let mut trace = ReActTrace {
            prompt: self.prompt.clone(),
            steps: Vec::new(),
            final_answer: None,
        };

        // Build effective prompt by prepending the react preamble if present.
        let effective_prompt = match &self.react_preamble {
            Some(preamble) => format!("{preamble}\n\n{}", self.prompt),
            None => self.prompt.clone(),
        };

        let mut current_prompt = Message::User {
            content: rig_core::OneOrMany::one(UserContent::text(effective_prompt)),
        };

        for cycle in 0..self.max_cycles {
            self.span_emitter.emit_cycle_start(cycle);

            // --- 1. Send prompt to the model ------------------------------------
            let response = self
                .agent
                .prompt(current_prompt.clone())
                .with_history(self.history.iter().cloned())
                .extended_details()
                .await
                .map_err(|e| ReActError::Model(e.to_string()))?;

            // Extend the caller's history with the new messages from this turn.
            if let Some(messages) = response.messages {
                self.history.extend(messages);
            }

            // --- 2. Find the last assistant message and extract content ---------
            let assistant_content = match self.history.iter().rev().find_map(|msg| match msg {
                Message::Assistant { content, .. } => Some(content),
                _ => None,
            }) {
                Some(content) => content.clone(),
                None => {
                    // Fallback: no assistant message in history — treat the output
                    // text as a final answer and terminate.
                    let text = response.output.clone();
                    let fa = FinalAnswer {
                        text,
                        cycles: cycle + 1,
                    };
                    trace.steps.push(ReActStep::FinalAnswer(fa.clone()));
                    trace.final_answer = Some(fa.clone());
                    if let Some(cb) = &self.on_final {
                        cb(&fa);
                    }
                    self.span_emitter.emit_cycle_end(cycle, &trace);
                    return Err(ReActError::NoToolCallsAndNoFinalAnswer { cycle });
                }
            };

            // Separate reasoning/text-before-first-tool-call from tool calls
            // and any trailing text.
            let mut reasoning_texts: Vec<String> = Vec::new();
            let mut tool_calls: Vec<&ToolCall> = Vec::new();
            let mut trailing_texts: Vec<String> = Vec::new();
            let mut seen_tool_call = false;

            for item in assistant_content.iter() {
                match item {
                    AssistantContent::Reasoning(r) => {
                        let text = r.display_text();
                        if !text.is_empty() {
                            reasoning_texts.push(text);
                        }
                    }
                    AssistantContent::Text(t) => {
                        if seen_tool_call {
                            trailing_texts.push(t.text.clone());
                        } else {
                            reasoning_texts.push(t.text.clone());
                        }
                    }
                    AssistantContent::ToolCall(tc) => {
                        seen_tool_call = true;
                        tool_calls.push(tc);
                    }
                    AssistantContent::Image(_) => {
                        // Ignore image content in the reasoning extraction.
                    }
                }
            }

            // --- 2.1 Check if no tool calls were made -----------------------------
            // If the model did not request any tool calls, treat the entire response
            // text immediately as the final answer and terminate without emitting a thought.
            if tool_calls.is_empty() {
                let text = response.output.clone();
                if text.is_empty() {
                    let err = ReActError::NoToolCallsAndNoFinalAnswer { cycle };
                    if let Some(cb) = &self.on_error {
                        cb(&err);
                    }
                    self.span_emitter.emit_cycle_end(cycle, &trace);
                    return Err(err);
                }
                let fa = FinalAnswer {
                    text,
                    cycles: cycle + 1,
                };
                trace.steps.push(ReActStep::FinalAnswer(fa.clone()));
                trace.final_answer = Some(fa.clone());
                if let Some(cb) = &self.on_final {
                    cb(&fa);
                }
                self.span_emitter.emit_cycle_end(cycle, &trace);
                return Ok(trace);
            }

            // --- 3. Emit Thoughts -----------------------------------------------
            let thought_text = reasoning_texts.join("\n").trim().to_string();
            if !thought_text.is_empty() {
                let thought = Thought {
                    reasoning: thought_text,
                    cycle,
                };
                if let Some(cb) = &self.on_thought {
                    cb(&thought);
                }
                self.span_emitter.emit_thought(&thought);
                trace.steps.push(ReActStep::Thought(thought));
            }

            // --- 4. Check for Final Answer sentinel ----------------------------
            let full_trailing = trailing_texts.join("").trim().to_string();
            let final_answer = detect_final_answer(&full_trailing);

            if let Some(text) = final_answer {
                let fa = FinalAnswer {
                    text,
                    cycles: cycle + 1,
                };
                trace.steps.push(ReActStep::FinalAnswer(fa.clone()));
                trace.final_answer = Some(fa.clone());
                if let Some(cb) = &self.on_final {
                    cb(&fa);
                }
                self.span_emitter.emit_cycle_end(cycle, &trace);
                return Ok(trace);
            }

            // --- 6. Execute tool calls and build observations -------------------
            let mut next_prompt = None;
            let num_tool_calls = tool_calls.len();

            for (i, tc) in tool_calls.iter().enumerate() {
                let args_str = tc.function.arguments.to_string();
                let action = Action {
                    tool_name: tc.function.name.clone(),
                    args: args_str.clone(),
                    tool_call_id: Some(tc.id.clone()),
                    cycle,
                };
                if let Some(cb) = &self.on_action {
                    cb(&action);
                }
                self.span_emitter.emit_action(&action);
                trace.steps.push(ReActStep::Action(action.clone()));

                let start = Instant::now();
                let result = self
                    .agent
                    .tool_server_handle
                    .call_tool(&tc.function.name, &args_str)
                    .await;
                let duration = start.elapsed();

                let observation = match result {
                    Ok(s) => Observation {
                        tool_name: tc.function.name.clone(),
                        result: s,
                        is_error: false,
                        cycle,
                        duration,
                    },
                    Err(e) => Observation {
                        tool_name: tc.function.name.clone(),
                        result: tool_error_to_string(&e),
                        is_error: true,
                        cycle,
                        duration,
                    },
                };

                if let Some(cb) = &self.on_observation {
                    cb(&observation);
                }
                self.span_emitter.emit_observation(&observation);
                trace
                    .steps
                    .push(ReActStep::Observation(observation.clone()));

                // Inject the tool result as a user message for the next cycle.
                let call_id = tc
                    .call_id
                    .clone()
                    .unwrap_or_else(|| format!("react-cycle-{cycle}"));
                let tool_result_content = ToolResultContent::text(observation.result.clone());
                let user_content = UserContent::tool_result_with_call_id(
                    tc.id.clone(),
                    call_id,
                    rig_core::OneOrMany::one(tool_result_content),
                );
                let msg = Message::User {
                    content: rig_core::OneOrMany::one(user_content),
                };

                // The last tool result is held as the prompt for the next cycle.
                // Any other preceding tool results in this cycle are pushed to self.history.
                if i == num_tool_calls - 1 {
                    next_prompt = Some(msg);
                } else {
                    self.history.push(msg);
                }
            }

            if let Some(msg) = next_prompt {
                current_prompt = msg;
            }

            self.span_emitter.emit_cycle_end(cycle, &trace);
        }

        // If we exhausted the loop without a final answer:
        let err = ReActError::MaxCyclesExceeded {
            cycles: self.max_cycles,
        };
        if let Some(cb) = &self.on_error {
            cb(&err);
        }
        Err(err)
    }
}
