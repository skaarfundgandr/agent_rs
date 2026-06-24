//! Execution helper methods for [`ReActLoop`].
//!
//! Contains the `ParsedContent` / `PromptRecovery` types and a split
//! `impl ReActLoop` block with: content parsing, thought emission,
//! final-answer finalization, error recovery, and tool-call execution.

use std::time::{Duration, Instant};
use tracing::Instrument;

use rig_core::OneOrMany;
use rig_core::completion::PromptError;
use rig_core::message::{AssistantContent, Message, ToolCall, ToolResultContent, UserContent};

use crate::domain::agent::{Action, FinalAnswer, Observation, ReActStep, ReActTrace, Thought};
use crate::domain::errors::ReActError;

use super::helpers::{find_assistant_with_tool_call, tool_error_to_string};
use super::react_loop::ReActLoop;

// ---------------------------------------------------------------------------
// ParsedContent — result of classifying assistant content items
// ---------------------------------------------------------------------------

/// Reasoning text, tool calls, and trailing text extracted from an assistant
/// message's content items.
pub(super) struct ParsedContent<'a> {
    pub reasoning_texts: Vec<String>,
    pub tool_calls: Vec<&'a ToolCall>,
    pub trailing_texts: Vec<String>,
}

impl<'a> ParsedContent<'a> {
    pub fn parse(content: &'a OneOrMany<AssistantContent>) -> Self {
        let mut reasoning_texts = Vec::new();
        let mut tool_calls = Vec::new();
        let mut trailing_texts = Vec::new();
        let mut seen_tool_call = false;

        for item in content.iter() {
            match item {
                AssistantContent::Reasoning(r) => {
                    let text = r.display_text();
                    if !text.is_empty() {
                        reasoning_texts.push(text);
                    }
                }
                AssistantContent::Text(t) => {
                    // Pre-tool-call text is treated as reasoning. Most providers
                    // surface chain-of-thought in a dedicated `Reasoning` block,
                    // but some emit it as plain `Text` before the first tool
                    // call. Trailing text (after any tool call) is treated as
                    // the model's natural-language response.
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
                AssistantContent::Image(_) => {}
            }
        }

        Self {
            reasoning_texts,
            tool_calls,
            trailing_texts,
        }
    }
}

// ---------------------------------------------------------------------------
// PromptRecovery — outcome of attempting to recover from a PromptError
// ---------------------------------------------------------------------------

/// The result of attempting to recover from a `PromptError` inside the ReAct
/// loop. `Recovered` means the loop should continue with a new prompt;
/// `Abort` means the error is unrecoverable and the loop must terminate.
///
/// Exposed publicly so integration tests in `tests/` can drive the recovery
/// logic without going through the full `execute()` API (which requires a
/// live completion model).
pub enum PromptRecovery {
    /// Recovery succeeded; the loop should continue with the given prompt.
    Recovered(Message),
    /// The error is unrecoverable; the loop must abort with this error.
    Abort(ReActError),
}

// ---------------------------------------------------------------------------
// Split impl — utility methods for ReActLoop
// ---------------------------------------------------------------------------

impl<'a, M, P> ReActLoop<'a, M, P>
where
    M: rig_core::completion::CompletionModel
        + rig_core::wasm_compat::WasmCompatSend
        + rig_core::wasm_compat::WasmCompatSync
        + 'static,
    P: rig_core::agent::PromptHook<M>
        + rig_core::wasm_compat::WasmCompatSend
        + rig_core::wasm_compat::WasmCompatSync
        + 'static,
{
    /// Find and clone the content of the last `Message::Assistant` in history.
    pub(super) fn find_last_assistant_content(&self) -> Option<OneOrMany<AssistantContent>> {
        self.history.iter().rev().find_map(|msg| match msg {
            Message::Assistant { content, .. } => Some(content.clone()),
            _ => None,
        })
    }

    /// Emit reasoning thoughts. Each thought is wrapped in a dedicated child
    /// span so reasoning content appears as a distinct traced element.
    pub(super) fn emit_thoughts(
        &self,
        reasoning_texts: &[String],
        cycle: usize,
        trace: &mut ReActTrace,
    ) {
        let thought_text = reasoning_texts.join("\n").trim().to_string();
        if thought_text.is_empty() {
            return;
        }

        let thought = Thought {
            reasoning: thought_text,
            cycle,
        };

        let thought_span = tracing::info_span!(
            "reasoning",
            "langsmith.span.kind" = tracing::field::Empty,
            "openinference.span.kind" = tracing::field::Empty,
            "gen_ai.operation.name" = tracing::field::Empty,
            "gen_ai.content.reasoning" = tracing::field::Empty,
            "react.cycle" = tracing::field::Empty,
        );

        thought_span.in_scope(|| {
            if let Some(cb) = &self.on_thought {
                cb(&thought);
            }
            self.span_emitter.emit_thought(&thought);
        });
        trace.steps.push(ReActStep::Thought(thought));
    }

    /// If `text` is non-empty, record a final answer and return `Ok(trace)`.
    /// Otherwise, record an error and return `Err`.
    ///
    /// Takes `trace` by value because this is always a terminal call site
    /// (`return self.try_finalize_answer(...)`).
    pub(super) fn try_finalize_answer(
        &self,
        text: &str,
        cycle: usize,
        mut trace: ReActTrace,
    ) -> Result<ReActTrace, ReActError> {
        if text.is_empty() {
            // Emit the cycle-end before failing so the failure is visible in
            // OTel/LangSmith traces rather than silently dropped.
            self.span_emitter.emit_cycle_end(cycle, &trace);
            return self.fail_loop(ReActError::NoToolCallsAndNoFinalAnswer { cycle });
        }
        let fa = FinalAnswer {
            text: text.to_string(),
            cycles: cycle + 1,
        };
        trace.steps.push(ReActStep::FinalAnswer(fa.clone()));
        trace.final_answer = Some(fa.clone());
        if let Some(cb) = &self.on_final {
            cb(&fa);
        }
        self.span_emitter.emit_cycle_end(cycle, &trace);
        Ok(trace)
    }

    /// Record a terminal error: invoke the error callback, emit the error
    /// span, and return the error.
    pub(super) fn fail_loop(&self, err: ReActError) -> Result<ReActTrace, ReActError> {
        if let Some(cb) = &self.on_error {
            cb(&err);
        }
        self.span_emitter.emit_error(&err);
        Err(err)
    }

    /// Attempt to recover from a `PromptError`. Returns `Recovered` with the
    /// next prompt if the error is recoverable, or `Abort` if it is not.
    ///
    /// Exposed publicly so integration tests in `tests/` can drive the
    /// recovery logic without going through the full `execute()` API (which
    /// requires a live completion model). The intended call site is
    /// [`ReActLoop::execute`].
    pub async fn recover_from_prompt_error(
        &mut self,
        e: PromptError,
        cycle: usize,
        trace: &mut ReActTrace,
        orig_hist_len: usize,
    ) -> PromptRecovery {
        // --- UnknownToolCall: feed corrective tool result, let model retry ---
        if let PromptError::UnknownToolCall {
            tool_name,
            available_tools,
            chat_history,
            ..
        } = &e
            && let Some(assistant_msg) =
                find_assistant_with_tool_call(chat_history.as_slice(), tool_name)
        {
            return self
                .recover_unknown_tool_call(tool_name, available_tools, &assistant_msg, cycle, trace)
                .await;
        }

        // --- MaxTurnsError: merge rig's internal history, continue loop ---
        if let PromptError::MaxTurnsError {
            chat_history,
            prompt,
            ..
        } = &e
        {
            let new_messages = &chat_history[orig_hist_len..];
            if new_messages.len() > 1 {
                self.history
                    .extend(new_messages[..new_messages.len() - 1].iter().cloned());
            }
            return PromptRecovery::Recovered((**prompt).clone());
        }

        // --- Unrecoverable: abort ---
        PromptRecovery::Abort(ReActError::Model(e.to_string()))
    }

    /// Build corrective tool-result messages for an invalid tool call and
    /// emit trace events. Returns `Recovered` with the next prompt.
    async fn recover_unknown_tool_call(
        &mut self,
        tool_name: &str,
        available_tools: &[String],
        assistant_msg: &Message,
        cycle: usize,
        trace: &mut ReActTrace,
    ) -> PromptRecovery {
        self.history.push(assistant_msg.clone());

        let feedback = format!(
            "Error: tool '{}' is not available. Available tools: [{}]. \
             Please call one of the available tools.",
            tool_name,
            available_tools.join(", ")
        );

        let tool_span = tracing::info_span!(
            "execute_tool",
            "langsmith.span.kind" = tracing::field::Empty,
            "openinference.span.kind" = tracing::field::Empty,
            "gen_ai.tool.name" = tracing::field::Empty,
            "input.value" = tracing::field::Empty,
            "output.value" = tracing::field::Empty,
            "react.is_error" = tracing::field::Empty,
            "react.duration_ms" = tracing::field::Empty,
        );

        let mut recovery_msgs = async {
            let mut msgs = Vec::new();
            let Message::Assistant { content, .. } = assistant_msg else {
                return msgs;
            };
            for item in content.iter() {
                let AssistantContent::ToolCall(tc) = item else {
                    continue;
                };
                let is_invalid = tc.function.name.as_str() == tool_name;
                let result_text = if is_invalid {
                    feedback.clone()
                } else {
                    "Tool call skipped because a peer tool call was \
                     invalid. Please retry this tool call."
                        .to_string()
                };

                let call_id = tc
                    .call_id
                    .clone()
                    .unwrap_or_else(|| format!("react-recovery-{cycle}"));
                let tool_result_content = ToolResultContent::text(result_text.clone());
                let user_content = UserContent::tool_result_with_call_id(
                    tc.id.clone(),
                    call_id,
                    OneOrMany::one(tool_result_content),
                );
                msgs.push(Message::User {
                    content: OneOrMany::one(user_content),
                });

                // Emit trace for the invalid tool call only.
                if is_invalid {
                    let action = Action {
                        tool_name: tc.function.name.clone(),
                        args: tc.function.arguments.to_string(),
                        tool_call_id: Some(tc.id.clone()),
                        cycle,
                    };
                    if let Some(cb) = &self.on_action {
                        cb(&action);
                    }
                    self.span_emitter.emit_action(&action);
                    trace.steps.push(ReActStep::Action(action));

                    let observation = Observation {
                        tool_name: tc.function.name.clone(),
                        result: result_text,
                        is_error: true,
                        cycle,
                        duration: Duration::default(),
                    };
                    if let Some(cb) = &self.on_observation {
                        cb(&observation);
                    }
                    self.span_emitter.emit_observation(&observation);
                    trace.steps.push(ReActStep::Observation(observation));
                }
            }
            msgs
        }
        .instrument(tool_span)
        .await;

        // Push all but the last recovery message to history; the last one
        // becomes the prompt for the next cycle (mirrors normal tool path).
        if let Some(last) = recovery_msgs.pop() {
            self.history.extend(recovery_msgs);
            PromptRecovery::Recovered(last)
        } else {
            // No tool calls found in the assistant message — abort.
            PromptRecovery::Abort(ReActError::Model(format!(
                "UnknownToolCall recovery failed: no tool calls in assistant message for '{tool_name}'"
            )))
        }
    }

    /// Execute all tool calls in a cycle, emitting actions/observations and
    /// building the next cycle's prompt. The last tool result becomes the
    /// next prompt; preceding ones are pushed to history.
    pub(super) async fn execute_tool_calls(
        &mut self,
        tool_calls: &[&ToolCall],
        cycle: usize,
        trace: &mut ReActTrace,
    ) -> Message {
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

            let tool_span = tracing::info_span!(
                "execute_tool",
                "langsmith.span.kind" = tracing::field::Empty,
                "openinference.span.kind" = tracing::field::Empty,
                "gen_ai.tool.name" = tracing::field::Empty,
                "input.value" = tracing::field::Empty,
                "output.value" = tracing::field::Empty,
                "react.is_error" = tracing::field::Empty,
                "react.duration_ms" = tracing::field::Empty,
            );

            let observation = async {
                let internal_call_id = format!("react-cycle-{cycle}");
                if let Some(ref hook) = self.agent.hook {
                    let _ = hook
                        .on_tool_call(
                            &tc.function.name,
                            Some(tc.id.clone()),
                            &internal_call_id,
                            &args_str,
                        )
                        .await;
                }

                let start = Instant::now();
                let result = self
                    .agent
                    .tool_server_handle
                    .call_tool(&tc.function.name, &args_str)
                    .await;
                let duration = start.elapsed();

                let obs = match result {
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
                    cb(&obs);
                }
                self.span_emitter.emit_observation(&obs);

                if let Some(ref hook) = self.agent.hook {
                    let _ = hook
                        .on_tool_result(
                            &tc.function.name,
                            Some(tc.id.clone()),
                            &internal_call_id,
                            &args_str,
                            &obs.result,
                        )
                        .await;
                }

                obs
            }
            .instrument(tool_span)
            .await;

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
                OneOrMany::one(tool_result_content),
            );
            let msg = Message::User {
                content: OneOrMany::one(user_content),
            };

            // The last tool result is held as the prompt for the next cycle.
            // Any other preceding tool results in this cycle are pushed to
            // self.history.
            if i == num_tool_calls - 1 {
                next_prompt = Some(msg);
            } else {
                self.history.push(msg);
            }
        }

        // next_prompt is always set because the loop runs at least once
        // (tool_calls is non-empty when execute_tool_calls is called).
        match next_prompt {
            Some(msg) => msg,
            None => {
                let err = ReActError::Model(
                    "execute_tool_calls called with empty tool_calls".to_string(),
                );
                tracing::error!(error = %err, "execute_tool_calls invariant violated");
                Message::User {
                    content: OneOrMany::one(UserContent::text(err.to_string())),
                }
            }
        }
    }
}
