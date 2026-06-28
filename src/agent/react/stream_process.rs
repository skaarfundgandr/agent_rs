use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::domain::agent::ReActStreamItem;
use crate::domain::agent::{Action, Observation, ReActStep, ReActTrace, Thought};

use super::callbacks::{ActionCb, ObservationCb, ThoughtCb};
use super::emitter::ReActSpanEmitter;
use super::streaming::send_or_break;

/// Process a `StreamAssistantItem` (text, tool call, or reasoning).
///
/// Returns `true` if the inner stream loop should break.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn process_assistant_item<R>(
    assistant_item: rig_core::streaming::StreamedAssistantContent<R>,
    tx: &tokio::sync::mpsc::Sender<ReActStreamItem>,
    trace: &mut ReActTrace,
    has_tool_calls: &mut bool,
    final_answer_buffer: &mut String,
    pending_tool_calls: &mut HashMap<String, (String, Instant)>,
    current_cycle: usize,
    on_thought_cb: &Option<ThoughtCb>,
    on_action_cb: &Option<ActionCb>,
    span_emitter: &Arc<dyn ReActSpanEmitter>,
) -> bool {
    use rig_core::streaming::StreamedAssistantContent;

    match assistant_item {
        StreamedAssistantContent::Text(text) => {
            if *has_tool_calls {
                final_answer_buffer.push_str(&text.text);
                if send_or_break(
                    tx,
                    ReActStreamItem::FinalAnswerDelta {
                        delta: text.text,
                        cycle: current_cycle,
                    },
                )
                .await
                {
                    return true;
                }
            } else if send_or_break(
                tx,
                ReActStreamItem::ThoughtDelta {
                    delta: text.text,
                    cycle: current_cycle,
                },
            )
            .await
            {
                return true;
            }
        }
        StreamedAssistantContent::ToolCall { tool_call, .. } => {
            let action = Action {
                tool_name: tool_call.function.name.clone(),
                args: tool_call.function.arguments.to_string(),
                tool_call_id: Some(tool_call.id.clone()),
                cycle: current_cycle,
            };
            pending_tool_calls.insert(
                tool_call.id.clone(),
                (tool_call.function.name.clone(), Instant::now()),
            );
            if let Some(cb) = on_action_cb {
                cb(&action);
            }
            span_emitter.emit_action(&action);
            trace.steps.push(ReActStep::Action(action.clone()));
            *has_tool_calls = true;
            if send_or_break(
                tx,
                ReActStreamItem::Action {
                    tool_name: action.tool_name,
                    args: action.args,
                    tool_call_id: action.tool_call_id,
                    cycle: action.cycle,
                },
            )
            .await
            {
                return true;
            }
        }
        StreamedAssistantContent::Reasoning(reasoning) => {
            let text = reasoning.display_text();
            if !text.is_empty() {
                let thought = Thought {
                    reasoning: text.clone(),
                    cycle: current_cycle,
                };
                if let Some(cb) = on_thought_cb {
                    cb(&thought);
                }
                span_emitter.emit_thought(&thought);
                trace.steps.push(ReActStep::Thought(thought));
                if send_or_break(
                    tx,
                    ReActStreamItem::ThoughtDelta {
                        delta: text,
                        cycle: current_cycle,
                    },
                )
                .await
                {
                    return true;
                }
            }
        }
        StreamedAssistantContent::ReasoningDelta { reasoning, .. } if !reasoning.is_empty() => {
            if send_or_break(
                tx,
                ReActStreamItem::ThoughtDelta {
                    delta: reasoning,
                    cycle: current_cycle,
                },
            )
            .await
            {
                return true;
            }
        }
        StreamedAssistantContent::ReasoningDelta { .. } => {}
        _ => {}
    }
    false
}

/// Process a `StreamUserItem::ToolResult`.
///
/// Returns `true` if the inner stream loop should break.
pub(crate) async fn process_tool_result(
    tool_result: rig_core::message::ToolResult,
    tx: &tokio::sync::mpsc::Sender<ReActStreamItem>,
    trace: &mut ReActTrace,
    pending_tool_calls: &mut HashMap<String, (String, Instant)>,
    current_cycle: usize,
    on_observation_cb: &Option<ObservationCb>,
    span_emitter: &Arc<dyn ReActSpanEmitter>,
) -> bool {
    let result_text = tool_result
        .content
        .iter()
        .filter_map(|c| match c {
            rig_core::message::ToolResultContent::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect::<String>();
    let (tool_name, start) = pending_tool_calls
        .remove(&tool_result.id)
        .unwrap_or_else(|| ("unknown".to_string(), Instant::now()));
    let observation = Observation {
        tool_name,
        result: result_text,
        is_error: false,
        cycle: current_cycle,
        duration: start.elapsed(),
    };
    if let Some(cb) = on_observation_cb {
        cb(&observation);
    }
    span_emitter.emit_observation(&observation);
    trace
        .steps
        .push(ReActStep::Observation(observation.clone()));
    send_or_break(
        tx,
        ReActStreamItem::Observation {
            tool_name: observation.tool_name,
            result: observation.result,
            is_error: observation.is_error,
            cycle: observation.cycle,
            duration: observation.duration,
        },
    )
    .await
}
