use rig_core::message::{AssistantContent, Message, ToolCall};

use crate::domain::agent::FinalAnswer;

/// Parsed contents of an assistant message, classified by type.
pub(crate) struct ParsedAssistantContent<'a> {
    /// Reasoning texts (from `Reasoning` items and `Text` items before the
    /// first tool call).
    pub reasoning_texts: Vec<String>,
    /// Tool calls in order.
    pub tool_calls: Vec<&'a ToolCall>,
    /// Text items that appear after the last tool call.
    pub trailing_texts: Vec<String>,
}

/// Classify assistant content items into reasoning, tool calls, and trailing
/// text.
pub(crate) fn classify_assistant_content(
    content: &[AssistantContent],
) -> ParsedAssistantContent<'_> {
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

    ParsedAssistantContent {
        reasoning_texts,
        tool_calls,
        trailing_texts,
    }
}

/// Find the last assistant message's content in a message list.
/// Returns owned content to avoid borrow conflicts with `working_history`.
pub(crate) fn find_assistant_content(history: &[Message]) -> Option<Vec<AssistantContent>> {
    history.iter().rev().find_map(|msg| match msg {
        Message::Assistant { content, .. } => Some(content.iter().cloned().collect()),
        _ => None,
    })
}

/// Try to detect a final answer from trailing text after tool calls.
/// Returns `Some(text)` if a final answer marker is found, `None` otherwise.
pub(crate) fn try_detect_final_answer(trailing_texts: &[String]) -> Option<String> {
    let full_trailing = trailing_texts.join("").trim().to_string();
    super::helpers::detect_final_answer(&full_trailing)
}

/// Build a [`FinalAnswer`] from the model's output text and emit the
/// appropriate callbacks/tracing. Returns the final answer.
pub(crate) fn emit_final_answer_from_output(
    output: String,
    cycle: usize,
    trace: &mut crate::domain::agent::ReActTrace,
    on_final: &Option<super::callbacks::FinalCb>,
    span_emitter: &std::sync::Arc<dyn super::emitter::ReActSpanEmitter>,
) -> FinalAnswer {
    let fa = FinalAnswer {
        text: output,
        cycles: cycle + 1,
    };
    trace
        .steps
        .push(crate::domain::agent::ReActStep::FinalAnswer(fa.clone()));
    trace.final_answer = Some(fa.clone());
    if let Some(cb) = on_final {
        cb(&fa);
    }
    span_emitter.emit_cycle_end(cycle, trace);
    fa
}

/// Append prompt + assistant response to shared history when
/// `append_to_shared_history` is true.
pub(crate) fn append_to_history_if_needed(
    append: bool,
    shared_history: &std::sync::Arc<crate::agent::utils::Mutex<Vec<Message>>>,
    prompt: &str,
    response_text: &str,
) {
    if append {
        let mut h = crate::agent::utils::lock_mutex(shared_history);
        h.push(Message::User {
            content: rig_core::OneOrMany::one(rig_core::message::UserContent::text(prompt)),
        });
        h.push(Message::assistant(response_text));
    }
}
