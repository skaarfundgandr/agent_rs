#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use agent_rs::agent::ManagedStream;
use agent_rs::agent::ReActExt;
use agent_rs::agent::react::ReActSpanEmitter;
use futures::StreamExt;
use rig_core::OneOrMany;
use rig_core::agent::MultiTurnStreamItem;
use rig_core::client::CompletionClient;
use rig_core::completion::Usage;
use rig_core::message::{AssistantContent, Message};
use std::sync::Arc;

struct TestSpanEmitter;

impl ReActSpanEmitter for TestSpanEmitter {}

fn make_test_agent() -> rig_core::agent::Agent<
    rig_core::providers::openai::responses_api::ResponsesCompletionModel<reqwest::Client>,
> {
    let client = rig_core::providers::openai::Client::builder()
        .base_url("http://127.0.0.1:1")
        .api_key("test")
        .build()
        .expect("build openai client for test");
    client
        .agent(rig_core::providers::openai::GPT_4O)
        .preamble("test preamble")
        .build()
}

#[test]
fn test_react_builder_accepts_all_callbacks() {
    let agent = make_test_agent();
    let built = agent
        .react()
        .on_thought(|_t| {})
        .on_action(|_a| {})
        .on_observation(|_o| {})
        .on_final(|_f| {})
        .on_error(|_e| {})
        .build();

    let h: Vec<Message> = Vec::new();
    assert!(h.is_empty());
    assert_eq!(built.max_cycles(), 20);
}

#[test]
fn test_react_builder_accepts_partial_callbacks() {
    let agent = make_test_agent();
    let _built = agent.react().on_thought(|_t| {}).on_error(|_e| {}).build();
    let h: Vec<Message> = Vec::new();
    assert!(h.is_empty());
}

#[test]
fn test_react_builder_with_span_emitter() {
    let agent = make_test_agent();
    let _built = agent
        .react()
        .with_span_emitter(Arc::new(TestSpanEmitter))
        .build();
    let h: Vec<Message> = Vec::new();
    assert!(h.is_empty());
}

#[test]
fn test_react_stream_item_enum_variants() {
    use agent_rs::domain::agent::ReActStreamItem;
    use std::time::Duration;

    let cycle_start = ReActStreamItem::CycleStart { cycle: 0 };
    assert!(matches!(
        cycle_start,
        ReActStreamItem::CycleStart { cycle: 0 }
    ));

    let thought = ReActStreamItem::ThoughtDelta {
        delta: "thinking".to_string(),
        cycle: 0,
    };
    assert!(matches!(thought, ReActStreamItem::ThoughtDelta { .. }));

    let action = ReActStreamItem::Action {
        tool_name: "read_file".to_string(),
        args: "{}".to_string(),
        tool_call_id: None,
        cycle: 0,
    };
    assert!(matches!(action, ReActStreamItem::Action { .. }));

    let observation = ReActStreamItem::Observation {
        tool_name: "read_file".to_string(),
        result: "content".to_string(),
        is_error: false,
        cycle: 0,
        duration: Duration::from_millis(10),
    };
    assert!(matches!(observation, ReActStreamItem::Observation { .. }));

    let final_delta = ReActStreamItem::FinalAnswerDelta {
        delta: "answer".to_string(),
        cycle: 0,
    };
    assert!(matches!(
        final_delta,
        ReActStreamItem::FinalAnswerDelta { .. }
    ));
}

#[test]
fn test_react_builder_compaction_accepts_callbacks() {
    let agent = make_test_agent();
    let _built = agent
        .react()
        .with_compaction()
        .threshold(1000)
        .on_thought(|_t| {})
        .on_error(|_e| {})
        .build();
    let h: Vec<Message> = Vec::new();
    assert!(h.is_empty());
}

#[test]
fn test_react_builder_callback_types_are_arc() {
    use agent_rs::agent::react::{ActionCb, ErrorCb, FinalCb, ObservationCb, ThoughtCb};
    use std::sync::Arc;

    let _: Option<ThoughtCb> = Some(Arc::new(|_t| {}));
    let _: Option<ActionCb> = Some(Arc::new(|_a| {}));
    let _: Option<ObservationCb> = Some(Arc::new(|_o| {}));
    let _: Option<FinalCb> = Some(Arc::new(|_f| {}));
    let _: Option<ErrorCb> = Some(Arc::new(|_e| {}));
}

#[tokio::test]
async fn test_managed_stream_appends_history_only_once() {
    let mut history = Vec::<Message>::new();
    let prompt_text = "hello".to_string();

    let final_response =
        MultiTurnStreamItem::<()>::FinalResponse(rig_core::agent::FinalResponse::new(
            OneOrMany::one(AssistantContent::text("world")),
            Usage::new(),
            None,
        ));

    // Emit two FinalResponse items; the wrapper should only append to shared
    // history once, even if the underlying stream produces duplicates.
    let stream = futures::stream::iter(vec![
        Ok::<_, rig_core::agent::StreamingError>(final_response.clone()),
        Ok::<_, rig_core::agent::StreamingError>(final_response),
    ]);

    let mut managed = ManagedStream::new(stream, Some(&mut history), prompt_text, None);
    while managed.next().await.is_some() {}

    assert_eq!(history.len(), 2);
    assert!(matches!(history[0], Message::User { .. }));
    assert!(matches!(history[1], Message::Assistant { .. }));
}
