#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use agent_rs_lib::agent::ManagedExt;
use agent_rs_lib::agent::memory::context::ContextManager;
use agent_rs_lib::agent::memory::tokenizer::{count_messages_tokens, count_string_tokens};
use rig_core::OneOrMany;
use rig_core::client::CompletionClient;
use rig_core::completion::{Prompt, PromptError};
use rig_core::message::{AssistantContent, Message, UserContent};
use rig_core::wasm_compat::WasmCompatSend;

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

#[derive(Clone)]
struct MockCompactor {
    response: String,
}

impl Prompt for MockCompactor {
    #[allow(refining_impl_trait)]
    async fn prompt(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> Result<String, PromptError> {
        if self.response != "Fallback" {
            return Ok(self.response.clone());
        }
        let msg = prompt.into();
        match msg {
            Message::System { content } => Ok(format!("Mock response for: {}", content)),
            Message::User { content } => {
                let text = match content.first_ref() {
                    UserContent::Text(t) => t.text.clone(),
                    _ => "".to_string(),
                };
                Ok(format!("Mock response for: {}", text))
            }
            _ => Ok(self.response.clone()),
        }
    }
}

#[test]
fn test_count_string_tokens() {
    let text = "Hello, world!";
    let count = count_string_tokens(text);
    // "Hello, world!" is 4 tokens under cl100k_base
    assert_eq!(count, 4);
}

#[test]
fn test_count_messages_tokens_basic() {
    let messages = vec![
        Message::System {
            content: "You are a helpful assistant.".to_string(),
        },
        Message::User {
            content: rig_core::OneOrMany::one(UserContent::text("Hello")),
        },
        Message::Assistant {
            id: None,
            content: rig_core::OneOrMany::one(AssistantContent::text("Hi there!")),
        },
    ];

    let count = count_messages_tokens(&messages);
    // Expected token calculation:
    // system: "You are a helpful assistant." (6 tokens) + overhead (4 tokens) = 10 tokens
    // user: "Hello" (1 token) + overhead (4 tokens) = 5 tokens
    // assistant: "Hi there!" (3 tokens) + overhead (4 tokens) = 7 tokens
    // Total: 22 tokens
    assert_eq!(count, 22);
}

#[test]
fn test_count_messages_tokens_complex_fallback() {
    // We will test serialization fallback for complex content
    // Create a UserContent::Image
    let img = rig_core::completion::message::Image {
        data: rig_core::completion::message::DocumentSourceKind::String("dGVzdA==".to_string()),
        media_type: None,
        detail: None,
        additional_params: None,
    };
    let messages = vec![Message::User {
        content: rig_core::OneOrMany::one(UserContent::Image(img)),
    }];

    let count = count_messages_tokens(&messages);
    // Since it serializes the image variant to string, it should be > 4 overhead tokens.
    assert!(count > 4);
}

#[tokio::test]
async fn test_context_manager_no_compaction_under_threshold() {
    let compactor = MockCompactor {
        response: "Compacted".to_string(),
    };
    // Threshold set high (1000 tokens)
    let manager = ContextManager::new(1000, compactor);

    let mut history = vec![Message::User {
        content: rig_core::OneOrMany::one(UserContent::text("Short message")),
    }];

    let compacted = manager
        .compact_history_if_needed(&mut history, "Short prompt")
        .await
        .unwrap();

    assert!(!compacted);
    assert_eq!(history.len(), 1);
    if let Message::User { content } = &history[0] {
        if let UserContent::Text(t) = content.first_ref() {
            assert_eq!(t.text, "Short message");
        } else {
            panic!("Expected text user content");
        }
    } else {
        panic!("Expected user message");
    }
}

#[tokio::test]
async fn test_context_manager_compaction_above_threshold() {
    let compactor = MockCompactor {
        response: "Compacted summary history".to_string(),
    };
    // Threshold set low (15 tokens)
    let manager = ContextManager::new(15, compactor);

    let mut history = vec![Message::User {
        content: rig_core::OneOrMany::one(UserContent::text(
            "A very long message that is guaranteed to exceed the 15 token threshold",
        )),
    }];

    let compacted = manager
        .compact_history_if_needed(&mut history, "Another prompt")
        .await
        .unwrap();

    assert!(compacted);
    assert_eq!(history.len(), 1);
    if let Message::System { content } = &history[0] {
        assert_eq!(content, "Compacted summary history");
    } else {
        panic!("Expected system message after compaction");
    }
}

#[tokio::test]
async fn test_context_manager_custom_compaction_prompt() {
    let compactor = MockCompactor {
        response: "Fallback".to_string(),
    };
    // Threshold low (15 tokens)
    let manager =
        ContextManager::new(15, compactor).with_compaction_prompt_formatter(|history_text| {
            format!("CUSTOM HEADER: {}", history_text)
        });

    let mut history = vec![Message::User {
        content: rig_core::OneOrMany::one(UserContent::text(
            "A very long message that is guaranteed to exceed the 15 token threshold",
        )),
    }];

    let compacted = manager
        .compact_history_if_needed(&mut history, "Another prompt")
        .await
        .unwrap();

    assert!(compacted);
    assert_eq!(history.len(), 1);
    if let Message::System { content } = &history[0] {
        assert!(content.starts_with("Mock response for: CUSTOM HEADER:"));
    } else {
        panic!("Expected system message after compaction");
    }
}

#[test]
fn test_managed_builder_defaults() {
    let agent = make_test_agent();
    let built = agent.managed().build();
    assert!(built.history().is_empty());
}

#[test]
fn test_managed_builder_with_history_seeds() {
    let agent = make_test_agent();
    let msg = Message::user("hello");
    let built = agent.managed().with_history(vec![msg.clone()]).build();
    assert_eq!(built.history().len(), 1);
}

#[tokio::test]
async fn test_managed_prompt_does_not_mutate_history() {
    let agent = make_test_agent();
    let built = agent.managed().build();
    let before = built.history().len();
    // .prompt() will fail (no real LLM), but history should remain unchanged.
    let _ = built.prompt("test").await;
    assert_eq!(built.history().len(), before);
}

#[test]
#[should_panic(expected = "threshold")]
fn test_managed_builder_compaction_panics_without_threshold() {
    let agent = make_test_agent();
    let _ = agent.managed().with_compaction().build(); // panics
}

#[tokio::test]
async fn test_managed_stream_appends_history() {
    use agent_rs_lib::agent::ManagedStream;
    use futures::StreamExt;
    use rig_core::agent::MultiTurnStreamItem;
    use rig_core::completion::Usage;
    use std::sync::{Arc, Mutex};

    let final_history = vec![
        Message::User {
            content: rig_core::OneOrMany::one(UserContent::text("Hello")),
        },
        Message::Assistant {
            id: None,
            content: rig_core::OneOrMany::one(AssistantContent::text("Hi there!")),
        },
    ];

    let final_item: MultiTurnStreamItem<()> = MultiTurnStreamItem::final_response_with_history(
        OneOrMany::one(AssistantContent::text("Hi there!")),
        Usage::new(),
        Some(final_history.clone()),
    );

    let inner_stream = futures::stream::iter(vec![Ok(final_item)]);
    let history = Arc::new(Mutex::new(Vec::<Message>::new()));
    let mut managed_stream = ManagedStream::new(inner_stream, Some(Arc::clone(&history)));

    while let Some(item) = managed_stream.next().await {
        assert!(item.is_ok());
    }

    let updated_history = history.lock().unwrap();
    assert_eq!(updated_history.len(), 2);
    if let Message::Assistant { content, .. } = &updated_history[1] {
        if let AssistantContent::Text(t) = content.first_ref() {
            assert_eq!(t.text, "Hi there!");
        } else {
            panic!("Expected text assistant content");
        }
    } else {
        panic!("Expected assistant message");
    }
}

#[tokio::test]
async fn test_managed_stream_no_history_no_append() {
    use agent_rs_lib::agent::ManagedStream;
    use futures::StreamExt;
    use rig_core::agent::MultiTurnStreamItem;
    use rig_core::completion::Usage;
    use std::sync::{Arc, Mutex};

    let final_item: MultiTurnStreamItem<()> = MultiTurnStreamItem::final_response(
        OneOrMany::one(AssistantContent::text("Hi there!")),
        Usage::new(),
    );

    let inner_stream = futures::stream::iter(vec![Ok(final_item)]);
    let history = Arc::new(Mutex::new(Vec::<Message>::new()));
    let mut managed_stream = ManagedStream::new(inner_stream, Some(Arc::clone(&history)));

    while let Some(item) = managed_stream.next().await {
        assert!(item.is_ok());
    }

    // FinalResponse without history() => no append
    let updated_history = history.lock().unwrap();
    assert!(updated_history.is_empty());
}

#[test]
fn test_strip_reasoning_mixed_content_keeps_text_and_tool_call() {
    use rig_core::message::Reasoning;
    use rig_core::message::{ToolCall, ToolFunction};

    let history = vec![
        Message::User {
            content: rig_core::OneOrMany::one(UserContent::text("hello")),
        },
        Message::Assistant {
            id: None,
            content: rig_core::OneOrMany::many(vec![
                AssistantContent::Reasoning(Reasoning::new("thinking...")),
                AssistantContent::text("Hi there!"),
                AssistantContent::ToolCall(ToolCall::new(
                    "call_1".to_string(),
                    ToolFunction::new("search".to_string(), serde_json::json!({})),
                )),
            ])
            .unwrap(),
        },
    ];

    let result = agent_rs_lib::agent::strip_reasoning_from_history(history);
    assert_eq!(result.len(), 2);

    // User message unchanged
    assert!(matches!(result[0], Message::User { .. }));

    // Assistant message: reasoning stripped, text and tool call preserved
    if let Message::Assistant { content, .. } = &result[1] {
        assert_eq!(content.iter().count(), 2);
        assert!(matches!(content.first_ref(), AssistantContent::Text(_)));
        let second = content.iter().nth(1).unwrap();
        assert!(matches!(second, AssistantContent::ToolCall(_)));
    } else {
        panic!("Expected assistant message");
    }
}

#[test]
fn test_strip_reasoning_drops_pure_reasoning_message() {
    use rig_core::message::Reasoning;

    let history = vec![
        Message::User {
            content: rig_core::OneOrMany::one(UserContent::text("hello")),
        },
        Message::Assistant {
            id: None,
            content: rig_core::OneOrMany::one(AssistantContent::Reasoning(Reasoning::new(
                "just thinking",
            ))),
        },
    ];

    let result = agent_rs_lib::agent::strip_reasoning_from_history(history);
    assert_eq!(result.len(), 1);
    assert!(matches!(result[0], Message::User { .. }));
}

#[test]
fn test_strip_reasoning_passes_through_non_assistant_messages() {
    let history = vec![
        Message::System {
            content: "You are helpful.".to_string(),
        },
        Message::User {
            content: rig_core::OneOrMany::one(UserContent::text("hello")),
        },
    ];

    let result = agent_rs_lib::agent::strip_reasoning_from_history(history.clone());
    assert_eq!(result.len(), 2);
    assert_eq!(result, history);
}

#[test]
fn test_strip_reasoning_empty_input() {
    let result = agent_rs_lib::agent::strip_reasoning_from_history(vec![]);
    assert!(result.is_empty());
}

#[test]
fn test_strip_reasoning_multiple_assistant_messages_independently_filtered() {
    use rig_core::message::Reasoning;

    let history = vec![
        Message::User {
            content: rig_core::OneOrMany::one(UserContent::text("first")),
        },
        Message::Assistant {
            id: None,
            content: rig_core::OneOrMany::many(vec![
                AssistantContent::Reasoning(Reasoning::new("thinking")),
                AssistantContent::text("response 1"),
            ])
            .unwrap(),
        },
        Message::User {
            content: rig_core::OneOrMany::one(UserContent::text("second")),
        },
        Message::Assistant {
            id: None,
            content: rig_core::OneOrMany::one(AssistantContent::Reasoning(Reasoning::new(
                "more thinking",
            ))),
        },
    ];

    let result = agent_rs_lib::agent::strip_reasoning_from_history(history);
    // First assistant: reasoning stripped, text kept → still present
    // Second assistant: pure reasoning → dropped
    assert_eq!(result.len(), 3);
    assert!(matches!(result[0], Message::User { .. }));
    assert!(matches!(result[1], Message::Assistant { .. }));
    assert!(matches!(result[2], Message::User { .. }));
}

#[test]
fn test_strip_reasoning_preserves_assistant_id() {
    use rig_core::message::Reasoning;

    let history = vec![Message::Assistant {
        id: Some("msg_123".to_string()),
        content: rig_core::OneOrMany::many(vec![
            AssistantContent::Reasoning(Reasoning::new("thinking")),
            AssistantContent::text("result"),
        ])
        .unwrap(),
    }];

    let result = agent_rs_lib::agent::strip_reasoning_from_history(history);
    assert_eq!(result.len(), 1);
    if let Message::Assistant { id, content, .. } = &result[0] {
        assert_eq!(id.as_deref(), Some("msg_123"));
        assert_eq!(content.iter().count(), 1);
    } else {
        panic!("Expected assistant message");
    }
}
