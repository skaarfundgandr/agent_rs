//! Shared test utilities for integration tests.
#![allow(dead_code, clippy::duplicate_mod, clippy::manual_async_fn)]

use rig_core::OneOrMany;
use rig_core::agent::{AgentBuilder, StreamingPromptRequest};
use rig_core::completion::{
    CompletionError, CompletionModel, CompletionRequest, CompletionResponse, Usage,
};
use rig_core::message::{AssistantContent, Message, ToolCall, ToolFunction, UserContent};
use rig_core::streaming::{RawStreamingChoice, StreamingChat, StreamingCompletionResponse};
use rig_core::tool::Tool;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use agent_rs::agent::ReActExt;
use agent_rs::agent::react::BuiltReAct;

/// A deterministic mock [`CompletionModel`] that returns canned responses from a queue.
///
/// Used across `react_e2e`, `react_recovery_tests`, and `agents_tests` to test
/// the ReAct loop and managed agent without a real LLM backend.
#[derive(Clone, Debug)]
pub struct MockCompletionModel {
    responses: Arc<Mutex<VecDeque<MockResponse>>>,
    /// Optional canned text for streaming responses.
    streaming_text: Option<String>,
}

/// What the mock returns for a single `completion()` call.
#[derive(Clone, Debug)]
pub enum MockResponse {
    /// Return a successful `CompletionResponse` with the given assistant content.
    Ok(OneOrMany<AssistantContent>),
    /// Return a non-transient error (not retried by the ReAct loop).
    Err(String),
}

impl MockCompletionModel {
    /// Create a new mock with the given response queue.
    pub fn new(responses: Vec<MockResponse>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(VecDeque::from(responses))),
            streaming_text: None,
        }
    }

    /// Create a mock that supports streaming with the given canned text.
    pub fn with_streaming_text(text: &str) -> Self {
        Self {
            responses: Arc::new(Mutex::new(VecDeque::new())),
            streaming_text: Some(text.to_string()),
        }
    }

    /// Helper: build a text-only successful response.
    pub fn text(text: &str) -> MockResponse {
        MockResponse::Ok(OneOrMany::one(AssistantContent::text(text)))
    }

    /// Helper: build a tool-call successful response.
    pub fn tool_call(call_id: &str, name: &str, args: serde_json::Value) -> MockResponse {
        MockResponse::Ok(OneOrMany::one(AssistantContent::ToolCall(ToolCall::new(
            call_id.to_string(),
            ToolFunction::new(name.to_string(), args),
        ))))
    }

    /// Helper: build a non-transient error response.
    pub fn json_error(msg: &str) -> MockResponse {
        MockResponse::Err(msg.to_string())
    }
}

impl CompletionModel for MockCompletionModel {
    type Response = ();
    type StreamingResponse = ();
    type Client = ();

    fn make(_client: &Self::Client, _model: impl Into<String>) -> Self {
        Self::new(vec![])
    }

    fn completion(
        &self,
        _request: CompletionRequest,
    ) -> impl std::future::Future<
        Output = Result<CompletionResponse<Self::Response>, CompletionError>,
    > + Send {
        let responses = self.responses.clone();
        async move {
            let mut q = responses.lock().expect("mock lock poisoned");
            match q.pop_front() {
                Some(MockResponse::Ok(choice)) => Ok(CompletionResponse {
                    choice,
                    usage: Usage::new(),
                    raw_response: (),
                    message_id: None,
                }),
                Some(MockResponse::Err(msg)) => {
                    let json_err: serde_json::Error =
                        serde_json::from_str::<serde_json::Value>("invalid").unwrap_err();
                    let _ = msg;
                    Err(CompletionError::JsonError(json_err))
                }
                None => Err(CompletionError::ProviderError(
                    "mock: no more responses configured".into(),
                )),
            }
        }
    }

    fn stream(
        &self,
        _request: CompletionRequest,
    ) -> impl std::future::Future<
        Output = Result<
            rig_core::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
            CompletionError,
        >,
    > + Send {
        let text = self.streaming_text.clone();
        async move {
            match text {
                Some(t) => Ok(mock_streaming_response(&t)),
                None => Err(CompletionError::ProviderError(
                    "mock: streaming not configured (use with_streaming_text)".into(),
                )),
            }
        }
    }
}

/// Build a mock streaming response that yields the given text as a single chunk.
pub fn mock_streaming_response(text: &str) -> StreamingCompletionResponse<()> {
    let chunks: Vec<Result<RawStreamingChoice<()>, CompletionError>> =
        vec![Ok(RawStreamingChoice::Message(text.to_string()))];
    let stream = futures::stream::iter(chunks);
    StreamingCompletionResponse::stream(Box::pin(stream))
}

impl StreamingChat<MockCompletionModel, ()> for MockCompletionModel {
    fn stream_chat<I, T>(
        &self,
        prompt: impl Into<rig_core::message::Message> + rig_core::wasm_compat::WasmCompatSend,
        chat_history: I,
    ) -> StreamingPromptRequest<MockCompletionModel>
    where
        I: IntoIterator<Item = T> + rig_core::wasm_compat::WasmCompatSend,
        T: Into<rig_core::message::Message>,
    {
        let agent = AgentBuilder::new(self.clone()).default_max_turns(1).build();
        StreamingPromptRequest::<MockCompletionModel>::new(Arc::new(agent), prompt)
            .history(chat_history)
    }
}

// ---------------------------------------------------------------------------
// Simple echo tool for ReAct loop tests
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct EchoTool;

#[derive(serde::Deserialize, serde::Serialize)]
pub struct EchoArgs {
    pub text: String,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct EchoOutput(pub String);

impl std::fmt::Display for EchoOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Tool for EchoTool {
    const NAME: &'static str = "echo";
    type Args = EchoArgs;
    type Output = EchoOutput;
    type Error = std::io::Error;

    fn description(&self) -> String {
        "Echo back the input text".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to echo back"
                }
            },
            "required": ["text"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(EchoOutput(args.text))
    }
}

/// Build an `Agent<MockCompletionModel>` with the given response queue
/// and an `echo` tool registered. Sets `default_max_turns(1)` so rig-core's
/// internal multi-turn loop runs exactly once per `agent.prompt()` call,
/// preventing it from consuming multiple mock responses.
pub fn mock_agent(responses: Vec<MockResponse>) -> rig_core::agent::Agent<MockCompletionModel> {
    let model = MockCompletionModel::new(responses);
    AgentBuilder::new(model)
        .tool(EchoTool)
        .default_max_turns(1)
        .build()
}

pub fn react_with_responses(
    responses: Vec<MockResponse>,
) -> BuiltReAct<MockCompletionModel> {
    let agent = mock_agent(responses);
    agent.react().build()
}

pub fn react_with_responses_compact(
    responses: Vec<MockResponse>,
    threshold: usize,
) -> BuiltReAct<MockCompletionModel, rig_core::agent::Agent<MockCompletionModel>> {
    let agent = mock_agent(responses);
    agent.react().with_compaction().threshold(threshold).build()
}

pub fn mock_history(msgs: &[&str]) -> Vec<Message> {
    msgs.iter()
        .map(|&text| Message::User {
            content: OneOrMany::one(UserContent::text(text)),
        })
        .collect()
}

#[cfg(feature = "rag")]
pub async fn rag_pipeline(tmp: &tempfile::TempDir) -> agent_rs::rag::BuiltRag {
    agent_rs::rag::RagPipeline::builder()
        .embedder(agent_rs::agent::embeddings::EmbeddingService::new(
            RagMockEmbeddingModel,
        ))
        .store_at(tmp.path())
        .build()
        .await
        .unwrap()
}

#[cfg(feature = "rag")]
#[derive(Clone)]
struct RagMockEmbeddingModel;

#[cfg(feature = "rag")]
impl rig_core::embeddings::EmbeddingModel for RagMockEmbeddingModel {
    const MAX_DOCUMENTS: usize = 8;
    type Client = ();

    fn make(_: &Self::Client, _: impl Into<String>, _: Option<usize>) -> Self {
        Self
    }

    fn ndims(&self) -> usize {
        8
    }

    async fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> std::result::Result<
        Vec<rig_core::embeddings::Embedding>,
        rig_core::embeddings::EmbeddingError,
    > {
        Ok(texts
            .into_iter()
            .map(|text| rig_core::embeddings::Embedding {
                document: text.clone(),
                vec: vec![text.len() as f64, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            })
            .collect())
    }
}
