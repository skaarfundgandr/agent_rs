//! Shared test utilities for integration tests.
#![allow(dead_code, clippy::duplicate_mod, clippy::manual_async_fn)]

use rig_core::OneOrMany;
use rig_core::agent::{AgentBuilder, StreamingPromptRequest};
use rig_core::completion::{
    CompletionError, CompletionModel, CompletionRequest, CompletionResponse, Usage,
};
use rig_core::message::{AssistantContent, ToolCall, ToolFunction};
use rig_core::streaming::{RawStreamingChoice, StreamingChat, StreamingCompletionResponse};
use rig_core::tool::Tool;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

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
    /// Note: CompletionError is not Clone, so we wrap the error kind as a string
    /// and reconstruct the error on demand.
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
                    let _ = msg; // informational only
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
    type Hook = ();

    fn stream_chat<I, T>(
        &self,
        prompt: impl Into<rig_core::message::Message> + rig_core::wasm_compat::WasmCompatSend,
        chat_history: I,
    ) -> StreamingPromptRequest<MockCompletionModel, ()>
    where
        I: IntoIterator<Item = T> + rig_core::wasm_compat::WasmCompatSend,
        T: Into<rig_core::message::Message>,
    {
        // Build a minimal Agent via AgentBuilder, wrap in Arc, and pass to
        // StreamingPromptRequest::new(). Agent is non-exhaustive, so we
        // must go through the builder.
        let agent = AgentBuilder::new(self.clone()).default_max_turns(1).build();
        StreamingPromptRequest::<MockCompletionModel, ()>::new(Arc::new(agent), prompt)
            .with_history(chat_history)
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

    async fn definition(&self, _prompt: String) -> rig_core::completion::ToolDefinition {
        rig_core::completion::ToolDefinition {
            name: "echo".to_string(),
            description: "Echo back the input text".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to echo back"
                    }
                },
                "required": ["text"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(EchoOutput(args.text))
    }
}

/// Build an `Agent<MockCompletionModel, ()>` with the given response queue
/// and an `echo` tool registered. Sets `default_max_turns(1)` so rig-core's
/// internal multi-turn loop runs exactly once per `agent.prompt()` call,
/// preventing it from consuming multiple mock responses.
pub fn mock_agent(responses: Vec<MockResponse>) -> rig_core::agent::Agent<MockCompletionModel, ()> {
    let model = MockCompletionModel::new(responses);
    AgentBuilder::new(model)
        .tool(EchoTool)
        .default_max_turns(1)
        .build()
}
