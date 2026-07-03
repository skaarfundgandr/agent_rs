//! OpenTelemetry GenAI + LangSmith semantic-convention attribute names.
//!
//! rig 0.39 natively emits `gen_ai.operation.name`, `gen_ai.usage.*`, and
//! `gen_ai.tool.name` — those constants are no longer needed here. Only
//! LangSmith/OpenInference-specific constants remain.

pub const LANGSMITH_SPAN_KIND: &str = "langsmith.span.kind";
pub const OPENINFERENCE_SPAN_KIND: &str = "openinference.span.kind";
pub const INPUT_VALUE: &str = "input.value";
pub const OUTPUT_VALUE: &str = "output.value";
pub const GEN_AI_REASONING: &str = "gen_ai.content.reasoning";

/// LangSmith run-typing values for `langsmith.span.kind`.
pub const KIND_LLM: &str = "llm";
pub const KIND_CHAIN: &str = "chain";
pub const KIND_TOOL: &str = "tool";
pub const KIND_AGENT: &str = "agent";
pub const KIND_RETRIEVER: &str = "retriever";
pub const KIND_EMBEDDING: &str = "embedding";
