//! OpenTelemetry GenAI + LangSmith semantic-convention attribute names.
//!
//! rig 0.40 natively emits `gen_ai.operation.name`, `gen_ai.usage.*`, and
//! `gen_ai.tool.name` — those constants are no longer needed here. Only
//! LangSmith/OpenInference-specific constants remain.

/// Attribute recording the LangSmith run-typing span kind (see the `KIND_*`
/// constants below).
pub const LANGSMITH_SPAN_KIND: &str = "langsmith.span.kind";
/// Attribute recording the OpenInference span kind (e.g. `"LLM"`, `"TOOL"`).
pub const OPENINFERENCE_SPAN_KIND: &str = "openinference.span.kind";
/// Attribute recording the serialized input of a tool call.
pub const INPUT_VALUE: &str = "input.value";
/// Attribute recording the serialized output of a tool result.
pub const OUTPUT_VALUE: &str = "output.value";
/// GenAI attribute carrying the model's reasoning content.
pub const GEN_AI_REASONING: &str = "gen_ai.content.reasoning";

/// LangSmith run-typing values for `langsmith.span.kind`.
pub const KIND_LLM: &str = "llm";
/// LangSmith run type for a chain span.
pub const KIND_CHAIN: &str = "chain";
/// LangSmith run type for a tool span.
pub const KIND_TOOL: &str = "tool";
/// LangSmith run type for an agent span.
pub const KIND_AGENT: &str = "agent";
/// LangSmith run type for a retriever span.
pub const KIND_RETRIEVER: &str = "retriever";
/// LangSmith run type for an embedding span.
pub const KIND_EMBEDDING: &str = "embedding";
