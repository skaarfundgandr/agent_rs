//! OpenTelemetry GenAI + LangSmith semantic-convention attribute names.

pub const GEN_AI_SYSTEM: &str = "gen_ai.system";
pub const GEN_AI_OPERATION_NAME: &str = "gen_ai.operation.name";
pub const GEN_AI_TOOL_NAME: &str = "gen_ai.tool.name";
pub const GEN_AI_USAGE_INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";
pub const GEN_AI_USAGE_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";
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
