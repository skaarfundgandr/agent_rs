# AgentRS Examples

This directory contains functional example applications showcasing the features of the `agent_rs` framework.

## 🚀 Prerequisites

Before running any example, ensure you have:
1. An OpenAI-compatible LLM provider running (e.g. Google Gemma, LM Studio, or OpenAI).
2. Environment variables configured in a `.env` file in the workspace root:
   ```env
   # Core API configuration
   API_KEY=your-api-key
   CHAT_MODEL=google/gemma-4-e4b
   CHAT_BASE_URL=http://127.0.0.1:1234/v1

   # LangSmith tracing (Required only for the langsmith_react example)
   LANGSMITH_API_KEY=your-langsmith-key
   LANGSMITH_PROJECT=default
   OTEL_EXPORTER_OTLP_ENDPOINT=https://api.smith.langchain.com/otel/v1/traces
   OTEL_SERVICE_NAME=agent_rs
   ```

---

## 🛠️ Available Examples

All examples must be run from the workspace root directory:

### 1. Interactive Chatbot (with RAG & MCP)
```bash
cargo run --example cli_chatbot --features rag
```
*   **File**: [cli_chatbot.rs](cli_chatbot.rs)
*   **Description**: An interactive terminal chatbot with local file reading/writing, PDF/text dynamic ingestion, and MCP tool registries.

### 2. Autonomous Web Researcher (Dynamic RAG)
```bash
cargo run --example web_researcher --features rag
```
*   **File**: [web_researcher.rs](web_researcher.rs)
*   **Description**: Queries search tools, writes fetched web pages to the local sandbox, registers them into `RagPipeline`, and queries them dynamically.

### 3. Multi-Agent Coder & Reviewer
```bash
cargo run --example multi_agent_coder
```
*   **File**: [multi_agent_coder.rs](multi_agent_coder.rs)
*   **Description**: Uses `AgentDispatcher` to orchestrate a ReAct developer agent and a Managed reviewer agent working together.

### 4. Secure Database Assistant (Permission Gates)
```bash
cargo run --example db_assistant
```
*   **File**: [db_assistant.rs](db_assistant.rs)
*   **Description**: Uses `PolicyMap` to allow read queries automatically but prompts the user for confirmation via CLI when attempting write operations.

### 5. LangSmith & OpenTelemetry Tracing
```bash
cargo run --example langsmith_react --features opentelemetry
```
*   **File**: [langsmith_react.rs](langsmith_react.rs)
*   **Description**: Traces each step of the ReAct cycle with custom OTel spans exported to LangSmith. Requires the LangSmith/OTel environment variables to be configured in `.env`.

