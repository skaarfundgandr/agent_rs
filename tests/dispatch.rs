#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::future::Future;
use std::pin::Pin;

use agent_rs::agent::dispatch::{
    AgentDefinition, AgentDispatcher, AgentInput, AgentKind, AgentOutput,
};
use agent_rs::domain::agent::{FinalAnswer, ReActTrace};

struct MockReActDef {
    name: String,
    tool_groups: Vec<String>,
    description: String,
    max_retries: u32,
    max_cycles: usize,
    preamble: Option<String>,
    answer: String,
}

impl AgentDefinition for MockReActDef {
    fn name(&self) -> &str {
        &self.name
    }
    fn kind(&self) -> AgentKind {
        AgentKind::ReAct
    }
    fn tool_groups(&self) -> &[String] {
        &self.tool_groups
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn max_retries(&self) -> u32 {
        self.max_retries
    }
    fn max_cycles(&self) -> Option<usize> {
        Some(self.max_cycles)
    }
    fn react_preamble(&self) -> Option<&str> {
        self.preamble.as_deref()
    }
    fn run<'a>(
        &'a self,
        _input: AgentInput,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<AgentOutput>> + Send + 'a>> {
        let answer = self.answer.clone();
        Box::pin(async move {
            let trace = ReActTrace {
                prompt: "mock".into(),
                steps: vec![],
                final_answer: Some(FinalAnswer {
                    text: answer.clone(),
                    cycles: 1,
                }),
            };
            Ok(AgentOutput {
                answer,
                trace: Some(trace),
            })
        })
    }
}

struct MockManagedDef {
    name: String,
    tool_groups: Vec<String>,
    description: String,
    max_retries: u32,
    answer: String,
}

impl AgentDefinition for MockManagedDef {
    fn name(&self) -> &str {
        &self.name
    }
    fn kind(&self) -> AgentKind {
        AgentKind::Managed
    }
    fn tool_groups(&self) -> &[String] {
        &self.tool_groups
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn max_retries(&self) -> u32 {
        self.max_retries
    }
    fn max_cycles(&self) -> Option<usize> {
        None
    }
    fn react_preamble(&self) -> Option<&str> {
        None
    }
    fn run<'a>(
        &'a self,
        _input: AgentInput,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<AgentOutput>> + Send + 'a>> {
        let answer = self.answer.clone();
        Box::pin(async move {
            Ok(AgentOutput {
                answer,
                trace: None,
            })
        })
    }
}

fn make_react_def() -> MockReActDef {
    MockReActDef {
        name: "react-agent".into(),
        tool_groups: vec!["fs".into(), "grep".into()],
        description: "A mock ReAct agent".into(),
        max_retries: 3,
        max_cycles: 5,
        preamble: Some("Think carefully.".into()),
        answer: "react-answer".into(),
    }
}

fn make_managed_def() -> MockManagedDef {
    MockManagedDef {
        name: "managed-agent".into(),
        tool_groups: vec!["fs".into()],
        description: "A mock Managed agent".into(),
        max_retries: 2,
        answer: "managed-answer".into(),
    }
}

fn input(prompt: &str) -> AgentInput {
    AgentInput {
        prompt: prompt.into(),
        context: None,
    }
}

#[tokio::test]
async fn dispatch_react_returns_trace() -> anyhow::Result<()> {
    let mut dispatcher = AgentDispatcher::new();
    dispatcher.register(Box::new(make_react_def()))?;
    let out = dispatcher.dispatch("react-agent", input("hello")).await?;
    assert_eq!(out.answer, "react-answer");
    assert!(out.trace.is_some());
    Ok(())
}

#[tokio::test]
async fn dispatch_managed_returns_no_trace() -> anyhow::Result<()> {
    let mut dispatcher = AgentDispatcher::new();
    dispatcher.register(Box::new(make_managed_def()))?;
    let out = dispatcher.dispatch("managed-agent", input("hello")).await?;
    assert_eq!(out.answer, "managed-answer");
    assert!(out.trace.is_none());
    Ok(())
}

#[tokio::test]
async fn dispatch_missing_agent_errors() {
    let dispatcher = AgentDispatcher::new();
    let err = dispatcher
        .dispatch("nope", input("hello"))
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("nope"),
        "error should mention the missing name: {err}"
    );
}

#[test]
fn accessors_reflect_config() {
    let react = make_react_def();
    assert_eq!(react.kind(), AgentKind::ReAct);
    assert_eq!(react.tool_groups(), &["fs", "grep"]);
    assert_eq!(react.max_cycles(), Some(5));
    assert_eq!(react.react_preamble(), Some("Think carefully."));

    let managed = make_managed_def();
    assert_eq!(managed.kind(), AgentKind::Managed);
    assert_eq!(managed.tool_groups(), &["fs"]);
    assert_eq!(managed.max_cycles(), None);
    assert_eq!(managed.react_preamble(), None);
}

#[test]
fn names_returns_all_registered() -> anyhow::Result<()> {
    let mut dispatcher = AgentDispatcher::new();
    dispatcher.register(Box::new(make_react_def()))?;
    dispatcher.register(Box::new(make_managed_def()))?;
    let mut names: Vec<String> = dispatcher.names().map(String::from).collect();
    names.sort();
    assert_eq!(names, vec!["managed-agent", "react-agent"]);
    Ok(())
}

#[test]
fn register_duplicate_name_errors() {
    let mut dispatcher = AgentDispatcher::new();
    dispatcher.register(Box::new(make_react_def())).unwrap();
    let err = dispatcher.register(Box::new(make_react_def())).unwrap_err();
    assert!(
        err.to_string().contains("react-agent"),
        "error should mention the duplicate name: {err}"
    );
}
