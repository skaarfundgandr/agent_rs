use agent_rs_lib::agent::tools::document::{
    ReadDocumentArgs, ReadDocumentTool, WriteDocumentArgs, WriteDocumentTool,
};
use rig::tool::Tool;
use std::collections::HashSet;
use std::fs;

#[tokio::test]
async fn test_read_txt() {
    let temp_dir = tempfile::tempdir().unwrap();
    let tool = ReadDocumentTool::new(
        temp_dir.path(),
        HashSet::from(["txt", "md", "pdf"].map(String::from)),
    );

    let file_path = temp_dir.path().join("test_read.txt");
    fs::write(&file_path, "Hello World").unwrap();

    let args = ReadDocumentArgs {
        path: "test_read.txt".to_string(),
    };
    let result = tool.call(args).await.unwrap();

    assert_eq!(result, "Hello World");
}

#[tokio::test]
#[ignore]
async fn test_read_pdf() {
    let tool = ReadDocumentTool::new("./", HashSet::from(["txt", "md", "pdf"].map(String::from)));
    let path = "Stellaron Architecture Overview.pdf";

    let args = ReadDocumentArgs {
        path: path.to_string(),
    };
    let result = tool.call(args).await.unwrap();

    assert!(!result.is_empty());
    assert!(result.contains("Stellaron"));
}

#[tokio::test]
async fn test_write_document() {
    let temp_dir = tempfile::tempdir().unwrap();
    let tool = WriteDocumentTool::new(
        temp_dir.path(),
        HashSet::from(["txt", "md"].map(String::from)),
    );

    // Test overwrite
    let args = WriteDocumentArgs {
        path: "test_write.txt".to_string(),
        content: "First line\n".to_string(),
        append: None,
    };
    tool.call(args).await.unwrap();

    // Test append
    let args = WriteDocumentArgs {
        path: "test_write.txt".to_string(),
        content: "Second line".to_string(),
        append: Some(true),
    };
    tool.call(args).await.unwrap();

    let content = fs::read_to_string(temp_dir.path().join("test_write.txt")).unwrap();
    assert_eq!(content, "First line\nSecond line");
}

#[tokio::test]
async fn test_sandbox_escape_read() {
    let temp_dir = tempfile::tempdir().unwrap();
    let tool = ReadDocumentTool::new(
        temp_dir.path(),
        HashSet::from(["txt", "md", "pdf"].map(String::from)),
    );

    let args = ReadDocumentArgs {
        path: "../escaped.txt".to_string(),
    };
    let err = tool
        .call(args)
        .await
        .expect_err("should reject path traversal");

    assert!(
        matches!(
            err,
            agent_rs_lib::domain::errors::DocumentError::SandboxEscape(_)
        ),
        "Expected SandboxEscape error, got {:?}",
        err
    );
}

#[tokio::test]
async fn test_sandbox_escape_write() {
    let temp_dir = tempfile::tempdir().unwrap();
    let tool = WriteDocumentTool::new(
        temp_dir.path(),
        HashSet::from(["txt", "md"].map(String::from)),
    );

    let args = WriteDocumentArgs {
        path: "../../escaped_write.txt".to_string(),
        content: "malicious".to_string(),
        append: None,
    };
    let err = tool
        .call(args)
        .await
        .expect_err("should reject path traversal");

    assert!(
        matches!(
            err,
            agent_rs_lib::domain::errors::DocumentError::SandboxEscape(_)
        ),
        "Expected SandboxEscape error, got {:?}",
        err
    );
}
