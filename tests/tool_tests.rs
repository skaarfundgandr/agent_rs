use agent_rs_lib::agent::tools::document::{
    ReadDocumentArgs, ReadDocumentTool, WriteDocumentArgs, WriteDocumentTool,
};
use rig::tool::Tool;
use std::fs;

#[tokio::test]
async fn test_read_txt() {
    let tool = ReadDocumentTool;
    let temp_file = tempfile::Builder::new()
        .suffix(".txt")
        .tempfile()
        .unwrap();
    let path = temp_file.path().to_str().unwrap().to_string();
    fs::write(&path, "Hello World").unwrap();

    let args = ReadDocumentArgs {
        path: path.clone(),
    };
    let result = tool.call(args).await.unwrap();

    assert_eq!(result, "Hello World");
}

#[tokio::test]
#[ignore]
async fn test_read_pdf() {
    let tool = ReadDocumentTool;
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
    let tool = WriteDocumentTool;
    let temp_file = tempfile::Builder::new()
        .suffix(".txt")
        .tempfile()
        .unwrap();
    let path = temp_file.path().to_str().unwrap().to_string();

    // Test overwrite
    let args = WriteDocumentArgs {
        path: path.clone(),
        content: "First line\n".to_string(),
        append: None,
    };
    tool.call(args).await.unwrap();

    // Test append
    let args = WriteDocumentArgs {
        path: path.clone(),
        content: "Second line".to_string(),
        append: Some(true),
    };
    tool.call(args).await.unwrap();

    let content = fs::read_to_string(&path).unwrap();
    assert_eq!(content, "First line\nSecond line");
}
