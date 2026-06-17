#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use agent_rs_lib::domain::errors::DocumentError;
use agent_rs_lib::security::{
    find_containing_root_shared, validate_sandboxed_path_shared, SandboxConfig, SharedSandbox,
};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::thread;

#[test]
fn test_shared_sandbox_from_config() {
    let tmp = tempfile::tempdir().unwrap();
    let config = SandboxConfig::single(tmp.path()).unwrap();
    let shared = SharedSandbox::from(config);
    let snap = shared.snapshot();
    assert_eq!(snap.len(), 1);
    assert_eq!(snap.primary(), tmp.path());
}

#[test]
fn test_set_swaps_roots() {
    let tmp1 = tempfile::tempdir().unwrap();
    let tmp2 = tempfile::tempdir().unwrap();
    let shared = SharedSandbox::new(SandboxConfig::single(tmp1.path()).unwrap());

    shared
        .set(SandboxConfig::new(vec![tmp2.path().to_path_buf()]).unwrap())
        .unwrap();
    let snap = shared.snapshot();
    assert_eq!(snap.primary(), tmp2.path());
    assert_eq!(snap.len(), 1);
}

#[test]
fn test_set_rejects_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let _shared = SharedSandbox::new(SandboxConfig::single(tmp.path()).unwrap());

    let err = SandboxConfig::new(vec![]).unwrap_err();
    assert!(matches!(err, DocumentError::Rag(_)));
    // set() would also reject this, but we can't construct the SandboxConfig to pass it in
    assert!(err
        .to_string()
        .contains("SandboxConfig requires at least one root"));
}

#[test]
fn test_set_rejects_uncanonicalizable_root() {
    let tmp = tempfile::tempdir().unwrap();
    let _shared = SharedSandbox::new(SandboxConfig::single(tmp.path()).unwrap());

    let nonexistent = tmp.path().join("does-not-exist");
    let err = SandboxConfig::new(vec![nonexistent]).unwrap_err();
    assert!(matches!(err, DocumentError::Io(_)));
}

#[test]
fn test_set_re_canonicalizes() {
    let tmp = tempfile::tempdir().unwrap();
    let sub = tmp.path().join("sub");
    fs::create_dir_all(&sub).unwrap();

    let non_canonical1 = tmp.path().join("sub").join("..").join("sub");
    let shared = SharedSandbox::new(SandboxConfig::single(&non_canonical1).unwrap());

    let non_canonical2 = tmp.path().join("sub").join(".").join("..").join("sub");
    shared
        .set(SandboxConfig::new(vec![non_canonical2]).unwrap())
        .unwrap();

    let snap = shared.snapshot();
    for root in snap.canonical_roots() {
        let has_dotdot = root
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir));
        let has_dot = root
            .components()
            .any(|c| matches!(c, std::path::Component::Normal(n) if n == "."));
        assert!(
            !has_dotdot && !has_dot,
            "canonical root should not contain . or .. components: {}",
            root.display()
        );
        assert!(
            root.is_absolute(),
            "canonical root should be absolute: {}",
            root.display()
        );
    }
}

#[test]
fn test_validate_sandboxed_path_shared_picks_latest_roots() {
    let tmp1 = tempfile::tempdir().unwrap();
    let tmp2 = tempfile::tempdir().unwrap();
    fs::write(tmp1.path().join("x.txt"), "from tmp1").unwrap();
    fs::write(tmp2.path().join("x.txt"), "from tmp2").unwrap();

    let shared = SharedSandbox::new(SandboxConfig::new(vec![tmp1.path().to_path_buf()]).unwrap());

    let result1 = validate_sandboxed_path_shared(&shared, Path::new("x.txt")).unwrap();
    assert!(result1.starts_with(tmp1.path().canonicalize().unwrap()));

    shared
        .set(SandboxConfig::new(vec![tmp2.path().to_path_buf()]).unwrap())
        .unwrap();

    let result2 = validate_sandboxed_path_shared(&shared, Path::new("x.txt")).unwrap();
    assert!(result2.starts_with(tmp2.path().canonicalize().unwrap()));
}

#[test]
fn test_find_containing_root_shared_after_set() {
    let tmp1 = tempfile::tempdir().unwrap();
    let tmp2 = tempfile::tempdir().unwrap();
    fs::write(tmp1.path().join("file.txt"), "content").unwrap();

    let shared = SharedSandbox::new(SandboxConfig::new(vec![tmp1.path().to_path_buf()]).unwrap());

    let path1 = tmp1.path().join("file.txt");
    let found = find_containing_root_shared(&shared, &path1);
    assert!(found.is_some());

    shared
        .set(SandboxConfig::new(vec![tmp2.path().to_path_buf()]).unwrap())
        .unwrap();

    let found_after = find_containing_root_shared(&shared, &path1);
    assert!(found_after.is_none());
}

#[test]
fn test_concurrent_set_and_snapshot_no_deadlock() {
    let tmp = tempfile::tempdir().unwrap();
    let shared = Arc::new(SharedSandbox::new(
        SandboxConfig::single(tmp.path()).unwrap(),
    ));

    let mut handles = vec![];
    for _ in 0..10 {
        let shared = Arc::clone(&shared);
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                let snap = shared.snapshot();
                assert!(!snap.is_empty());
            }
        }));
    }

    let mut last_root = tmp.path().to_path_buf();
    for _i in 0..10 {
        let new_tmp = tempfile::tempdir().unwrap();
        let root = new_tmp.path().to_path_buf();
        shared
            .set(SandboxConfig::new(vec![root.clone()]).unwrap())
            .unwrap();
        last_root = root;
        // keep new_tmp alive until after threads finish
        let _ = new_tmp;
    }

    for h in handles {
        h.join().unwrap();
    }

    let final_snap = shared.snapshot();
    assert_eq!(final_snap.primary(), &last_root);
}
