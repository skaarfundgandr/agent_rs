#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use agent_rs::domain::errors::DocumentError;
use agent_rs::security::{
    SandboxConfig, SharedSandbox, find_containing_root_shared, validate_sandboxed_path_shared,
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
    assert!(
        err.to_string()
            .contains("SandboxConfig requires at least one root")
    );
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

// --- Phase 4 tests: add_root / remove_root / add_roots / contains_root ---

#[test]
fn test_add_root_appends_canonical() {
    let tmp1 = tempfile::tempdir().unwrap();
    let tmp2 = tempfile::tempdir().unwrap();
    let mut config = SandboxConfig::single(tmp1.path()).unwrap();
    assert_eq!(config.len(), 1);

    config.add_root(tmp2.path()).unwrap();
    assert_eq!(config.len(), 2);
    assert_eq!(config.primary(), tmp1.path());
    assert!(
        config
            .canonical_roots()
            .contains(&tmp2.path().canonicalize().unwrap())
    );
}

#[test]
fn test_add_root_is_idempotent() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = SandboxConfig::single(tmp.path()).unwrap();

    // add same root by original path
    config.add_root(tmp.path()).unwrap();
    assert_eq!(config.len(), 1);

    // add same root by canonical form
    config.add_root(tmp.path().canonicalize().unwrap()).unwrap();
    assert_eq!(config.len(), 1);
}

#[test]
fn test_add_root_rejects_uncanonicalizable() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = SandboxConfig::single(tmp.path()).unwrap();
    let nonexistent = tmp.path().join("does-not-exist");

    let err = config.add_root(&nonexistent).unwrap_err();
    assert!(matches!(err, DocumentError::Io(_)));
    assert_eq!(config.len(), 1);

    // verify write lock was released by successfully calling set()
    let new_config = SandboxConfig::single(tmp.path()).unwrap();
    SharedSandbox::from(config).set(new_config).unwrap();
}

#[test]
fn test_add_roots_batch_appends_all() {
    let tmp1 = tempfile::tempdir().unwrap();
    let tmp2 = tempfile::tempdir().unwrap();
    let tmp3 = tempfile::tempdir().unwrap();
    let mut config = SandboxConfig::single(tmp1.path()).unwrap();

    config.add_roots(vec![tmp2.path(), tmp3.path()]).unwrap();
    assert_eq!(config.len(), 3);
    assert!(
        config
            .canonical_roots()
            .contains(&tmp2.path().canonicalize().unwrap())
    );
    assert!(
        config
            .canonical_roots()
            .contains(&tmp3.path().canonicalize().unwrap())
    );
}

#[test]
fn test_add_roots_partial_failure_leaves_partial_state() {
    let tmp1 = tempfile::tempdir().unwrap();
    let tmp2 = tempfile::tempdir().unwrap();
    let nonexistent = tmp1.path().join("does-not-exist");
    let mut config = SandboxConfig::single(tmp1.path()).unwrap();

    let err = config
        .add_roots(vec![tmp2.path(), &nonexistent])
        .unwrap_err();
    assert!(matches!(err, DocumentError::Io(_)));
    // tmp2 was added before the failure
    assert_eq!(config.len(), 2);
    assert!(
        config
            .canonical_roots()
            .contains(&tmp2.path().canonicalize().unwrap())
    );
}

#[test]
fn test_add_roots_dedup_across_batch() {
    let tmp1 = tempfile::tempdir().unwrap();
    let tmp2 = tempfile::tempdir().unwrap();
    let mut config = SandboxConfig::single(tmp1.path()).unwrap();

    let canonical = tmp2.path().canonicalize().unwrap();
    config.add_roots(vec![tmp2.path(), &canonical]).unwrap();
    // only one new root added (dedup by canonical form)
    assert_eq!(config.len(), 2);
}

#[test]
fn test_remove_root_drops_entry() {
    let tmp1 = tempfile::tempdir().unwrap();
    let tmp2 = tempfile::tempdir().unwrap();
    let mut config =
        SandboxConfig::new(vec![tmp1.path().to_path_buf(), tmp2.path().to_path_buf()]).unwrap();
    assert_eq!(config.len(), 2);

    config.remove_root(tmp1.path()).unwrap();
    assert_eq!(config.len(), 1);
    assert!(
        config
            .canonical_roots()
            .contains(&tmp2.path().canonicalize().unwrap())
    );
}

#[test]
fn test_remove_last_root_errors_sandbox() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = SandboxConfig::single(tmp.path()).unwrap();

    let err = config.remove_root(tmp.path()).unwrap_err();
    assert!(matches!(err, DocumentError::Sandbox(_)));
    assert_eq!(config.len(), 1);
}

#[test]
fn test_remove_missing_root_is_ok() {
    let tmp = tempfile::tempdir().unwrap();
    let other = tempfile::tempdir().unwrap();
    let mut config = SandboxConfig::single(tmp.path()).unwrap();

    // removing a root that isn't in the config is a no-op
    config.remove_root(other.path()).unwrap();
    assert_eq!(config.len(), 1);
}

#[test]
fn test_remove_root_can_promote_primary() {
    let tmp1 = tempfile::tempdir().unwrap();
    let tmp2 = tempfile::tempdir().unwrap();
    let mut config =
        SandboxConfig::new(vec![tmp1.path().to_path_buf(), tmp2.path().to_path_buf()]).unwrap();
    assert_eq!(config.primary(), tmp1.path());

    config.remove_root(tmp1.path()).unwrap();
    // tmp2 is now the sole root and thus primary
    assert_eq!(config.primary(), tmp2.path());
}

#[test]
fn test_contains_root_strict_canonical_comparison() {
    let tmp = tempfile::tempdir().unwrap();
    let other = tempfile::tempdir().unwrap();
    let config = SandboxConfig::single(tmp.path()).unwrap();

    // original path
    assert!(config.contains_root(tmp.path()).unwrap());
    // canonical form
    assert!(
        config
            .contains_root(tmp.path().canonicalize().unwrap())
            .unwrap()
    );
    // unrelated tempdir
    assert!(!config.contains_root(other.path()).unwrap());
    // nonexistent path → Io error
    let nonexistent = tmp.path().join("nope");
    let err = config.contains_root(&nonexistent).unwrap_err();
    assert!(matches!(err, DocumentError::Io(_)));
}

#[test]
fn test_add_then_validation_picks_up_new_root() {
    let tmp1 = tempfile::tempdir().unwrap();
    let tmp2 = tempfile::tempdir().unwrap();
    // Create file ONLY under tmp2 — not under tmp1
    fs::write(tmp2.path().join("only_in_tmp2.txt"), "hello").unwrap();

    let shared = SharedSandbox::new(SandboxConfig::single(tmp1.path()).unwrap());

    // Absolute path outside tmp1 → Err (absolute path escapes the single root)
    let abs_path = tmp2.path().join("only_in_tmp2.txt");
    assert!(validate_sandboxed_path_shared(&shared, &abs_path).is_err());

    // add tmp2 as a root
    shared.add_root(tmp2.path()).unwrap();

    // now the absolute path resolves under tmp2
    let resolved = validate_sandboxed_path_shared(&shared, &abs_path).unwrap();
    assert!(resolved.starts_with(tmp2.path().canonicalize().unwrap()));
}

#[test]
fn test_remove_then_validation_rejects_former_root() {
    let tmp1 = tempfile::tempdir().unwrap();
    let tmp2 = tempfile::tempdir().unwrap();
    // Create file ONLY under tmp1 — not under tmp2
    fs::write(tmp1.path().join("only_in_tmp1.txt"), "hello").unwrap();

    let shared = SharedSandbox::new(
        SandboxConfig::new(vec![tmp1.path().to_path_buf(), tmp2.path().to_path_buf()]).unwrap(),
    );

    // Absolute path under tmp1 → Ok
    let abs_path = tmp1.path().join("only_in_tmp1.txt");
    let resolved = validate_sandboxed_path_shared(&shared, &abs_path).unwrap();
    assert!(resolved.starts_with(tmp1.path().canonicalize().unwrap()));

    // remove tmp1
    shared.remove_root(tmp1.path()).unwrap();

    // Absolute path no longer in any root → Err
    assert!(validate_sandboxed_path_shared(&shared, &abs_path).is_err());
}

#[test]
fn test_concurrent_add_remove_snapshot_no_deadlock() {
    let tmp = tempfile::tempdir().unwrap();
    let shared = Arc::new(SharedSandbox::new(
        SandboxConfig::single(tmp.path()).unwrap(),
    ));

    // Pre-populate with multiple roots so remove_root has something to work with
    // even when concurrent threads are removing roots.
    let extra_dirs: Vec<tempfile::TempDir> = (0..5).map(|_| tempfile::tempdir().unwrap()).collect();
    for d in &extra_dirs {
        shared.add_root(d.path()).unwrap();
    }

    let mut handles = vec![];
    for _ in 0..8 {
        let shared = Arc::clone(&shared);
        handles.push(thread::spawn(move || {
            for _ in 0..10 {
                let snap = shared.snapshot();
                assert!(!snap.is_empty());
            }
        }));
    }

    // Concurrent add/remove from the main thread
    for d in &extra_dirs {
        let _ = shared.remove_root(d.path());
    }

    for h in handles {
        h.join().unwrap();
    }

    // Should still have at least the original root
    let final_snap = shared.snapshot();
    assert!(!final_snap.is_empty());
}
