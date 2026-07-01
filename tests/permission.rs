#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use agent_rs::agent::permission::{
    PermissionEvent, PermissionGate, PermissionObserver, PermissionPolicy, PermissionResult,
    PolicyMap,
};
use std::sync::Arc;

#[tokio::test]
async fn allow_all_returns_allow() {
    let result = PermissionPolicy::AllowAll
        .evaluate("any_tool", "any desc")
        .await;
    assert!(matches!(result, PermissionResult::Allow));
}

#[tokio::test]
async fn deny_all_returns_deny_with_reason() {
    let result = PermissionPolicy::DenyAll
        .evaluate("any_tool", "any desc")
        .await;
    match result {
        PermissionResult::Deny { reason } => assert!(!reason.is_empty()),
        other => panic!("expected Deny, got {other:?}"),
    }
}

#[tokio::test]
async fn custom_gate_delegates_to_gate_result() {
    struct AllowGate;
    #[async_trait::async_trait]
    impl PermissionGate for AllowGate {
        async fn check_permission(&self, _: &str, _: &str) -> PermissionResult {
            PermissionResult::Allow
        }
    }
    let policy = PermissionPolicy::Custom(Arc::new(AllowGate));
    let result = policy.evaluate("tool", "desc").await;
    assert!(matches!(result, PermissionResult::Allow));
}

#[tokio::test]
async fn is_allow_helper() {
    assert!(PermissionResult::Allow.is_allow());
    assert!(
        !PermissionResult::Deny {
            reason: "x".to_string()
        }
        .is_allow()
    );
}

#[tokio::test]
async fn policy_map_default_fallback() {
    let map = PolicyMap::new(PermissionPolicy::DenyAll);
    let result = map.evaluate("any_tool", "desc").await;
    assert!(matches!(result, PermissionResult::Deny { .. }));
}

#[tokio::test]
async fn policy_map_override_takes_precedence() {
    let map =
        PolicyMap::new(PermissionPolicy::DenyAll).tool("allow_this", PermissionPolicy::AllowAll);
    let result = map.evaluate("allow_this", "desc").await;
    assert!(matches!(result, PermissionResult::Allow));

    let result2 = map.evaluate("other_tool", "desc").await;
    assert!(matches!(result2, PermissionResult::Deny { .. }));
}

#[tokio::test]
async fn policy_map_no_observer_does_not_panic() {
    let map = PolicyMap::new(PermissionPolicy::AllowAll);
    let _ = map.evaluate("any_tool", "desc").await;
}

#[tokio::test]
async fn policy_map_observer_fires_with_correct_variant() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingObserver {
        count: AtomicUsize,
        last_variant: std::sync::Mutex<&'static str>,
    }
    impl PermissionObserver for CountingObserver {
        fn on_evaluation(&self, event: &PermissionEvent) {
            self.count.fetch_add(1, Ordering::SeqCst);
            *self.last_variant.lock().unwrap() = event.policy_variant;
        }
    }

    let observer = Arc::new(CountingObserver {
        count: AtomicUsize::new(0),
        last_variant: std::sync::Mutex::new(""),
    });
    let map = PolicyMap::new(PermissionPolicy::AllowAll)
        .with_observer(observer.clone())
        .tool("overridden_tool", PermissionPolicy::DenyAll);

    let _ = map.evaluate("overridden_tool", "desc").await;
    assert_eq!(observer.count.load(Ordering::SeqCst), 1);
    assert_eq!(*observer.last_variant.lock().unwrap(), "Override");

    let _ = map.evaluate("default_tool", "desc").await;
    assert_eq!(observer.count.load(Ordering::SeqCst), 2);
    assert_eq!(*observer.last_variant.lock().unwrap(), "AllowAll");
}
