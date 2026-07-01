use std::future::Future;
use std::time::Duration;

use rig_core::completion::request::PromptError;

pub(crate) fn is_retryable(e: &PromptError) -> bool {
    matches!(
        e,
        PromptError::CompletionError(
            rig_core::completion::request::CompletionError::HttpError(_)
                | rig_core::completion::request::CompletionError::ProviderError(_)
        )
    )
}

pub(crate) async fn retry_with_backoff<F, Fut, T>(
    max_retries: u32,
    mut op: F,
) -> Result<T, PromptError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, PromptError>>,
{
    let mut attempt = 0u32;
    loop {
        attempt += 1;
        match op().await {
            Ok(v) => return Ok(v),
            Err(e) if is_retryable(&e) && attempt < max_retries => {
                tokio::time::sleep(Duration::from_millis(500 * 2u64.pow(attempt - 1))).await;
            }
            Err(e) => return Err(e),
        }
    }
}
