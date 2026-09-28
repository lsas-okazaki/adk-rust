//! Reactive context compaction when the model rejects a request as too long.
//!
//! An LLM agent reports a failed model call as an `Err` item in its event
//! stream, not as an `Err` from `run()`. These tests pin that the runner
//! compacts and reruns the agent for that shape too, once per invocation.
#![cfg(feature = "context-compaction")]

use adk_core::{
    AdkError, Agent, Content, ErrorCategory, ErrorComponent, Event, EventStream, InvocationContext,
    Result, SessionId, UserId,
};
use adk_runner::Runner;
use adk_runner::compaction::{
    CompactionConfig, CompactionStrategy, apply_reactive_compaction, estimate_event_tokens,
};
use adk_session::{InMemorySessionService, SessionService};
use async_trait::async_trait;
use futures::StreamExt;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

fn overflow() -> AdkError {
    AdkError::new(
        ErrorComponent::Model,
        ErrorCategory::InvalidInput,
        "model.bedrock.context_overflow",
        "Bedrock API error: ValidationException: Input length (135735) exceeds model's \
         maximum context length (131072).",
    )
}

/// Fails its first `failures` runs with an in-stream overflow, then answers.
struct OverflowingAgent {
    failures: usize,
    runs: Arc<AtomicUsize>,
    history_lens: Arc<std::sync::Mutex<Vec<usize>>>,
}

#[async_trait]
impl Agent for OverflowingAgent {
    fn name(&self) -> &str {
        "overflowing_agent"
    }
    fn description(&self) -> &str {
        "fails with a context overflow inside its stream"
    }
    fn sub_agents(&self) -> &[Arc<dyn Agent>] {
        &[]
    }
    async fn run(&self, ctx: Arc<dyn InvocationContext>) -> Result<EventStream> {
        let run = self.runs.fetch_add(1, Ordering::SeqCst);
        self.history_lens.lock().unwrap().push(ctx.session().conversation_history().len());
        if run < self.failures {
            return Ok(Box::pin(futures::stream::iter(vec![Err(overflow())])));
        }
        Ok(Box::pin(futures::stream::once(async {
            let mut event = Event::new("inv");
            event.author = "overflowing_agent".to_string();
            event.set_content(Content::new("model").with_text("answer"));
            Ok(event)
        })))
    }
}

/// Keeps only the newest event — enough to show compaction ran.
struct KeepNewest;

#[async_trait]
impl CompactionStrategy for KeepNewest {
    async fn compact(&self, events: Vec<Event>, _budget: usize) -> Result<Vec<Event>> {
        Ok(events.into_iter().last().into_iter().collect())
    }
}

/// Drops the oldest events until the estimate fits the budget it is given.
struct DropOldestToBudget;

#[async_trait]
impl CompactionStrategy for DropOldestToBudget {
    async fn compact(&self, mut events: Vec<Event>, budget: usize) -> Result<Vec<Event>> {
        while events.len() > 1 && estimate_event_tokens(&events) > budget {
            events.remove(0);
        }
        Ok(events)
    }
}

fn text_event(text: &str) -> Event {
    let mut event = Event::new("inv");
    event.author = "user".to_string();
    event.set_content(Content::new("user").with_text(text));
    event
}

async fn run_once(failures: usize) -> (Vec<Result<Event>>, usize, Vec<usize>) {
    let session_service = Arc::new(InMemorySessionService::new());
    session_service
        .create(adk_session::CreateRequest {
            app_name: "app".to_string(),
            user_id: "user".to_string(),
            session_id: Some("sess".to_string()),
            state: Default::default(),
        })
        .await
        .unwrap();
    // Earlier turns, so compaction has something to drop.
    for i in 0..6 {
        session_service
            .append_event("sess", text_event(&format!("earlier turn {i} {}", "y".repeat(200))))
            .await
            .unwrap();
    }
    let runs = Arc::new(AtomicUsize::new(0));
    let history_lens = Arc::new(std::sync::Mutex::new(Vec::new()));
    let agent = Arc::new(OverflowingAgent {
        failures,
        runs: runs.clone(),
        history_lens: history_lens.clone(),
    });
    let runner = Runner::builder()
        .app_name("app")
        .agent(agent as Arc<dyn Agent>)
        .session_service(session_service as Arc<dyn SessionService>)
        // A budget far above this tiny history: proactive compaction never
        // fires, exactly as in the run whose provider rejected the request.
        .context_compaction(CompactionConfig::new(Box::new(KeepNewest), 1_000_000))
        .build()
        .unwrap();

    let mut stream = runner
        .run(
            UserId::new("user").unwrap(),
            SessionId::new("sess").unwrap(),
            Content::new("user").with_text("hello"),
        )
        .await
        .unwrap();
    let mut results = Vec::new();
    while let Some(r) = stream.next().await {
        results.push(r);
    }
    let lens = history_lens.lock().unwrap().clone();
    (results, runs.load(Ordering::SeqCst), lens)
}

#[tokio::test]
async fn an_in_stream_overflow_is_compacted_and_the_agent_rerun() {
    let (results, runs, history_lens) = run_once(1).await;
    assert_eq!(runs, 2, "the agent should run again after compaction");
    assert!(
        history_lens[1] < history_lens[0],
        "the rerun should see the compacted history: {history_lens:?}"
    );
    assert!(results.iter().all(|r| r.is_ok()), "no error should reach the caller: {results:?}");
    assert!(results.iter().any(|r| r.as_ref().is_ok_and(|e| e.author == "overflowing_agent")));
}

#[tokio::test]
async fn a_second_overflow_in_one_invocation_reaches_the_caller() {
    let (results, runs, _) = run_once(2).await;
    assert_eq!(runs, 2, "reactive compaction is attempted once per invocation");
    let err = results.iter().find_map(|r| r.as_ref().err()).expect("an error");
    assert_eq!(err.category, ErrorCategory::InvalidInput);
}

#[tokio::test]
async fn reactive_compaction_shrinks_a_history_already_under_budget() {
    let events: Vec<Event> = (0..8).map(|i| text_event(&"x".repeat(400 + i))).collect();
    let before = estimate_event_tokens(&events);
    // The budget alone would keep everything: this history is under it, which
    // is why proactive compaction never fired before the provider said no.
    let config = CompactionConfig::new(Box::new(DropOldestToBudget), before * 2);

    let compacted = apply_reactive_compaction(&config, events).await.unwrap();
    let after = estimate_event_tokens(&compacted);
    assert!(after * 100 <= before * 75, "{after} should be at most 75% of {before}");
}
