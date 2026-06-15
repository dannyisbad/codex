//! Runs the external `cyrus` setup engine for Hannah Montana mode and renders
//! its live progress.
//!
//! The only integration point is spawning the `cyrus` binary and parsing its
//! `--json` stdout (one JSON object per line). This module owns the progress
//! model, the event reducer, and the async run that drives the
//! `SignInState::HannahMontanaSetup` UI.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::RwLock;

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::prelude::Widget;
use ratatui::style::Color;
use ratatui::style::Style;
use ratatui::style::Stylize;
use ratatui::text::Line;
use ratatui::widgets::Paragraph;
use ratatui::widgets::Wrap;
use serde::Deserialize;
use tokio::io::AsyncBufReadExt;
use tokio::io::BufReader;
use tokio::process::Command;
use uuid::Uuid;

use crate::tui::FrameRequester;

use super::AuthModeWidget;
use super::SignInState;

/// The ordered setup steps reported by `cyrus setup --json`. Keys match the
/// `step` field of the JSON event contract.
const SETUP_STEPS: [(&str, &str); 6] = [
    ("secrets", "Preparing credentials"),
    ("chrome", "Connecting to Chrome"),
    ("tunnel", "Opening the public tunnel"),
    ("stack", "Starting local servers"),
    ("connector", "Wiring the ChatGPT connector"),
    // Not "Writing config": the provider is injected at launch, nothing is
    // written to the user's codex config (the engine cleans stale blocks).
    ("codex_config", "Configuring codex"),
];

/// How `cyrus` should expose itself to the user, chosen via the tunnel picker.
/// Maps directly to the `--tunnel` flag passed to `cyrus setup`.
pub(crate) enum TunnelArg {
    Quick,
    Ngrok(String),
    Named,
}

/// Builds the `cyrus setup` arguments for a tunnel selection, appended after
/// `--json`:
/// - Quick    -> `--tunnel quick`
/// - Named    -> `--tunnel named`
/// - Ngrok(d) -> `--tunnel ngrok --ngrok-domain <d>`
pub(crate) fn tunnel_args(tunnel: &TunnelArg) -> Vec<String> {
    match tunnel {
        TunnelArg::Quick => vec!["--tunnel".to_string(), "quick".to_string()],
        TunnelArg::Named => vec!["--tunnel".to_string(), "named".to_string()],
        TunnelArg::Ngrok(domain) => vec![
            "--tunnel".to_string(),
            "ngrok".to_string(),
            "--ngrok-domain".to_string(),
            domain.clone(),
        ],
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum StepStatus {
    Pending,
    Running,
    Done,
}

#[derive(Clone)]
pub(crate) struct SetupStep {
    pub key: String,
    pub label: String,
    pub status: StepStatus,
    pub detail: Option<String>,
}

/// Progress model for an in-flight (or failed) Hannah Montana setup run.
#[derive(Clone)]
pub(crate) struct HannahMontanaSetupState {
    /// Distinguishes one spawned run from another so a stale task does not
    /// clobber the state of a newer attempt (mirrors the device-code guard).
    pub(crate) attempt_id: String,
    pub(crate) steps: Vec<SetupStep>,
    /// Prompt for the user to act (e.g. "log in to ChatGPT"), if any.
    pub(crate) needs_user_action: Option<String>,
    /// Terminal error; when set the run has failed and can be retried.
    pub(crate) error: Option<String>,
    /// Actionable hint shown alongside `error`, when the engine supplied one.
    pub(crate) remedy: Option<String>,
}

impl HannahMontanaSetupState {
    pub(crate) fn new(attempt_id: String) -> Self {
        let steps = SETUP_STEPS
            .iter()
            .map(|(key, label)| SetupStep {
                key: (*key).to_string(),
                label: (*label).to_string(),
                status: StepStatus::Pending,
                detail: None,
            })
            .collect();
        Self {
            attempt_id,
            steps,
            needs_user_action: None,
            error: None,
            remedy: None,
        }
    }

    fn step_mut(&mut self, key: &str) -> Option<&mut SetupStep> {
        self.steps.iter_mut().find(|step| step.key == key)
    }
}

/// One JSON object emitted on a line of the `cyrus setup --json` stream.
#[derive(Debug, Deserialize, PartialEq)]
#[serde(tag = "event", rename_all = "snake_case")]
pub(crate) enum SetupEvent {
    StepStarted {
        step: String,
        #[allow(dead_code)]
        label: Option<String>,
    },
    StepDone {
        step: String,
        detail: Option<String>,
    },
    NeedsUserAction {
        #[allow(dead_code)]
        step: String,
        instruction: String,
    },
    UserActionResolved {
        #[allow(dead_code)]
        step: String,
    },
    Done {
        #[serde(default)]
        #[allow(dead_code)]
        public_url: Option<String>,
        #[serde(default)]
        #[allow(dead_code)]
        shim_base_url: Option<String>,
        #[serde(default)]
        #[allow(dead_code)]
        connector_id: Option<String>,
        #[serde(default)]
        #[allow(dead_code)]
        link_id: Option<String>,
        #[serde(default)]
        #[allow(dead_code)]
        tool_count: Option<u32>,
        #[serde(default)]
        #[allow(dead_code)]
        fully_reused: Option<bool>,
    },
    Error {
        message: String,
        /// Actionable hint for the failed step (engine `Step::remedy()`); absent
        /// for front-end-internal errors (e.g. failing to spawn `cyrus`).
        #[serde(default)]
        remedy: Option<String>,
    },
}

/// Outcome of applying one event to the progress model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReduceOutcome {
    /// Setup is still running; keep reading events.
    Continue,
    /// Setup finished successfully (a `done` event was seen).
    Completed,
    /// Setup failed (an `error` event was seen); `state.error` is set.
    Failed,
}

/// Applies a single parsed event to the progress model, returning whether the
/// run should continue, has completed, or has failed.
pub(crate) fn reduce(state: &mut HannahMontanaSetupState, event: SetupEvent) -> ReduceOutcome {
    match event {
        SetupEvent::StepStarted { step, .. } => {
            if let Some(s) = state.step_mut(&step) {
                s.status = StepStatus::Running;
            }
            ReduceOutcome::Continue
        }
        SetupEvent::StepDone { step, detail } => {
            if let Some(s) = state.step_mut(&step) {
                s.status = StepStatus::Done;
                s.detail = detail;
            }
            // A finished step clears any action prompt scoped to it.
            state.needs_user_action = None;
            ReduceOutcome::Continue
        }
        SetupEvent::NeedsUserAction { instruction, .. } => {
            state.needs_user_action = Some(instruction);
            ReduceOutcome::Continue
        }
        SetupEvent::UserActionResolved { .. } => {
            state.needs_user_action = None;
            ReduceOutcome::Continue
        }
        SetupEvent::Done { .. } => {
            for step in &mut state.steps {
                step.status = StepStatus::Done;
            }
            state.needs_user_action = None;
            state.error = None;
            ReduceOutcome::Completed
        }
        SetupEvent::Error { message, remedy } => {
            state.needs_user_action = None;
            state.error = Some(message);
            state.remedy = remedy;
            ReduceOutcome::Failed
        }
    }
}

/// Resolves the `cyrus` binary to invoke:
/// 1. `CYRUS_BIN` env var (full path to the executable),
/// 2. a sibling of the current executable named `cyrus`/`cyrus.exe`,
/// 3. a bare `cyrus` command on PATH (trusting PATH; spawn fails if absent).
pub(crate) fn locate_cyrus_bin() -> Option<PathBuf> {
    if let Some(bin) = std::env::var_os("CYRUS_BIN") {
        let path = PathBuf::from(bin);
        if !path.as_os_str().is_empty() {
            return Some(path);
        }
    }

    if let Ok(current) = std::env::current_exe()
        && let Some(dir) = current.parent()
    {
        let sibling = dir.join(cyrus_exe_name());
        if sibling.exists() {
            return Some(sibling);
        }
    }

    // Trust PATH: return the bare command and let the spawn surface failures.
    Some(PathBuf::from("cyrus"))
}

fn cyrus_exe_name() -> &'static str {
    if cfg!(windows) { "cyrus.exe" } else { "cyrus" }
}

/// Begins (or restarts) the Hannah Montana setup run: installs the initial
/// progress state, schedules a frame, and spawns the task that drives `cyrus`.
pub(super) fn start_hannah_montana_setup(widget: &AuthModeWidget, tunnel: TunnelArg) {
    let attempt_id = Uuid::new_v4().to_string();
    *widget.sign_in_state.write().unwrap() =
        SignInState::HannahMontanaSetup(HannahMontanaSetupState::new(attempt_id.clone()));
    widget.request_frame.schedule_frame();

    let sign_in_state = widget.sign_in_state.clone();
    let request_frame = widget.request_frame.clone();
    let cwd = widget.cwd.clone();
    let bin = locate_cyrus_bin();

    tokio::spawn(async move {
        run_setup(sign_in_state, request_frame, attempt_id, cwd, bin, tunnel).await;
    });
}

async fn run_setup(
    sign_in_state: Arc<RwLock<SignInState>>,
    request_frame: FrameRequester,
    attempt_id: String,
    cwd: PathBuf,
    bin: Option<PathBuf>,
    tunnel: TunnelArg,
) {
    let Some(bin) = bin else {
        finish_with_error(
            &sign_in_state,
            &request_frame,
            &attempt_id,
            "Could not locate the cyrus binary. Set CYRUS_BIN or add cyrus to PATH.".to_string(),
        );
        return;
    };

    let mut child = match Command::new(&bin)
        .arg("setup")
        .arg("--repo")
        .arg(&cwd)
        .arg("--json")
        .args(tunnel_args(&tunnel))
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            finish_with_error(
                &sign_in_state,
                &request_frame,
                &attempt_id,
                format!("Failed to launch {}: {err}", bin.display()),
            );
            return;
        }
    };

    let Some(stdout) = child.stdout.take() else {
        finish_with_error(
            &sign_in_state,
            &request_frame,
            &attempt_id,
            "cyrus did not provide a stdout stream".to_string(),
        );
        return;
    };

    let mut lines = BufReader::new(stdout).lines();
    loop {
        match lines.next_line().await {
            Ok(Some(line)) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let event = match serde_json::from_str::<SetupEvent>(trimmed) {
                    Ok(event) => event,
                    Err(err) => {
                        tracing::warn!("ignoring unparseable cyrus event: {err}: {trimmed}");
                        continue;
                    }
                };
                match apply_event_for_active_attempt(
                    &sign_in_state,
                    &request_frame,
                    &attempt_id,
                    event,
                ) {
                    // The attempt is no longer active; a newer run superseded us.
                    AttemptUpdate::Stale => return,
                    AttemptUpdate::Applied(ReduceOutcome::Continue) => {}
                    AttemptUpdate::Applied(ReduceOutcome::Completed) => {
                        complete_for_active_attempt(&sign_in_state, &request_frame, &attempt_id);
                        return;
                    }
                    // Error already recorded by the reducer; stop reading.
                    AttemptUpdate::Applied(ReduceOutcome::Failed) => return,
                }
            }
            Ok(None) => break,
            Err(err) => {
                finish_with_error(
                    &sign_in_state,
                    &request_frame,
                    &attempt_id,
                    format!("Error reading cyrus output: {err}"),
                );
                return;
            }
        }
    }

    // Stream ended without a terminal `done`/`error`. Surface the exit status.
    let message = match child.wait().await {
        Ok(status) if status.success() => "cyrus exited before completing setup".to_string(),
        Ok(status) => format!("cyrus exited with {status}"),
        Err(err) => format!("Failed to wait for cyrus: {err}"),
    };
    finish_with_error(&sign_in_state, &request_frame, &attempt_id, message);
}

enum AttemptUpdate {
    Applied(ReduceOutcome),
    Stale,
}

fn attempt_matches(state: &SignInState, attempt_id: &str) -> bool {
    matches!(
        state,
        SignInState::HannahMontanaSetup(state) if state.attempt_id == attempt_id
    )
}

fn apply_event_for_active_attempt(
    sign_in_state: &Arc<RwLock<SignInState>>,
    request_frame: &FrameRequester,
    attempt_id: &str,
    event: SetupEvent,
) -> AttemptUpdate {
    let mut guard = sign_in_state.write().unwrap();
    let SignInState::HannahMontanaSetup(state) = &mut *guard else {
        return AttemptUpdate::Stale;
    };
    if state.attempt_id != attempt_id {
        return AttemptUpdate::Stale;
    }
    let outcome = reduce(state, event);
    drop(guard);
    request_frame.schedule_frame();
    AttemptUpdate::Applied(outcome)
}

/// Transitions a successful run to the existing `HannahMontanaConfigured`
/// state so the onboarding step completes and the provider override applies.
fn complete_for_active_attempt(
    sign_in_state: &Arc<RwLock<SignInState>>,
    request_frame: &FrameRequester,
    attempt_id: &str,
) {
    let mut guard = sign_in_state.write().unwrap();
    if !attempt_matches(&guard, attempt_id) {
        return;
    }
    *guard = SignInState::HannahMontanaConfigured;
    drop(guard);
    request_frame.schedule_frame();
}

fn finish_with_error(
    sign_in_state: &Arc<RwLock<SignInState>>,
    request_frame: &FrameRequester,
    attempt_id: &str,
    message: String,
) {
    let mut guard = sign_in_state.write().unwrap();
    let SignInState::HannahMontanaSetup(state) = &mut *guard else {
        return;
    };
    if state.attempt_id != attempt_id {
        return;
    }
    state.needs_user_action = None;
    state.error = Some(message);
    state.remedy = None; // front-end-internal error: its message is self-contained.
    drop(guard);
    request_frame.schedule_frame();
}

pub(super) fn render_hannah_montana_setup(
    widget: &AuthModeWidget,
    area: Rect,
    buf: &mut Buffer,
    state: &HannahMontanaSetupState,
) {
    let mut lines: Vec<Line> = vec![
        Line::from(vec![
            "  ".into(),
            "Setting up cyrus".bold(),
        ]),
        "".into(),
    ];

    for step in &state.steps {
        let (marker, style) = match step.status {
            StepStatus::Done => ("✓", Style::default().fg(Color::Green)),
            StepStatus::Running => ("⏳", Style::default().fg(Color::Cyan)),
            StepStatus::Pending => (" ", Style::default().add_modifier(ratatui::style::Modifier::DIM)),
        };
        let mut text = format!("{marker} {label}", label = step.label);
        if let Some(detail) = &step.detail
            && !detail.is_empty()
        {
            text.push_str(&format!(" — {detail}"));
        }
        lines.push(Line::from(format!("  {text}")).style(style));
    }

    if let Some(instruction) = &state.needs_user_action {
        lines.push("".into());
        lines.push(Line::from(vec![
            "  ".into(),
            "Action needed: ".fg(Color::Yellow).bold(),
            instruction.clone().fg(Color::Yellow),
        ]));
    }

    if let Some(error) = &state.error {
        lines.push("".into());
        lines.push(Line::from(format!("  {error}")).fg(Color::Red));
        if let Some(remedy) = &state.remedy
            && !remedy.is_empty()
        {
            lines.push("".into());
            lines.push(Line::from(vec!["  Try: ".dim(), remedy.clone().into()]));
        }
        lines.push("".into());
        lines.push(Line::from(vec![
            "  Press ".dim(),
            widget.confirm_binding().into(),
            " to retry".dim(),
        ]));
    }

    Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .render(area, buf);
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn parse(line: &str) -> SetupEvent {
        serde_json::from_str(line).expect("event should parse")
    }

    #[test]
    fn tunnel_args_emit_expected_flag_vectors() {
        assert_eq!(tunnel_args(&TunnelArg::Quick), vec!["--tunnel", "quick"]);
        assert_eq!(tunnel_args(&TunnelArg::Named), vec!["--tunnel", "named"]);
        assert_eq!(
            tunnel_args(&TunnelArg::Ngrok("my.ngrok-free.app".to_string())),
            vec!["--tunnel", "ngrok", "--ngrok-domain", "my.ngrok-free.app"]
        );
    }

    #[test]
    fn parses_each_event_in_the_contract() {
        assert_eq!(
            parse(r#"{"event":"step_started","step":"chrome","label":"Connecting Chrome"}"#),
            SetupEvent::StepStarted {
                step: "chrome".to_string(),
                label: Some("Connecting Chrome".to_string()),
            }
        );
        assert_eq!(
            parse(r#"{"event":"step_done","step":"chrome","detail":"reusing running Chrome"}"#),
            SetupEvent::StepDone {
                step: "chrome".to_string(),
                detail: Some("reusing running Chrome".to_string()),
            }
        );
        assert_eq!(
            parse(
                r#"{"event":"needs_user_action","step":"connector","instruction":"log in to ChatGPT"}"#
            ),
            SetupEvent::NeedsUserAction {
                step: "connector".to_string(),
                instruction: "log in to ChatGPT".to_string(),
            }
        );
        assert_eq!(
            parse(r#"{"event":"user_action_resolved","step":"connector"}"#),
            SetupEvent::UserActionResolved {
                step: "connector".to_string(),
            }
        );
        assert_eq!(
            parse(r#"{"event":"error","message":"boom"}"#),
            SetupEvent::Error {
                message: "boom".to_string(),
                remedy: None,
            }
        );
        assert_eq!(
            parse(r#"{"event":"error","message":"boom","remedy":"free the port"}"#),
            SetupEvent::Error {
                message: "boom".to_string(),
                remedy: Some("free the port".to_string()),
            }
        );
    }

    #[test]
    fn reduces_full_sequence_to_completed() {
        let lines = [
            r#"{"event":"step_started","step":"secrets","label":"Loading secrets"}"#,
            r#"{"event":"step_done","step":"secrets","detail":"loaded"}"#,
            r#"{"event":"step_started","step":"chrome","label":"Connecting Chrome"}"#,
            r#"{"event":"needs_user_action","step":"chrome","instruction":"log in to ChatGPT"}"#,
            r#"{"event":"user_action_resolved","step":"chrome"}"#,
            r#"{"event":"step_done","step":"chrome","detail":"reusing running Chrome"}"#,
            r#"{"event":"step_started","step":"tunnel","label":"Opening tunnel"}"#,
            r#"{"event":"step_done","step":"tunnel","detail":"up"}"#,
            r#"{"event":"step_started","step":"stack","label":"Starting stack"}"#,
            r#"{"event":"step_done","step":"stack","detail":"running"}"#,
            r#"{"event":"step_started","step":"connector","label":"Wiring connector"}"#,
            r#"{"event":"step_done","step":"connector","detail":"wired"}"#,
            r#"{"event":"step_started","step":"codex_config","label":"Writing config"}"#,
            r#"{"event":"step_done","step":"codex_config","detail":"written"}"#,
            r#"{"event":"done","public_url":"https://x","shim_base_url":"https://y","connector_id":"c","link_id":"l","tool_count":34,"fully_reused":false}"#,
        ];

        let mut state = HannahMontanaSetupState::new("attempt-1".to_string());
        let mut outcome = ReduceOutcome::Continue;
        for line in lines {
            outcome = reduce(&mut state, parse(line));
        }

        assert_eq!(outcome, ReduceOutcome::Completed);
        assert_eq!(state.error, None);
        assert_eq!(state.needs_user_action, None);
        assert!(
            state
                .steps
                .iter()
                .all(|step| step.status == StepStatus::Done),
            "every step should be marked done"
        );
        let chrome = state
            .steps
            .iter()
            .find(|step| step.key == "chrome")
            .expect("chrome step present");
        assert_eq!(chrome.detail.as_deref(), Some("reusing running Chrome"));
    }

    #[test]
    fn needs_user_action_is_tracked_then_cleared() {
        let mut state = HannahMontanaSetupState::new("attempt-1".to_string());

        reduce(
            &mut state,
            parse(r#"{"event":"step_started","step":"chrome","label":"Connecting Chrome"}"#),
        );
        reduce(
            &mut state,
            parse(
                r#"{"event":"needs_user_action","step":"chrome","instruction":"log in to ChatGPT"}"#,
            ),
        );
        assert_eq!(
            state.needs_user_action.as_deref(),
            Some("log in to ChatGPT")
        );

        reduce(
            &mut state,
            parse(r#"{"event":"user_action_resolved","step":"chrome"}"#),
        );
        assert_eq!(state.needs_user_action, None);
    }

    #[test]
    fn error_event_sets_failed_terminal_state() {
        let mut state = HannahMontanaSetupState::new("attempt-1".to_string());
        reduce(
            &mut state,
            parse(r#"{"event":"step_started","step":"secrets","label":"Loading secrets"}"#),
        );

        let outcome = reduce(&mut state, parse(r#"{"event":"error","message":"chrome not found"}"#));

        assert_eq!(outcome, ReduceOutcome::Failed);
        assert_eq!(state.error.as_deref(), Some("chrome not found"));
        assert_eq!(state.needs_user_action, None);
    }
}
