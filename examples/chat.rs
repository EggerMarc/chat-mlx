//! Interactive, **bidirectional** streaming chat over the chat-mlx provider.
//!
//!   cargo run --release --example chat -- \
//!     --model Qwen/Qwen3-0.6B --system "You are a pirate." --temp 0.7 --top-k 40
//!
//! The model's reply streams in the scrolling area; your input box is pinned to
//! the bottom row. Keep typing while it generates — pressing Enter sends a new
//! message that **interrupts** the in-flight generation and restarts on your new
//! turn (chat-rs `InputStreamed` + our cancellable decode). Reasoning
//! (`<think>`) shows in gray; a `get_weather` tool round-trip shows inline.
//! `/quit`, Esc, or Ctrl-C exits. Needs a real terminal (raw mode).

use std::io::{Stdout, Write, stdout};

use chat_core::builder::ChatBuilder;
use chat_core::parts;
use chat_core::types::messages::{self, content};
use chat_core::types::options::ChatOptions;
use chat_core::types::response::StreamEvent;
use clap::Parser;
use crossterm::event::{Event, EventStream, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, size};
use futures::{FutureExt, StreamExt, select};
use tools_rs::{collect_tools, tool};

use chat_mlx::{MlxBuilder, Quantize};

const GRAY: &str = "\x1b[90m";
const CYAN: &str = "\x1b[36m";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

/// Look up the current weather for a city.
#[tool]
async fn get_weather(city: String) -> String {
    format!("It is 22°C and sunny in {city}.")
}

#[derive(Parser)]
#[command(about = "Bidirectional streaming chat REPL (fixed bottom input)")]
struct Cli {
    #[clap(long, default_value = "Qwen/Qwen3-0.6B")]
    model: String,
    #[clap(long)]
    system: Option<String>,
    #[clap(long, default_value = "0.7")]
    temp: f32,
    #[clap(long)]
    top_k: Option<usize>,
    #[clap(long, default_value = "512")]
    max_tokens: u32,
    /// Quantization bit width: 2, 3, 4, or 8 (anything else = no quantization).
    #[clap(long, default_value = "4")]
    quantize: i32,
}

enum Key {
    Submit(String),
    Edit,
    Stop,
    Quit,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();

    eprintln!("[chat] loading {} …", cli.model);
    let mut builder = MlxBuilder::new().with_model(cli.model.clone());
    if let Some(q) = Quantize::from_bits(cli.quantize) {
        builder = builder.with_quantize(q);
    }
    let client = builder.build()?;

    let mut options = ChatOptions::default();
    options.temperature = Some(cli.temp);
    options.max_tokens = Some(cli.max_tokens);
    if let Some(k) = cli.top_k {
        options
            .metadata
            .insert("top_k".to_string(), serde_json::json!(k));
    }

    let mut chat = ChatBuilder::new()
        .with_model(client)
        .with_options(options)
        .with_tools(collect_tools())
        .with_max_steps(6)
        .with_input_stream()
        .build();

    let mut msgs = messages::Messages::default();
    if let Some(sys) = &cli.system {
        msgs.push(content::from_system(parts![sys.as_str()]));
    }

    let (cols, rows) = size()?;
    enable_raw_mode()?;
    let mut out = stdout();
    // Reserve three bottom rows: a separator, the input field, a separator.
    // Output scrolls in rows 1..=region_bottom; anchor the output cursor to the
    // bottom of that region so messages stack upward like a messenger app.
    let region_bottom = rows.saturating_sub(3).max(1);
    let top_sep = rows.saturating_sub(2);
    let sep = "─".repeat(cols as usize);
    let _ = write!(
        out,
        "\x1b[2J\x1b[1;{region_bottom}r\
         \x1b[{top_sep};1H{DIM}{sep}{RESET}\
         \x1b[{rows};1H{DIM}{sep}{RESET}\
         \x1b[{region_bottom};1H\x1b7",
    );
    let mut buf = String::new();
    let _ = draw_input(&mut out, rows, &buf);

    // Run inside an async block so `?` unwinds to here and the terminal is
    // always restored.
    let result: Result<(), Box<dyn std::error::Error + Send + Sync>> = async {
        let mut events = EventStream::new();

        // One turn per outer iteration: collect a message, open a stream for it,
        // drain it, then loop. (A chat-rs `InputStreamed` stream covers a single
        // turn — interruptible mid-flight — and ends on completion.)
        'session: loop {
            // Collect the next message (idle: typing starts a new turn).
            let msg = loop {
                match events.next().await {
                    Some(Ok(Event::Key(k))) => match handle_key(k, &mut buf) {
                        Key::Submit(s) if !s.trim().is_empty() => break s,
                        Key::Submit(_) => {}
                        Key::Stop => buf.clear(), // nothing generating; clear the line
                        Key::Quit => break 'session,
                        Key::Edit => {}
                    },
                    Some(Err(_)) | None => break 'session,
                    _ => {}
                }
                draw_input(&mut out, rows, &buf)?;
            };
            print_out(&mut out, rows, &buf, &format!("{CYAN}you> {msg}{RESET}\n"))?;
            msgs.push(content::from_user(parts![msg]));

            // Stream this turn. Typing mid-generation sends a new message, which
            // interrupts and restarts the turn (chat-rs merges + restarts).
            let (input, mut output) = chat.stream(&mut msgs).await?.split();
            'turn: loop {
                select! {
                    ev = output.next().fuse() => match ev {
                        Some(Ok(StreamEvent::ReasoningChunk(s))) =>
                            print_out(&mut out, rows, &buf, &format!("{GRAY}{s}{RESET}"))?,
                        Some(Ok(StreamEvent::TextChunk(s))) =>
                            print_out(&mut out, rows, &buf, &s)?,
                        Some(Ok(StreamEvent::ToolCall(c))) =>
                            print_out(&mut out, rows, &buf, &format!("\n{CYAN}⚙ {}({}){RESET}\n", c.name, c.arguments))?,
                        Some(Ok(StreamEvent::ToolResult(r))) =>
                            print_out(&mut out, rows, &buf, &format!("{DIM}⮑ {}{RESET}\n", r.result))?,
                        Some(Ok(StreamEvent::Done(_))) => print_out(&mut out, rows, &buf, "\n")?,
                        Some(Ok(_)) => {}
                        Some(Err(e)) =>
                            print_out(&mut out, rows, &buf, &format!("{GRAY}[err] {}{RESET}\n", e.err))?,
                        None => break 'turn, // turn finished → collect the next message
                    },
                    key = events.next().fuse() => match key {
                        Some(Ok(Event::Key(k))) => match handle_key(k, &mut buf) {
                            Key::Submit(s) => {
                                if !s.trim().is_empty() {
                                    print_out(&mut out, rows, &buf, &format!("\n{CYAN}you> {s}{RESET}\n"))?;
                                    if input.send(s).is_err() {
                                        break 'turn;
                                    }
                                }
                            }
                            Key::Stop => {
                                // Interrupt this turn and go back to collecting.
                                input.cancel();
                                print_out(&mut out, rows, &buf, &format!("{DIM} [stopped]{RESET}\n"))?;
                                break 'turn;
                            }
                            Key::Quit => {
                                input.cancel();
                                break 'session;
                            }
                            Key::Edit => draw_input(&mut out, rows, &buf)?,
                        },
                        Some(Err(_)) | None => break 'session,
                        _ => {}
                    },
                }
            }
        }
        Ok(())
    }
    .await;

    let _ = write!(out, "\x1b[r\x1b[{};1H\r\n", rows);
    let _ = out.flush();
    let _ = disable_raw_mode();
    result
}

fn handle_key(k: KeyEvent, buf: &mut String) -> Key {
    match (k.code, k.modifiers) {
        (KeyCode::Char('c'), KeyModifiers::CONTROL) => Key::Quit,
        (KeyCode::Esc, _) => Key::Stop,
        (KeyCode::Enter, _) => {
            let s = std::mem::take(buf);
            if s.trim() == "/quit" {
                Key::Quit
            } else {
                Key::Submit(s)
            }
        }
        (KeyCode::Backspace, _) => {
            buf.pop();
            Key::Edit
        }
        (KeyCode::Char(c), _) => {
            buf.push(c);
            Key::Edit
        }
        _ => Key::Edit,
    }
}

/// Print into the scrolling output area, then redraw the fixed input line. The
/// output cursor lives in the terminal save slot (`\x1b7`/`\x1b8`) so it resumes
/// where the last chunk left off, untouched by the input redraw. `\n` in `text`
/// becomes `\r\n` (raw mode).
fn print_out(out: &mut Stdout, rows: u16, buf: &str, text: &str) -> std::io::Result<()> {
    write!(out, "\x1b8")?;
    for (i, line) in text.split('\n').enumerate() {
        if i > 0 {
            write!(out, "\r\n")?;
        }
        write!(out, "{line}")?;
    }
    write!(out, "\x1b7")?;
    draw_input(out, rows, buf)
}

fn draw_input(out: &mut Stdout, rows: u16, buf: &str) -> std::io::Result<()> {
    // Input field sits on the row between the two separators.
    write!(out, "\x1b[{};1H\x1b[2K> {buf}", rows.saturating_sub(1))?;
    out.flush()
}
