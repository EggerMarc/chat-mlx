//! Interactive, streaming chat REPL driving `MlxClient` through chat-core.
//!
//! Compose the session on the command line, then talk to it:
//!   cargo run --release --example chat -- \
//!     --model Qwen/Qwen3-0.6B --system "You are a pirate." --temp 0.7 --top-k 40
//!
//! Streaming is on by default; disable with `--no-default-features`.
//!
//! Output streams live. Reasoning (`<think>…</think>`) is rendered in gray and
//! the tags are stripped. `/quit` (or Ctrl-D) exits. Conversation history is
//! kept across turns.

use std::io::{self, BufRead, Write};

use chat_core::builder::ChatBuilder;
use chat_core::parts;
use chat_core::types::messages::{self, content};
use chat_core::types::options::ChatOptions;
use chat_core::types::response::StreamEvent;
use clap::Parser;
use futures::StreamExt;

use chat_mlx::MlxBuilder;

const GRAY: &str = "\x1b[90m";
const RESET: &str = "\x1b[0m";

#[derive(Parser)]
#[command(about = "Interactive streaming chat REPL over the chat-mlx provider")]
struct Cli {
    /// HF repo id of the model.
    #[clap(long, default_value = "Qwen/Qwen3-0.6B")]
    model: String,

    /// Optional system prompt.
    #[clap(long)]
    system: Option<String>,

    /// Sampling temperature (0.0 = greedy).
    #[clap(long, default_value = "0.0")]
    temp: f32,

    /// Top-k sampling cutoff.
    #[clap(long)]
    top_k: Option<usize>,

    /// Top-p (nucleus) sampling cutoff.
    #[clap(long)]
    top_p: Option<f32>,

    /// Max tokens to generate per turn.
    #[clap(long, default_value = "512")]
    max_tokens: u32,

    /// Max tokens retained in the rotating KV cache (0 = unbounded).
    #[clap(long, default_value = "4096")]
    max_context: i32,

    /// Runtime-quantize the loaded fp weights to 4-bit.
    #[clap(long)]
    quantize: bool,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();

    eprintln!("[chat] loading {} …", cli.model);
    let client = MlxBuilder::new()
        .with_model(cli.model.clone())
        .with_quantize(cli.quantize)
        .with_max_context(cli.max_context)
        .build()?;

    let mut options = ChatOptions::default();
    options.temperature = Some(cli.temp);
    options.max_tokens = Some(cli.max_tokens);
    options.top_p = cli.top_p;
    if let Some(k) = cli.top_k {
        options
            .metadata
            .insert("top_k".to_string(), serde_json::json!(k));
    }

    let mut chat = ChatBuilder::new()
        .with_model(client)
        .with_options(options)
        .build();

    let mut msgs = messages::Messages::default();
    if let Some(sys) = &cli.system {
        msgs.push(content::from_system(parts![sys.as_str()]));
    }

    eprintln!(
        "[chat] model={} temp={} top_k={:?} top_p={:?} max_tokens={}{}",
        cli.model,
        cli.temp,
        cli.top_k,
        cli.top_p,
        cli.max_tokens,
        if cli.quantize { " quantized" } else { "" },
    );
    eprintln!("[chat] reasoning shown in gray; type a message, /quit or Ctrl-D to exit.\n");

    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    loop {
        print!("you> ");
        io::stdout().flush()?;

        let Some(line) = lines.next() else {
            break; // EOF (Ctrl-D)
        };
        let line = line?;
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        if text == "/quit" || text == "/exit" {
            break;
        }

        msgs.push(content::from_user(parts![text]));

        print!("bot> ");
        io::stdout().flush()?;

        let mut stream = chat.stream(&mut msgs).await?;
        while let Some(ev) = stream.next().await {
            match ev {
                Ok(StreamEvent::ReasoningChunk(s)) => {
                    print!("{GRAY}{s}{RESET}");
                    io::stdout().flush()?;
                }
                Ok(StreamEvent::TextChunk(s)) => {
                    print!("{s}");
                    io::stdout().flush()?;
                }
                Ok(StreamEvent::Done(_)) => break,
                Ok(_) => {}
                Err(err) => {
                    eprintln!("\n[error] {}", err.err);
                    break;
                }
            }
        }
        println!("\n");
    }

    eprintln!("\n[chat] bye.");
    Ok(())
}
