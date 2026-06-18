//! Interactive, streaming chat REPL over the chat-mlx provider: CLI-composed
//! session config, multi-turn history, live streaming, gray-rendered `<think>`
//! reasoning, and tool calling. (See `examples/structured.rs` for structured
//! output.)
//!
//!   cargo run --release --example chat -- \
//!     --model Qwen/Qwen3-0.6B --system "You are a pirate." --temp 0.7 --top-k 40
//!
//! A demo `get_weather` tool is registered; ask about the weather to see the
//! round-trip. Type a message; `/quit` (or Ctrl-D) exits. Streaming is on by
//! default (disable with `--no-default-features`).

use std::io::{self, BufRead, Write};

use chat_core::builder::ChatBuilder;
use chat_core::parts;
use chat_core::types::messages::{self, content};
use chat_core::types::options::ChatOptions;
use chat_core::types::response::StreamEvent;
use clap::Parser;
use futures::StreamExt;
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
    let mut builder = MlxBuilder::new()
        .with_model(cli.model.clone())
        .with_max_context(cli.max_context);
    if cli.quantize {
        builder = builder.with_quantize(Quantize::Q4);
    }
    let client = builder.build()?;

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
        .with_tools(collect_tools())
        .with_max_steps(6)
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
    eprintln!("[chat] tool: get_weather(city). reasoning in gray. /quit or Ctrl-D to exit.\n");

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
                Ok(StreamEvent::ReasoningChunk(s)) => print_flush(&format!("{GRAY}{s}{RESET}"))?,
                Ok(StreamEvent::TextChunk(s)) => print_flush(&s)?,
                Ok(StreamEvent::ToolCall(call)) => {
                    print_flush(&format!("\n{CYAN}⚙ {}({}){RESET}\n", call.name, call.arguments))?
                }
                Ok(StreamEvent::ToolResult(resp)) => {
                    print_flush(&format!("{DIM}⮑ {}{RESET}\n", resp.result))?
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

fn print_flush(s: &str) -> io::Result<()> {
    print!("{s}");
    io::stdout().flush()
}
