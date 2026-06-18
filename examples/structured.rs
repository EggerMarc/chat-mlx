//! Structured output in *streaming* mode, showcasing both modes side by side.
//!
//!   cargo run --release --example structured -- [hf-repo-id]
//!
//! For each of `Prompt` and `Constrained`, the JSON is streamed live (reasoning
//! in gray) and then deserialized into a typed `Person`. Constrained mode masks
//! logits so only well-formed JSON can be sampled; prompt mode instructs via the
//! schema and parses the result.

use std::io::{self, Write};

use chat_core::parts;
use chat_core::types::messages::{self, content};
use chat_core::types::options::ChatOptions;
use chat_core::types::response::StreamEvent;
use futures::StreamExt;
use schemars::JsonSchema;
use serde::Deserialize;

use chat_mlx::{MlxBuilder, StructuredMode};

const GRAY: &str = "\x1b[90m";
const CYAN: &str = "\x1b[36m";
const RESET: &str = "\x1b[0m";

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)] // fields are read via the Debug print
struct Person {
    name: String,
    age: u32,
    hobbies: Vec<String>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Qwen/Qwen2.5-1.5B-Instruct".to_string());

    eprintln!("[structured] loading {model} …");
    let client = MlxBuilder::new().with_model(model).build()?;

    let schema = schemars::schema_for!(Person);
    let mut options = ChatOptions::default();
    options.max_tokens = Some(256);

    let prompt = "Ada Lovelace is 36 and enjoys mathematics, chess, and poetry.";

    for mode in [StructuredMode::Prompt, StructuredMode::Constrained] {
        println!("\n{CYAN}===== mode: {mode:?} ====={RESET}");
        let mut msgs = messages::Messages::default();
        msgs.push(content::from_user(parts![prompt]));

        print!("stream: ");
        io::stdout().flush()?;

        let mut stream = client.stream_structured(&msgs, Some(&options), &schema, mode)?;
        while let Some(ev) = stream.next().await {
            match ev? {
                StreamEvent::ReasoningChunk(s) => print!("{GRAY}{s}{RESET}"),
                StreamEvent::TextChunk(s) => print!("{s}"),
                StreamEvent::Structured(v) => {
                    let person: Person = serde_json::from_value(v)?;
                    println!("\nparsed:  {person:?}");
                }
                StreamEvent::Done(_) => break,
                _ => {}
            }
            io::stdout().flush()?;
        }
    }

    Ok(())
}
