//! Phase 2 check: tool calling through the chat-rs loop, parsed out of local
//! model output by the Hermes/Qwen `<tool_call>` format.
//!
//!   cargo run --release --example tools -- [hf-repo-id]
//!
//! Defaults to a tool-capable Qwen2.5 instruct model.

use chat_core::builder::ChatBuilder;
use chat_core::parts;
use chat_core::types::messages::{self, content};
use chat_core::types::messages::parts::PartEnum;
use chat_core::types::response::ChatOutcome;
use tools_rs::{collect_tools, tool};

use chat_mlx::MlxBuilder;

#[tool]
/// Look up the current weather for a city.
async fn get_weather(city: String) -> String {
    format!("It is 22°C and sunny in {city}.")
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Qwen/Qwen2.5-1.5B-Instruct".to_string());

    eprintln!("[tools] loading {model} …");
    let client = MlxBuilder::new().with_model(model).build()?;

    let mut chat = ChatBuilder::new()
        .with_model(client)
        .with_tools(collect_tools())
        .with_max_steps(5)
        .build();

    let mut msgs = messages::Messages::default();
    msgs.push(content::from_user(parts![
        "What's the weather in Paris? Use the tool."
    ]));

    match chat.complete(&mut msgs).await? {
        ChatOutcome::Complete(resp) => {
            println!("\n--- final answer ---");
            println!(
                "{}",
                resp.content
                    .parts
                    .text_response()
                    .map(|t| t.as_str())
                    .unwrap_or("<no text>")
            );

            println!("\n--- transcript ---");
            for content in &msgs.0 {
                for part in &content.parts.0 {
                    match part {
                        PartEnum::Text(t) => println!("[{:?}] text: {}", content.role, t.as_str()),
                        PartEnum::Tool(tool) => println!(
                            "[{:?}] tool {} -> {:?}",
                            content.role,
                            tool.call.name,
                            tool.response().map(|r| &r.result)
                        ),
                        PartEnum::Reasoning(_) => {}
                        other => println!("[{:?}] {other:?}", content.role),
                    }
                }
            }
        }
        ChatOutcome::Paused { reason } => println!("[paused] {reason:?}"),
    }

    Ok(())
}
