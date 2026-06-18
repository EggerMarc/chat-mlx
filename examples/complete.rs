//! Phase 1 end-to-end check: drive `MlxClient` through chat-core's `ChatBuilder`.
//!
//! Downloads the model on first use (default Qwen/Qwen3-0.6B). Run with:
//!   cargo run --release --example complete -- [hf-repo-id] [prompt]

use chat_core::builder::ChatBuilder;
use chat_core::parts;
use chat_core::types::messages::{self, content};
use chat_core::types::response::ChatOutcome;

use chat_mlx::MlxBuilder;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Qwen/Qwen3-0.6B".to_string());
    let prompt = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "Explain rotary position embeddings in one sentence.".to_string());

    eprintln!("[example] loading {model} …");
    let client = MlxBuilder::new().with_model(model).build()?;

    let mut chat = ChatBuilder::new().with_model(client).build();

    let mut msgs = messages::Messages::default();
    msgs.push(content::from_system(parts!["You are concise."]));
    msgs.push(content::from_user(parts![prompt]));

    match chat.complete(&mut msgs).await? {
        ChatOutcome::Complete(resp) => {
            let text = resp
                .content
                .parts
                .text_response()
                .map(|t| t.as_str())
                .unwrap_or("<no text part>");
            println!("\n--- response ---\n{text}");
            if let Some(md) = resp.metadata {
                println!("\n[usage] {:?}", md.usage);
            }
        }
        ChatOutcome::Paused { reason } => {
            println!("[paused] {reason:?}");
        }
    }

    Ok(())
}
