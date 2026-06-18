use chat_core::types::messages::content::{CompleteReasonEnum, Content, RoleEnum};
use chat_core::types::messages::parts::{PartEnum, Parts};
use chat_core::types::messages::reasoning::Reasoning;
use chat_core::types::messages::text::Text;
use chat_core::types::metadata::Metadata;
use chat_core::types::metadata::usage::Usage;
use chat_core::types::response::ChatResponse;

use crate::parsers::reasoning;

/// Build a chat-core [`ChatResponse`] from the full decoded assistant text,
/// splitting any `<think>…</think>` span into a `Reasoning` part.
pub fn into_core(
    model_id: &str,
    text: String,
    input_tokens: usize,
    output_tokens: usize,
    max_tokens: usize,
) -> ChatResponse {
    let (reasoning, body) = reasoning::split(&text);
    into_core_parts(
        model_id,
        reasoning,
        body,
        input_tokens,
        output_tokens,
        max_tokens,
    )
}

/// Build a [`ChatResponse`] from already-split reasoning and answer text. Used
/// by both the completion and streaming paths so they yield equivalent content.
pub fn into_core_parts(
    model_id: &str,
    reasoning_text: String,
    text: String,
    input_tokens: usize,
    output_tokens: usize,
    max_tokens: usize,
) -> ChatResponse {
    let complete_reason = if output_tokens >= max_tokens {
        CompleteReasonEnum::MaxTokens
    } else {
        CompleteReasonEnum::Stop
    };

    let mut parts = Vec::new();
    if !reasoning_text.trim().is_empty() {
        parts.push(PartEnum::Reasoning(Reasoning::new(reasoning_text)));
    }
    parts.push(PartEnum::Text(Text::new(text)));

    let content = Content {
        role: RoleEnum::Model,
        parts: Parts(parts),
        complete_reason,
    };

    let metadata = Metadata {
        model_slug: Some(model_id.to_string()),
        usage: Usage {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
        },
        ..Default::default()
    };

    ChatResponse {
        metadata: Some(metadata),
        content,
    }
}
