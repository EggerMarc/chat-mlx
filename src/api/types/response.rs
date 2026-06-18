use chat_core::types::messages::content::{CompleteReasonEnum, Content, RoleEnum};
use chat_core::types::messages::parts::{PartEnum, Parts};
use chat_core::types::messages::reasoning::Reasoning;
use chat_core::types::messages::text::Text;
use chat_core::types::metadata::Metadata;
use chat_core::types::metadata::usage::Usage;
use chat_core::types::response::ChatResponse;
use tools_rs::FunctionCall;

/// Assemble a [`ChatResponse`] from the split-out pieces of one generation.
/// Used by both the completion and streaming paths so they stay equivalent.
///
/// Part order is reasoning → text → tool calls. When there are tool calls the
/// finish reason is `ToolCall`, which is what drives the chat loop to execute
/// them and call again.
pub fn build(
    model_id: &str,
    reasoning_text: String,
    text: String,
    calls: Vec<FunctionCall>,
    input_tokens: usize,
    output_tokens: usize,
    max_tokens: usize,
) -> ChatResponse {
    let complete_reason = if !calls.is_empty() {
        CompleteReasonEnum::ToolCall
    } else if output_tokens >= max_tokens {
        CompleteReasonEnum::MaxTokens
    } else {
        CompleteReasonEnum::Stop
    };

    let mut parts = Vec::new();
    if !reasoning_text.trim().is_empty() {
        parts.push(PartEnum::Reasoning(Reasoning::new(reasoning_text)));
    }
    if !text.trim().is_empty() {
        parts.push(PartEnum::Text(Text::new(text)));
    }
    for call in calls {
        parts.push(PartEnum::from_function_call(call));
    }

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
