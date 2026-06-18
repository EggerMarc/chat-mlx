use chat_core::types::messages::Messages;
use chat_core::types::messages::content::RoleEnum;
use chat_core::types::messages::parts::PartEnum;
use chat_core::types::options::ChatOptions;

use crate::api::types::error::unsupported;
use crate::engine::sampler::SampleOpts;
use crate::engine::template::{Turn, chatml};
use chat_core::error::ChatFailure;

const DEFAULT_MAX_TOKENS: usize = 512;

/// A request lowered from chat-core types into what the engine consumes.
pub struct Prepared {
    pub prompt: String,
    pub sampler: SampleOpts,
    pub max_tokens: usize,
}

/// Lower chat-core `Messages` + `ChatOptions` into a prompt string and sampling
/// config. Tool declarations and structured output are rejected for now (they
/// land in later phases), mirroring how `chat-mistralrs` rejects unsupported
/// parts.
pub fn from_core(
    messages: &Messages,
    options: Option<&ChatOptions>,
    structured_output: Option<&schemars::Schema>,
    tools_present: bool,
) -> Result<Prepared, ChatFailure> {
    if tools_present {
        return Err(unsupported("tool declarations"));
    }
    if structured_output.is_some() {
        return Err(unsupported("structured outputs"));
    }

    let mut turns = Vec::with_capacity(messages.0.len());
    for content in &messages.0 {
        turns.push(Turn {
            role: map_role(&content.role),
            content: flatten_text(&content.parts.0)?,
        });
    }
    let prompt = chatml(&turns);

    let (sampler, max_tokens) = sampler_from_options(options);
    Ok(Prepared {
        prompt,
        sampler,
        max_tokens,
    })
}

fn map_role(role: &RoleEnum) -> &'static str {
    match role {
        RoleEnum::User => "user",
        RoleEnum::System => "system",
        RoleEnum::Model => "assistant",
    }
}

/// Collect all `Text` parts into one newline-joined string. Other part types
/// are not supported on the input path yet.
fn flatten_text(parts: &[PartEnum]) -> Result<String, ChatFailure> {
    let mut buf = String::new();
    for part in parts {
        match part {
            PartEnum::Text(t) => {
                if !buf.is_empty() {
                    buf.push('\n');
                }
                buf.push_str(t.as_str());
            }
            PartEnum::Reasoning(_) => return Err(unsupported("reasoning parts in input")),
            PartEnum::Tool(_) => return Err(unsupported("tool parts")),
            PartEnum::Structured(_) => return Err(unsupported("structured parts in input")),
            PartEnum::File(_) => return Err(unsupported("file parts")),
            PartEnum::Embeddings(_) => return Err(unsupported("embedding parts in input")),
        }
    }
    Ok(buf)
}

fn sampler_from_options(options: Option<&ChatOptions>) -> (SampleOpts, usize) {
    let mut sampler = SampleOpts {
        temp: 0.0,
        top_k: None,
        top_p: None,
    };
    let mut max_tokens = DEFAULT_MAX_TOKENS;

    if let Some(o) = options {
        if let Some(t) = o.temperature {
            sampler.temp = t;
        }
        if let Some(p) = o.top_p {
            sampler.top_p = Some(p);
        }
        if let Some(m) = o.max_tokens {
            max_tokens = m as usize;
        }
        if let Some(k) = o.metadata.get("top_k").and_then(|v| v.as_u64()) {
            sampler.top_k = Some(k as usize);
        }
    }

    (sampler, max_tokens)
}
