//! ChatML prompt formatting. Pure string work — no MLX, no tokenizer here.
//!
//! MiniCPM5 uses the ChatML layout:
//! `<|im_start|>{role}\n{content}<|im_end|>\n`, ending with an open
//! assistant turn so the model continues from there.

pub struct Turn {
    pub role: &'static str,
    pub content: String,
}

/// Build a ChatML prompt string from a sequence of turns, leaving the
/// assistant turn open for generation.
pub fn chatml(turns: &[Turn]) -> String {
    let mut s = String::new();
    for t in turns {
        s.push_str("<|im_start|>");
        s.push_str(t.role);
        s.push('\n');
        s.push_str(&t.content);
        s.push_str("<|im_end|>\n");
    }
    s.push_str("<|im_start|>assistant\n");
    s
}

/// Convenience: optional system prompt + a single user message.
pub fn simple(system: Option<&str>, user: &str) -> String {
    let mut turns = Vec::new();
    if let Some(sys) = system {
        turns.push(Turn {
            role: "system",
            content: sys.to_string(),
        });
    }
    turns.push(Turn {
        role: "user",
        content: user.to_string(),
    });
    chatml(&turns)
}
