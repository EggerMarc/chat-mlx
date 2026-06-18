//! Structured-output helpers: turn a JSON Schema into a prompt instruction, and
//! extract a JSON value back out of model output.

use serde_json::Value;

/// System-prompt instruction asking the model to emit JSON matching `schema`.
pub fn instruction(schema: &Value) -> String {
    format!(
        "You must respond with a single JSON value conforming to the following JSON Schema. \
         Output only the JSON value — no prose, no explanation, no markdown code fences.\n\n\
         JSON Schema:\n{schema}"
    )
}

/// Extract the first complete JSON value from model output. Tolerates leading
/// prose and ```json fences. Returns `None` if nothing parses.
pub fn extract(text: &str) -> Option<Value> {
    let trimmed = strip_fences(text.trim());

    // Fast path: the whole thing is already JSON.
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        return Some(v);
    }

    // Otherwise find the first balanced {...} or [...] and parse it.
    let bytes = trimmed.as_bytes();
    let start = bytes.iter().position(|&b| b == b'{' || b == b'[')?;
    let open = bytes[start];
    let close = if open == b'{' { b'}' } else { b']' };

    let mut depth = 0i32;
    let mut in_str = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_str {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            c if c == open => depth += 1,
            c if c == close => {
                depth -= 1;
                if depth == 0 {
                    return serde_json::from_str::<Value>(&trimmed[start..=i]).ok();
                }
            }
            _ => {}
        }
    }
    None
}

fn strip_fences(s: &str) -> &str {
    let s = s.trim();
    let Some(rest) = s.strip_prefix("```") else {
        return s;
    };
    // Drop an optional language tag on the opening fence line.
    let rest = rest.split_once('\n').map(|(_, body)| body).unwrap_or(rest);
    rest.trim().strip_suffix("```").unwrap_or(rest).trim()
}
