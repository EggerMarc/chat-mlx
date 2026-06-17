pub struct Turn {
    pub role: &'static str,
    pub content: String,
}

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
