//! Incremental splitter for `<think>…</think>` reasoning spans.
//!
//! Models like Qwen3 emit their chain-of-thought wrapped in literal `<think>` /
//! `</think>` text tags. This splits a streamed token-piece sequence into
//! reasoning vs. answer text, stripping the tags, and copes with a tag landing
//! across two token boundaries (e.g. `<th` then `ink>`).

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

/// A classified slice of decoded output.
pub enum Chunk {
    Text(String),
    Reasoning(String),
}

#[derive(Default)]
pub struct ReasoningSplitter {
    in_think: bool,
    pending: String,
    /// Full reasoning text seen so far (tags stripped).
    pub reasoning: String,
    /// Full answer text seen so far (tags stripped).
    pub text: String,
}

impl ReasoningSplitter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one decoded piece, returning any chunks that can be emitted now.
    /// A trailing fragment that might be the start of a tag is held back until
    /// the next `push` or `flush`.
    pub fn push(&mut self, piece: &str) -> Vec<Chunk> {
        self.pending.push_str(piece);
        let mut out = Vec::new();

        loop {
            let (marker, emit_reasoning) = if self.in_think {
                (CLOSE, true)
            } else {
                (OPEN, false)
            };

            if let Some(i) = self.pending.find(marker) {
                let before: String = self.pending.drain(..i).collect();
                self.pending.drain(..marker.len());
                self.in_think = !self.in_think;
                if !before.is_empty() {
                    out.push(self.record(before, emit_reasoning));
                }
                continue;
            }

            // No complete marker. Emit everything except a suffix that could be
            // the start of one.
            let keep = super::partial_suffix_len(&self.pending, marker);
            let emit_len = self.pending.len() - keep;
            if emit_len > 0 {
                let s: String = self.pending.drain(..emit_len).collect();
                out.push(self.record(s, emit_reasoning));
            }
            break;
        }

        out
    }

    /// Emit any held-back text once generation is finished.
    pub fn flush(&mut self) -> Vec<Chunk> {
        if self.pending.is_empty() {
            return Vec::new();
        }
        let s = std::mem::take(&mut self.pending);
        vec![self.record(s, self.in_think)]
    }

    fn record(&mut self, s: String, reasoning: bool) -> Chunk {
        if reasoning {
            self.reasoning.push_str(&s);
            Chunk::Reasoning(s)
        } else {
            self.text.push_str(&s);
            Chunk::Text(s)
        }
    }
}

/// One-shot split of a complete string into `(reasoning, answer)`.
pub fn split(text: &str) -> (String, String) {
    let mut s = ReasoningSplitter::new();
    let _ = s.push(text);
    let _ = s.flush();
    (s.reasoning, s.text)
}
