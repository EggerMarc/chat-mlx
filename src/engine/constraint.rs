use mlx_rs::{Array, error::Exception};

/// A per-step logit transform for constrained decoding: `mask` restricts which
/// tokens may be sampled next, and `accept` advances internal state with the
/// token that was chosen. Implemented by `parsers::json::JsonConstraint`.
pub trait LogitMask {
    /// Return `logits` with disallowed tokens pushed to `-inf`.
    fn mask(&self, logits: &Array) -> Result<Array, Exception>;

    /// Record the token that was actually sampled.
    fn accept(&mut self, token: u32);
}
