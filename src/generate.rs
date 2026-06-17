//! The autoregressive decode loop: prefill the prompt, then sample one token
//! at a time, feeding the KV cache forward. Backend-agnostic in spirit — it
//! only talks to `Model::forward` and the sampler.

use anyhow::Result;
use mlx_rs::{
    ops::indexing::{IndexOp, NewAxis},
    transforms::eval,
    Array,
};

use crate::{model::Model, sampler::sample};

/// Run generation, invoking `on_token` for each newly produced token id.
/// Stops at `max_tokens` or when an EOS id is produced. Returns the full list
/// of generated token ids (excluding the stopping EOS).
pub fn generate<F: FnMut(u32)>(
    model: &mut Model,
    prompt_ids: &[u32],
    max_tokens: usize,
    temp: f32,
    eos: &[u32],
    mut on_token: F,
) -> Result<Vec<u32>> {
    // [1, L]
    let prompt = Array::from(prompt_ids).index(NewAxis);

    // ---- Prefill ----
    let empty: Vec<Option<(Array, Array)>> = Vec::new();
    let (logits, mut cache) = model.forward(&prompt, &empty)?;
    // logits for the last position only: [1, vocab]
    let last = logits.index((.., -1, ..));
    let mut y = sample(&last, temp)?;
    eval(&y)?;

    let mut out = Vec::with_capacity(max_tokens);
    for _ in 0..max_tokens {
        let id = y.item::<u32>();
        if eos.contains(&id) {
            break;
        }
        on_token(id);
        out.push(id);

        // ---- One decode step ----
        let next = y.index((.., NewAxis)); // [1, 1]
        let (logits, new_cache) = model.forward(&next, &cache)?;
        cache = new_cache;
        let logits = logits.squeeze_axes(&[1])?; // [1, vocab]
        y = sample(&logits, temp)?;
        eval(&y)?;
    }

    Ok(out)
}
