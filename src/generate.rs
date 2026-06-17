use anyhow::Result;
use mlx_rs::{
    Array,
    ops::indexing::{IndexOp, NewAxis},
    transforms::eval,
};

use crate::{model::Model, sampler::sample};

pub fn generate<F: FnMut(u32)>(
    model: &mut Model,
    prompt_ids: &[u32],
    max_tokens: usize,
    temp: f32,
    eos: &[u32],
    mut on_token: F,
) -> Result<Vec<u32>> {
    let prompt = Array::from(prompt_ids).index(NewAxis);

    let empty: Vec<Option<(Array, Array)>> = Vec::new();
    let (logits, mut cache) = model.forward(&prompt, &empty)?;

    let last = logits.index((.., -1, ..));
    let mut y = sample(&last, temp)?;

    eval([&y])?;

    let mut out = Vec::with_capacity(max_tokens);
    for _ in 0..max_tokens {
        let id = y.item::<u32>();
        if eos.contains(&id) {
            break;
        }
        on_token(id);
        out.push(id);

        let next = y.index((.., NewAxis)); // [1, 1]
        let (logits, new_cache) = model.forward(&next, &cache)?;
        cache = new_cache;
        let logits = logits.squeeze_axes(&[1])?; // [1, vocab]
        y = sample(&logits, temp)?;
        eval([&y])?;
    }

    Ok(out)
}
