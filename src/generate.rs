use std::time::Instant;

use anyhow::Result;
use mlx_rs::{
    Array,
    ops::indexing::{IndexOp, NewAxis},
    transforms::eval,
};
use rand::rngs::StdRng;

use crate::{
    model::Model,
    sampler::{SampleOpts, sample},
};

pub struct GenStats {
    pub tokens: Vec<u32>,
    pub prefill_secs: f64,
    pub decode_secs: f64,
}

pub fn generate<F: FnMut(u32)>(
    model: &mut Model,
    prompt_ids: &[u32],
    max_tokens: usize,
    opts: &SampleOpts,
    rng: &mut StdRng,
    eos: &[u32],
    mut on_token: F,
) -> Result<GenStats> {
    let prompt = Array::from(prompt_ids).index(NewAxis);

    let t_prefill = Instant::now();
    let empty: Vec<Option<(Array, Array)>> = Vec::new();
    let (logits, mut cache) = model.forward(&prompt, &empty)?;
    let last = logits.index((.., -1, ..));
    let mut y = sample(&last, opts, rng)?;
    eval([&y])?;
    let prefill_secs = t_prefill.elapsed().as_secs_f64();

    let t_decode = Instant::now();
    let mut out = Vec::with_capacity(max_tokens);
    for _ in 0..max_tokens {
        let id = y.item::<u32>();
        if eos.contains(&id) {
            break;
        }
        on_token(id);
        out.push(id);

        let next = y.index((.., NewAxis));
        let (logits, new_cache) = model.forward(&next, &cache)?;
        cache = new_cache;
        let logits = logits.squeeze_axes(&[1])?;
        y = sample(&logits, opts, rng)?;
        eval([&y])?;
    }
    let decode_secs = t_decode.elapsed().as_secs_f64();

    Ok(GenStats {
        tokens: out,
        prefill_secs,
        decode_secs,
    })
}
