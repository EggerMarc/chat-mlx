use std::time::Instant;

use anyhow::Result;
use mlx_rs::{
    Array,
    ops::indexing::{IndexOp, NewAxis},
    transforms::eval,
};

use super::{
    cache::KvCache,
    constraint::LogitMask,
    model::Model,
    sampler::{SampleOpts, sample},
};

pub struct GenStats {
    pub tokens: Vec<u32>,
    pub prefill_secs: f64,
    pub decode_secs: f64,
}

#[allow(clippy::too_many_arguments)]
pub fn generate<F: FnMut(u32)>(
    model: &mut Model,
    prompt_ids: &[u32],
    max_tokens: usize,
    opts: &SampleOpts,
    eos: &[u32],
    tokens_per_eval: usize,
    cache: &mut [KvCache],
    mut on_token: F,
) -> Result<GenStats> {
    let prompt = Array::from(prompt_ids).index(NewAxis);

    let t_prefill = Instant::now();
    let logits = model.forward(&prompt, cache)?;
    let last = logits.index((.., -1, ..));
    let mut y = sample(&last, opts)?;
    eval([&y])?;
    let prefill_secs = t_prefill.elapsed().as_secs_f64();

    let t_decode = Instant::now();
    let mut out = Vec::with_capacity(max_tokens);

    let id0 = y.item::<u32>();
    let mut done = eos.contains(&id0);
    if !done {
        on_token(id0);
        out.push(id0);
    }

    let batch_size = tokens_per_eval.max(1);
    while out.len() < max_tokens && !done {
        let mut batch: Vec<Array> = Vec::with_capacity(batch_size);
        while batch.len() < batch_size && (out.len() + batch.len()) < max_tokens {
            let next = y.index((.., NewAxis));
            let logits = model.forward(&next, cache)?;
            let logits = logits.squeeze_axes(&[1])?;
            y = sample(&logits, opts)?;
            batch.push(y.clone());
        }

        eval(&batch)?;
        for a in &batch {
            let id = a.item::<u32>();
            if eos.contains(&id) {
                done = true;
                break;
            }
            on_token(id);
            out.push(id);
            if out.len() >= max_tokens {
                done = true;
                break;
            }
        }
    }
    let decode_secs = t_decode.elapsed().as_secs_f64();

    Ok(GenStats {
        tokens: out,
        prefill_secs,
        decode_secs,
    })
}

/// Like [`generate`], but applies a [`LogitMask`] before sampling each token
/// (constrained decoding). Runs one token per eval — the mask depends on the
/// realized token, so the batched pipeline can't be used.
#[allow(clippy::too_many_arguments)]
pub fn generate_constrained<F: FnMut(u32)>(
    model: &mut Model,
    prompt_ids: &[u32],
    max_tokens: usize,
    opts: &SampleOpts,
    eos: &[u32],
    cache: &mut [KvCache],
    constraint: &mut dyn LogitMask,
    mut on_token: F,
) -> Result<GenStats> {
    let prompt = Array::from(prompt_ids).index(NewAxis);

    let t_prefill = Instant::now();
    let logits = model.forward(&prompt, cache)?;
    let last = constraint.mask(&logits.index((.., -1, ..)))?;
    let mut y = sample(&last, opts)?;
    eval([&y])?;
    let prefill_secs = t_prefill.elapsed().as_secs_f64();

    let t_decode = Instant::now();
    let mut out = Vec::with_capacity(max_tokens);
    loop {
        let id = y.item::<u32>();
        if eos.contains(&id) {
            break;
        }
        on_token(id);
        out.push(id);
        constraint.accept(id);
        if out.len() >= max_tokens {
            break;
        }

        let next = y.index((.., NewAxis));
        let logits = model.forward(&next, cache)?;
        let logits = constraint.mask(&logits.squeeze_axes(&[1])?)?;
        y = sample(&logits, opts)?;
        eval([&y])?;
    }
    let decode_secs = t_decode.elapsed().as_secs_f64();

    Ok(GenStats {
        tokens: out,
        prefill_secs,
        decode_secs,
    })
}
