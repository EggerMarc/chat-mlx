use std::time::Instant;

use anyhow::Result;
use mlx_rs::{
    Array,
    ops::indexing::{IndexOp, NewAxis, argmax_axis},
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
pub fn generate<F: FnMut(u32) -> bool>(
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
        out.push(id0);
        if !on_token(id0) {
            done = true;
        }
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
            out.push(id);
            if !on_token(id) || out.len() >= max_tokens {
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

#[allow(clippy::too_many_arguments)]
pub fn generate_ngram<F: FnMut(u32) -> bool>(
    model: &mut Model,
    prompt_ids: &[u32],
    max_tokens: usize,
    eos: &[u32],
    cache: &mut [KvCache],
    n: usize,
    k: usize,
    mut on_token: F,
) -> Result<GenStats> {
    let mut tokens = prompt_ids.to_vec();

    let t_prefill = Instant::now();
    let prompt = Array::from(prompt_ids).index(NewAxis);
    let logits = model.forward(&prompt, cache)?;
    let first = argmax_axis(logits.index((.., -1, ..)), -1, None)?;
    eval([&first])?;
    let mut next_id = first.item::<u32>();
    let prefill_secs = t_prefill.elapsed().as_secs_f64();

    let t_decode = Instant::now();
    let mut out = Vec::with_capacity(max_tokens);
    'outer: loop {
        if eos.contains(&next_id) {
            break;
        }
        out.push(next_id);
        tokens.push(next_id);
        if !on_token(next_id) || out.len() >= max_tokens {
            break;
        }

        let draft = ngram_lookup(&tokens, n, k);
        let kept_before = cache.first().map(|c| c.offset()).unwrap_or(0);

        let mut input_ids = Vec::with_capacity(1 + draft.len());
        input_ids.push(next_id);
        input_ids.extend_from_slice(&draft);
        let input = Array::from(&input_ids[..]).index(NewAxis);

        let logits = model.forward(&input, cache)?;
        let preds = argmax_axis(&logits, -1, None)?;
        eval([&preds])?;
        let preds = preds.as_slice::<u32>();

        let mut m = 0;
        while m < draft.len() && draft[m] == preds[m] {
            m += 1;
        }

        let keep = kept_before + 1 + m as i32;
        for c in cache.iter_mut() {
            c.truncate(keep);
        }

        for &id in draft.iter().take(m) {
            if eos.contains(&id) {
                break 'outer;
            }
            out.push(id);
            tokens.push(id);
            if !on_token(id) || out.len() >= max_tokens {
                break 'outer;
            }
        }

        next_id = preds[m];
    }
    let decode_secs = t_decode.elapsed().as_secs_f64();

    Ok(GenStats {
        tokens: out,
        prefill_secs,
        decode_secs,
    })
}

fn ngram_lookup(tokens: &[u32], n: usize, k: usize) -> Vec<u32> {
    if n == 0 || k == 0 || tokens.len() <= n {
        return Vec::new();
    }
    let suffix = &tokens[tokens.len() - n..];
    let search_end = tokens.len() - n;
    for p in (0..search_end).rev() {
        if &tokens[p..p + n] == suffix {
            let start = p + n;
            let end = (start + k).min(tokens.len());
            return tokens[start..end].to_vec();
        }
    }
    Vec::new()
}

#[allow(clippy::too_many_arguments)]
pub fn generate_constrained<F: FnMut(u32) -> bool>(
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
        out.push(id);
        constraint.accept(id);
        if !on_token(id) || out.len() >= max_tokens {
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
