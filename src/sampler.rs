use anyhow::Result;
use mlx_rs::{Array, Dtype, ops::indexing::argmax_axis, transforms::eval};
use rand::{Rng, rngs::StdRng};

#[derive(Debug, Clone)]
pub struct SampleOpts {
    pub temp: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
}

pub fn sample(logits: &Array, opts: &SampleOpts, rng: &mut StdRng) -> Result<Array> {
    if opts.temp == 0.0 {
        return Ok(argmax_axis(logits, -1, None)?);
    }

    let flat = logits.as_dtype(Dtype::Float32)?.reshape(&[-1])?;
    eval([&flat])?;
    let logits_host = flat.as_slice::<f32>();

    let id = sample_host(logits_host, opts, rng);
    Ok(Array::from(&[id][..]))
}

fn sample_host(logits: &[f32], opts: &SampleOpts, rng: &mut StdRng) -> u32 {
    let inv_temp = 1.0 / opts.temp;

    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let k = opts.top_k.unwrap_or(idx.len()).clamp(1, idx.len());
    idx.truncate(k);

    let max_logit = logits[idx[0]];
    let mut probs: Vec<f32> = idx
        .iter()
        .map(|&i| ((logits[i] - max_logit) * inv_temp).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }

    if let Some(top_p) = opts.top_p {
        let mut cum = 0.0;
        let mut cutoff = probs.len();
        for (j, &p) in probs.iter().enumerate() {
            cum += p;
            if cum >= top_p {
                cutoff = j + 1;
                break;
            }
        }
        idx.truncate(cutoff);
        probs.truncate(cutoff);
        let s: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= s;
        }
    }

    let r: f32 = rng.r#gen::<f32>();
    let mut acc = 0.0;
    for (j, &p) in probs.iter().enumerate() {
        acc += p;
        if r <= acc {
            return idx[j] as u32;
        }
    }
    idx[idx.len() - 1] as u32
}
