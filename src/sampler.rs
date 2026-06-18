use anyhow::Result;
use mlx_rs::{
    Array,
    ops::{
        argpartition_axis, argsort_axis, cumsum,
        indexing::{IndexOp, NewAxis, argmax_axis, put_along_axis, take_along_axis},
        softmax_axis, which,
    },
    random,
};

#[derive(Debug, Clone)]
pub struct SampleOpts {
    pub temp: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
}

pub fn sample(logits: &Array, opts: &SampleOpts) -> Result<Array> {
    if opts.temp == 0.0 {
        return Ok(argmax_axis(logits, -1, None)?);
    }

    let scaled = logits.multiply(Array::from_f32(1.0 / opts.temp))?;
    let shape = scaled.shape();
    let vocab = shape[shape.len() - 1];

    let filtered = match opts.top_k {
        Some(k) if (k as i32) >= 1 && (k as i32) < vocab => apply_top_k(&scaled, k as i32)?,
        _ => scaled,
    };

    match opts.top_p {
        Some(p) if p > 0.0 && p < 1.0 => sample_top_p(&filtered, p),
        _ => Ok(random::categorical(&filtered, None, None, None)?),
    }
}

fn apply_top_k(logits: &Array, k: i32) -> Result<Array> {
    let vocab = {
        let shape = logits.shape();
        shape[shape.len() - 1]
    };
    let order = argpartition_axis(logits.multiply(Array::from_f32(-1.0))?, k - 1, -1)?;
    let drop = order.index((.., k..vocab));
    Ok(put_along_axis(
        logits,
        &drop,
        Array::from_f32(f32::NEG_INFINITY),
        -1,
    )?)
}

fn sample_top_p(logits: &Array, p: f32) -> Result<Array> {
    let probs = softmax_axis(logits, -1, None)?;
    let order = argsort_axis(&probs, -1)?;
    let sorted = take_along_axis(&probs, &order, -1)?;
    let cumulative = cumsum(&sorted, -1, false, true)?;

    let keep = cumulative.gt(Array::from_f32(1.0 - p))?;
    let kept = which(&keep, &sorted, Array::from_f32(0.0))?;

    let choice = random::categorical(&kept.log()?, None, None, None)?;
    let token = take_along_axis(&order, &choice.index((.., NewAxis)), -1)?;
    Ok(token.squeeze_axes(&[1])?)
}
