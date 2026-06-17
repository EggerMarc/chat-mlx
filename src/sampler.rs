use anyhow::Result;
use mlx_rs::{Array, array, ops::indexing::argmax_axis, random::categorical};

pub fn sample(logits: &Array, temp: f32) -> Result<Array> {
    if temp == 0.0 {
        Ok(argmax_axis(logits, -1, None)?)
    } else {
        let scaled = logits.multiply(array!(1.0 / temp))?;
        Ok(categorical(&scaled, None, None, None)?)
    }
}
