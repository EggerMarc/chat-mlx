//! Token sampling from logits. Operates on a tiny `[vocab]` array.
//!
//! Today: greedy (temp == 0) and temperature sampling, mirroring the mlx-rs
//! mistral example. top-p / top-k are TODO — see README.

use anyhow::Result;
use mlx_rs::{array, ops::indexing::argmax_axis, random::categorical, Array};

/// Pick the next token id from a `[.., vocab]` logits array.
pub fn sample(logits: &Array, temp: f32) -> Result<Array> {
    if temp == 0.0 {
        Ok(argmax_axis(logits, -1, None)?)
    } else {
        let scaled = logits.multiply(array!(1.0 / temp))?;
        Ok(categorical(&scaled, None, None, None)?)
    }
}
