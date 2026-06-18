use mlx_rs::{
    Array,
    error::Exception,
    ops::concatenate_axis,
    ops::indexing::{IndexMutOp, IndexOp},
    ops::zeros_dtype,
};

pub struct KvCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
    size: i32,
    step: i32,
    max_size: Option<i32>,
    keep: i32,
    ring: i32,
}

impl KvCache {
    pub fn new(step: i32, max_size: Option<i32>, keep: i32) -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
            size: 0,
            step: step.max(1),
            max_size,
            keep: keep.max(0),
            ring: keep.max(0),
        }
    }

    pub fn offset(&self) -> i32 {
        self.offset
    }

    pub fn update_and_fetch(
        &mut self,
        k: &Array,
        v: &Array,
    ) -> Result<(Array, Array), Exception> {
        let l = k.shape()[2];
        match self.max_size {
            Some(max) => self.update_rotating(k, v, l, max),
            None => self.update_growing(k, v, l),
        }
    }

    fn update_growing(
        &mut self,
        k: &Array,
        v: &Array,
        l: i32,
    ) -> Result<(Array, Array), Exception> {
        let prev = self.size;
        let needed = prev + l;
        let (mut kb, mut vb) = self.ensure_capacity(k, v, needed)?;

        kb.index_mut((.., .., prev..needed, ..), k.clone());
        vb.index_mut((.., .., prev..needed, ..), v.clone());

        self.size = needed;
        self.offset += l;

        let out = (
            kb.index((.., .., 0..self.size, ..)),
            vb.index((.., .., 0..self.size, ..)),
        );
        self.keys = Some(kb);
        self.values = Some(vb);
        Ok(out)
    }

    fn ensure_capacity(
        &mut self,
        k: &Array,
        v: &Array,
        needed: i32,
    ) -> Result<(Array, Array), Exception> {
        let b = k.shape()[0];
        let h = k.shape()[1];
        let d = k.shape()[3];
        let new_cap = ((needed + self.step - 1) / self.step) * self.step;

        match (self.keys.take(), self.values.take()) {
            (Some(kb), Some(vb)) if kb.shape()[2] >= needed => Ok((kb, vb)),
            (Some(kb), Some(vb)) => {
                let grow = new_cap - kb.shape()[2];
                let kpad = zeros_dtype(&[b, h, grow, d], k.dtype())?;
                let vpad = zeros_dtype(&[b, h, grow, d], v.dtype())?;
                Ok((
                    concatenate_axis(&[&kb, &kpad], 2)?,
                    concatenate_axis(&[&vb, &vpad], 2)?,
                ))
            }
            _ => Ok((
                zeros_dtype(&[b, h, new_cap, d], k.dtype())?,
                zeros_dtype(&[b, h, new_cap, d], v.dtype())?,
            )),
        }
    }

    fn update_rotating(
        &mut self,
        k: &Array,
        v: &Array,
        l: i32,
        max: i32,
    ) -> Result<(Array, Array), Exception> {
        let b = k.shape()[0];
        let h = k.shape()[1];
        let d = k.shape()[3];

        let (mut kb, mut vb) = match (self.keys.take(), self.values.take()) {
            (Some(kb), Some(vb)) => (kb, vb),
            _ => {
                self.ring = self.keep;
                (
                    zeros_dtype(&[b, h, max, d], k.dtype())?,
                    zeros_dtype(&[b, h, max, d], v.dtype())?,
                )
            }
        };

        if l == 1 {
            let idx = if self.size < max {
                self.size
            } else {
                let r = self.ring;
                self.ring += 1;
                if self.ring >= max {
                    self.ring = self.keep;
                }
                r
            };
            kb.index_mut((.., .., idx..idx + 1, ..), k.clone());
            vb.index_mut((.., .., idx..idx + 1, ..), v.clone());
            if self.size < max {
                self.size += 1;
            }
            self.offset += 1;
        } else if self.size != 0 {
            return Err(Exception::from(
                "rotating KV cache does not support multi-token updates after prefill",
            ));
        } else if l <= max {
            kb.index_mut((.., .., 0..l, ..), k.clone());
            vb.index_mut((.., .., 0..l, ..), v.clone());
            self.size = l;
            self.ring = self.keep;
            self.offset += l;
        } else {
            let recent = max - self.keep;
            kb.index_mut((.., .., 0..self.keep, ..), k.index((.., .., 0..self.keep, ..)));
            kb.index_mut((.., .., self.keep..max, ..), k.index((.., .., (l - recent)..l, ..)));
            vb.index_mut((.., .., 0..self.keep, ..), v.index((.., .., 0..self.keep, ..)));
            vb.index_mut((.., .., self.keep..max, ..), v.index((.., .., (l - recent)..l, ..)));
            self.size = max;
            self.ring = self.keep;
            self.offset += l;
        }

        let out = (
            kb.index((.., .., 0..self.size, ..)),
            vb.index((.., .., 0..self.size, ..)),
        );
        self.keys = Some(kb);
        self.values = Some(vb);
        Ok(out)
    }
}
