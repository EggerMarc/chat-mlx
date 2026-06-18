use mlx_rs::{
    Array,
    builder::Builder,
    error::Exception,
    fast::scaled_dot_product_attention,
    macros::{ModuleParameters, Quantizable},
    module::Module,
    nn,
    quantization::MaybeQuantized,
};

use crate::cache::KvCache;
use crate::config::ModelArgs;

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Attention {
    n_heads: i32,
    n_kv_heads: i32,
    scale: f32,
    use_qk_norm: bool,

    #[param]
    q_norm: nn::RmsNorm,
    #[param]
    k_norm: nn::RmsNorm,

    #[quantizable]
    #[param]
    q_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    k_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    v_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    o_proj: MaybeQuantized<nn::Linear>,

    #[param]
    rope: nn::Rope,
}

impl Attention {
    fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let q_dim = args.n_heads * args.head_dim;
        let kv_dim = args.n_kv_heads * args.head_dim;

        let q_proj = nn::LinearBuilder::new(args.dim, q_dim)
            .bias(false)
            .build()?;
        let k_proj = nn::LinearBuilder::new(args.dim, kv_dim)
            .bias(false)
            .build()?;
        let v_proj = nn::LinearBuilder::new(args.dim, kv_dim)
            .bias(false)
            .build()?;
        let o_proj = nn::LinearBuilder::new(q_dim, args.dim)
            .bias(false)
            .build()?;

        let rope = nn::RopeBuilder::new(args.head_dim)
            .traditional(false)
            .base(args.rope_theta)
            .build()?;

        Ok(Self {
            n_heads: args.n_heads,
            n_kv_heads: args.n_kv_heads,
            scale: (args.head_dim as f32).powf(-0.5),
            use_qk_norm: args.use_qk_norm,
            q_norm: nn::RmsNormBuilder::new(args.head_dim)
                .eps(args.norm_eps)
                .build()?,
            k_norm: nn::RmsNormBuilder::new(args.head_dim)
                .eps(args.norm_eps)
                .build()?,
            q_proj: MaybeQuantized::new(q_proj),
            k_proj: MaybeQuantized::new(k_proj),
            v_proj: MaybeQuantized::new(v_proj),
            o_proj: MaybeQuantized::new(o_proj),
            rope,
        })
    }

    #[allow(non_snake_case)]
    fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut KvCache,
    ) -> Result<Array, Exception> {
        let B = x.shape()[0];
        let L = x.shape()[1];

        let mut q = self.q_proj.forward(x)?;
        let mut k = self.k_proj.forward(x)?;
        let mut v = self.v_proj.forward(x)?;

        q = q
            .reshape(&[B, L, self.n_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        k = k
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        v = v
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        if self.use_qk_norm {
            q = self.q_norm.forward(&q)?;
            k = self.k_norm.forward(&k)?;
        }

        let offset = cache.offset();
        q = self.rope.forward((&q, offset))?;
        k = self.rope.forward((&k, offset))?;
        let (k, v) = cache.update_and_fetch(&k, &v)?;

        let out = scaled_dot_product_attention(q, &k, &v, self.scale, mask.map(Into::into))?;
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[B, L, -1])?;
        self.o_proj.forward(&out)
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Mlp {
    #[quantizable]
    #[param]
    gate_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    up_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    down_proj: MaybeQuantized<nn::Linear>,
}

impl Mlp {
    fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let gate_proj = nn::LinearBuilder::new(args.dim, args.hidden_dim)
            .bias(false)
            .build()?;
        let up_proj = nn::LinearBuilder::new(args.dim, args.hidden_dim)
            .bias(false)
            .build()?;
        let down_proj = nn::LinearBuilder::new(args.hidden_dim, args.dim)
            .bias(false)
            .build()?;
        Ok(Self {
            gate_proj: MaybeQuantized::new(gate_proj),
            up_proj: MaybeQuantized::new(up_proj),
            down_proj: MaybeQuantized::new(down_proj),
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gated = nn::silu(self.gate_proj.forward(x)?)?.multiply(self.up_proj.forward(x)?)?;
        self.down_proj.forward(&gated)
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct DecoderLayer {
    #[quantizable]
    #[param]
    self_attn: Attention,
    #[quantizable]
    #[param]
    mlp: Mlp,
    #[param]
    input_layernorm: nn::RmsNorm,
    #[param]
    post_attention_layernorm: nn::RmsNorm,
}

impl DecoderLayer {
    fn new(args: &ModelArgs) -> Result<Self, Exception> {
        Ok(Self {
            self_attn: Attention::new(args)?,
            mlp: Mlp::new(args)?,
            input_layernorm: nn::RmsNormBuilder::new(args.dim)
                .eps(args.norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(args.dim)
                .eps(args.norm_eps)
                .build()?,
        })
    }

    fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut KvCache,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(x)?;
        let attn = self.self_attn.forward(&normed, mask, cache)?;
        let h = x.add(attn)?;
        let ff = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&h)?)?;
        h.add(ff)
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Backbone {
    #[quantizable]
    #[param]
    embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    layers: Vec<DecoderLayer>,
    #[param]
    norm: nn::RmsNorm,
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Model {
    #[quantizable]
    #[param]
    model: Backbone,
    #[quantizable]
    #[param]
    lm_head: MaybeQuantized<nn::Linear>,
}

impl Model {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(args.vocab_size, args.dim)?;
        let layers = (0..args.n_layers)
            .map(|_| DecoderLayer::new(args))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(args.dim)
            .eps(args.norm_eps)
            .build()?;
        let lm_head = nn::LinearBuilder::new(args.dim, args.vocab_size)
            .bias(false)
            .build()?;

        Ok(Self {
            model: Backbone {
                embed_tokens: MaybeQuantized::new(embed_tokens),
                layers,
                norm,
            },
            lm_head: MaybeQuantized::new(lm_head),
        })
    }

    pub fn make_cache(&self, max_size: Option<i32>, keep: i32) -> Vec<KvCache> {
        (0..self.model.layers.len())
            .map(|_| KvCache::new(256, max_size, keep))
            .collect()
    }

    pub fn forward(&mut self, tokens: &Array, cache: &mut [KvCache]) -> Result<Array, Exception> {
        let mut h = self.model.embed_tokens.forward(tokens)?;

        let mask = if h.shape()[1] > 1 {
            let m = nn::MultiHeadAttention::create_additive_causal_mask::<f32>(h.shape()[1])?;
            Some(m.as_dtype(h.dtype())?)
        } else {
            None
        };

        for (layer, layer_cache) in self.model.layers.iter_mut().zip(cache.iter_mut()) {
            h = layer.forward(&h, mask.as_ref(), layer_cache)?;
        }

        let h = self.model.norm.forward(&h)?;
        self.lm_head.forward(&h)
    }
}
