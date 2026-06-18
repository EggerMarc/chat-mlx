use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::{Context, Result};
use hf_hub::{
    Repo, RepoType,
    api::sync::{Api, ApiRepo},
};
use mlx_rs::module::ModuleParametersExt;
use tokenizers::Tokenizer;

use crate::engine::config::{GenerationConfig, HfConfig, ModelArgs};
use crate::engine::model::Model;

pub struct Loaded {
    pub model: Model,
    pub tokenizer: Tokenizer,
    pub args: ModelArgs,
    pub eos: Vec<u32>,
    pub model_type: String,
}

/// Runtime quantization level (MLX group-affine, group size 64). Lower bits =
/// less memory bandwidth per token (faster decode) at some quality cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantize {
    /// 2-bit — fastest, lowest quality.
    Q2,
    /// 3-bit.
    Q3,
    /// 4-bit — the usual sweet spot.
    Q4,
    /// 8-bit — highest quality of the quantized options.
    Q8,
}

impl Quantize {
    pub fn bits(self) -> i32 {
        match self {
            Quantize::Q2 => 2,
            Quantize::Q3 => 3,
            Quantize::Q4 => 4,
            Quantize::Q8 => 8,
        }
    }

    pub fn group_size(self) -> i32 {
        64
    }

    /// Map a bit width to a level, if supported.
    pub fn from_bits(bits: i32) -> Option<Quantize> {
        match bits {
            2 => Some(Quantize::Q2),
            3 => Some(Quantize::Q3),
            4 => Some(Quantize::Q4),
            8 => Some(Quantize::Q8),
            _ => None,
        }
    }
}

pub fn load(model_id: &str, quant: Option<Quantize>) -> Result<Loaded> {
    let api = Api::new()?;
    let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

    let config_path = repo.get("config.json").context("fetch config.json")?;
    let tokenizer_path = repo.get("tokenizer.json").context("fetch tokenizer.json")?;
    let weight_paths = resolve_weights(&repo).context("resolve model weight files")?;

    let gen_cfg: GenerationConfig = match repo.get("generation_config.json") {
        Ok(p) => serde_json::from_reader(std::fs::File::open(p)?)?,
        Err(_) => GenerationConfig::default(),
    };

    let eos = gen_cfg.eos_or_default();

    let hf: HfConfig = serde_json::from_reader(std::fs::File::open(config_path)?)?;
    let model_type = hf.model_type.clone();
    let args: ModelArgs = hf.into();
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e))?;

    let mut model = Model::new(&args)?;
    for p in &weight_paths {
        model
            .load_safetensors(p)
            .context("load_safetensors (struct field names must match HF tensor keys)")?;
    }

    // Models with tied embeddings (e.g. Qwen2.5) ship no `lm_head.weight`;
    // share the input embeddings into the output projection. Must precede
    // quantization.
    if args.tie_word_embeddings {
        model.tie_lm_head();
    }

    if let Some(q) = quant {
        model = mlx_rs::nn::quantize(model, Some(q.group_size()), Some(q.bits()))?;
    }

    Ok(Loaded {
        model,
        tokenizer,
        args,
        eos,
        model_type,
    })
}

pub fn resolve_weights(repo: &ApiRepo) -> Result<Vec<PathBuf>> {
    if let Ok(index_path) = repo.get("model.safetensors.index.json") {
        let v: serde_json::Value = serde_json::from_reader(std::fs::File::open(index_path)?)?;
        let map = v
            .get("weight_map")
            .and_then(|m| m.as_object())
            .context("index.json missing weight_map")?;
        let shards: BTreeSet<String> = map
            .values()
            .filter_map(|x| x.as_str().map(String::from))
            .collect();
        let mut paths = Vec::with_capacity(shards.len());
        for shard in shards {
            paths.push(repo.get(&shard).with_context(|| format!("fetch {shard}"))?);
        }
        return Ok(paths);
    }
    Ok(vec![
        repo.get("model.safetensors")
            .context("fetch model.safetensors")?,
    ])
}
