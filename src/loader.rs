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

pub fn load(model_id: &str, quantize: bool) -> Result<Loaded> {
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

    if quantize {
        model = mlx_rs::nn::quantize(model, Some(64), Some(4))?;
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
