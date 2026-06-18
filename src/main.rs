mod cache;
mod config;
mod generate;
mod model;
mod prompt;
mod sampler;

use std::collections::BTreeSet;
use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use hf_hub::{Repo, RepoType, api::sync::Api, api::sync::ApiRepo};
use mlx_rs::module::ModuleParametersExt;
use tokenizers::Tokenizer;

use crate::{
    config::{GenerationConfig, HfConfig, ModelArgs},
    model::Model,
    sampler::SampleOpts,
};

#[derive(Parser)]
#[command(about = "Standalone MLX inference for MiniCPM5 / Llama-family models")]
struct Cli {
    /// HF repo id of the model (bf16 safetensors).
    #[clap(long, default_value = "openbmb/MiniCPM5-1B")]
    model: String,

    /// User message.
    #[clap(
        long,
        default_value = "Explain rotary position embeddings in one sentence."
    )]
    prompt: String,

    /// Optional system prompt.
    #[clap(long)]
    system: Option<String>,

    /// Max tokens to generate.
    #[clap(long, default_value = "256")]
    max_tokens: usize,

    /// Sampling temperature (0.0 = greedy).
    #[clap(long, default_value = "0.0")]
    temp: f32,

    #[clap(long)]
    top_k: Option<usize>,

    #[clap(long)]
    top_p: Option<f32>,

    #[clap(long, default_value = "8")]
    tokens_per_eval: usize,

    /// Max tokens retained in the KV cache (rotating window). 0 = unbounded.
    #[clap(long, default_value = "4096")]
    max_context: i32,

    /// Leading tokens pinned as attention sinks when the window rotates.
    #[clap(long, default_value = "4")]
    sink_tokens: i32,

    /// Runtime-quantize the loaded fp weights to 4-bit (group size 64).
    #[clap(long)]
    quantize: bool,

    /// PRNG seed.
    #[clap(long, default_value = "0")]
    seed: u64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    mlx_rs::random::seed(cli.seed)?;

    let api = Api::new()?;
    let repo = api.repo(Repo::new(cli.model.clone(), RepoType::Model));

    eprintln!("[info] resolving model files for {} …", cli.model);
    let config_path = repo.get("config.json").context("fetch config.json")?;
    let tokenizer_path = repo.get("tokenizer.json").context("fetch tokenizer.json")?;
    let weight_paths = resolve_weights(&repo).context("resolve model weight files")?;

    let gen_cfg: GenerationConfig = match repo.get("generation_config.json") {
        Ok(p) => serde_json::from_reader(std::fs::File::open(p)?)?,
        Err(_) => GenerationConfig::default(),
    };
    let eos = gen_cfg.eos_or_default();

    let hf: HfConfig = serde_json::from_reader(std::fs::File::open(config_path)?)?;
    let args: ModelArgs = hf.into();
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e))?;

    eprintln!(
        "[info] building model: dim={} layers={} heads={}/{} head_dim={} vocab={}",
        args.dim, args.n_layers, args.n_heads, args.n_kv_heads, args.head_dim, args.vocab_size
    );
    let mut m = Model::new(&args)?;
    eprintln!("[info] loading weights from {} shard(s) …", weight_paths.len());
    for p in &weight_paths {
        // Lenient: each call updates only the params present in that shard.
        m.load_safetensors(p)
            .context("load_safetensors (struct field names must match HF tensor keys)")?;
    }

    if cli.quantize {
        eprintln!("[info] runtime-quantizing to 4-bit …");
        m = mlx_rs::nn::quantize(m, Some(64), Some(4))?;
    }

    let prompt = prompt::simple(cli.system.as_deref(), &cli.prompt);
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!(e))?;
    let ids = encoding.get_ids();
    eprintln!("[info] prompt tokens: {}", ids.len());

    print!("{}", cli.prompt);
    let _ = std::io::stdout().flush();

    let opts = SampleOpts {
        temp: cli.temp,
        top_k: cli.top_k,
        top_p: cli.top_p,
    };
    let mut stream = tokenizer.decode_stream(true);

    let max_context = (cli.max_context > 0).then_some(cli.max_context);
    if let Some(n) = max_context {
        eprintln!(
            "[info] rotating KV cache: window={} sink={}",
            n, cli.sink_tokens
        );
    }
    let mut kv_cache = m.make_cache(max_context, cli.sink_tokens);

    let stats = generate::generate(
        &mut m,
        ids,
        cli.max_tokens,
        &opts,
        &eos,
        cli.tokens_per_eval,
        &mut kv_cache,
        |id| {
            if let Ok(Some(s)) = stream.step(id) {
                print!("{s}");
                let _ = std::io::stdout().flush();
            }
        },
    )?;
    println!();

    let n = stats.tokens.len();
    eprintln!(
        "[info] prefill {} tok in {:.3}s ({:.1} tok/s) | decode {} tok in {:.3}s ({:.1} tok/s)",
        ids.len(),
        stats.prefill_secs,
        ids.len() as f64 / stats.prefill_secs.max(1e-9),
        n,
        stats.decode_secs,
        n as f64 / stats.decode_secs.max(1e-9),
    );

    Ok(())
}

/// Resolve the model weight file(s) to local paths, downloading as needed.
/// Handles both single-file repos (`model.safetensors`) and the sharded
/// layout (`model.safetensors.index.json` -> N shards, e.g. MiniCPM5-1B's
/// `model-00000-of-00001.safetensors`).
fn resolve_weights(repo: &ApiRepo) -> Result<Vec<PathBuf>> {
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
