//! CLI + HuggingFace download wiring. Glue only — the interesting code is in
//! `model.rs` (architecture) and `generate.rs` (decode loop).

mod config;
mod generate;
mod model;
mod prompt;
mod sampler;

use std::io::Write;

use anyhow::{Context, Result};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use mlx_rs::module::ModuleParametersExt;
use tokenizers::Tokenizer;

use crate::{
    config::{GenerationConfig, HfConfig, ModelArgs},
    model::Model,
};

#[derive(Parser)]
#[command(about = "Standalone MLX inference for MiniCPM5 / Llama-family models")]
struct Cli {
    /// HF repo id of the model (bf16 safetensors).
    #[clap(long, default_value = "openbmb/MiniCPM5-1B")]
    model: String,

    /// User message.
    #[clap(long, default_value = "Explain rotary position embeddings in one sentence.")]
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
    let weights_path = repo
        .get("model.safetensors")
        .context("fetch model.safetensors")?;
    // generation_config.json is optional; fall back to known EOS ids.
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
    eprintln!("[info] loading weights …");
    m.load_safetensors(&weights_path)
        .context("load_safetensors (struct field names must match HF tensor keys)")?;

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

    let mut tok = tokenizer.clone();
    let _ = tok; // reserved for incremental detokenization tweaks

    let generated = generate::generate(&mut m, ids, cli.max_tokens, cli.temp, &eos, |_id| {})?;

    // Decode the full generation in one shot (incremental streaming detok is a
    // later refinement — see README).
    let text = tokenizer
        .decode(&generated, true)
        .map_err(|e| anyhow::anyhow!(e))?;
    println!("\n{text}");
    eprintln!("\n[info] generated {} tokens", generated.len());

    Ok(())
}
