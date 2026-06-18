use std::io::Write;

use anyhow::Result;
use clap::Parser;

use chat_mlx::engine::{generate, sampler::SampleOpts, template};
use chat_mlx::loader;

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

    eprintln!("[info] loading {} …", cli.model);
    let loaded = loader::load(&cli.model, cli.quantize)?;
    let args = &loaded.args;
    eprintln!(
        "[info] model: dim={} layers={} heads={}/{} head_dim={} vocab={}",
        args.dim, args.n_layers, args.n_heads, args.n_kv_heads, args.head_dim, args.vocab_size
    );
    let mut m = loaded.model;
    let tokenizer = loaded.tokenizer;
    let eos = loaded.eos;

    let prompt = template::simple(cli.system.as_deref(), &cli.prompt);
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
