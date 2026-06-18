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

    /// Runtime-quantize the loaded fp weights (group size 64).
    #[clap(long)]
    quantize: bool,

    /// Quantization bit width when --quantize is set (2, 3, 4, or 8).
    #[clap(long, default_value = "4")]
    bits: i32,

    /// Greedy decode with n-gram / prompt-lookup speculation (forces a growable cache).
    #[clap(long)]
    ngram: bool,

    /// N-gram suffix length to match for speculation.
    #[clap(long, default_value = "3")]
    ngram_n: usize,

    /// Max draft tokens proposed per speculation round.
    #[clap(long, default_value = "8")]
    ngram_k: usize,

    /// PRNG seed.
    #[clap(long, default_value = "0")]
    seed: u64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    mlx_rs::random::seed(cli.seed)?;

    let quant = if cli.quantize {
        Some(
            chat_mlx::Quantize::from_bits(cli.bits)
                .ok_or_else(|| anyhow::anyhow!("unsupported --bits {} (use 2, 3, 4, or 8)", cli.bits))?,
        )
    } else {
        None
    };

    eprintln!("[info] loading {} …", cli.model);
    let loaded = loader::load(&cli.model, quant)?;
    let args = &loaded.args;
    eprintln!(
        "[info] model: dim={} layers={} heads={}/{} head_dim={} vocab={}",
        args.dim, args.n_layers, args.n_heads, args.n_kv_heads, args.head_dim, args.vocab_size
    );
    let mut m = loaded.model;
    let tokenizer = loaded.tokenizer;
    let eos = loaded.eos;

    let mut turns = Vec::new();
    if let Some(sys) = cli.system.as_deref() {
        turns.push(template::Turn {
            role: "system",
            content: sys.to_string(),
        });
    }
    turns.push(template::Turn {
        role: "user",
        content: cli.prompt.clone(),
    });
    let prompt = loaded.chat_template.render(&turns);
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
    let mut emit = |id: u32| -> bool {
        if let Ok(Some(s)) = stream.step(id) {
            print!("{s}");
            let _ = std::io::stdout().flush();
        }
        true
    };

    let stats = if cli.ngram {
        eprintln!(
            "[info] n-gram speculation: n={} k={} (greedy, growable cache)",
            cli.ngram_n, cli.ngram_k
        );
        let mut kv_cache = m.make_cache(None, cli.sink_tokens);
        generate::generate_ngram(
            &mut m,
            ids,
            cli.max_tokens,
            &eos,
            &mut kv_cache,
            cli.ngram_n,
            cli.ngram_k,
            &mut emit,
        )?
    } else {
        let max_context = (cli.max_context > 0).then_some(cli.max_context);
        let mut kv_cache = m.make_cache(max_context, cli.sink_tokens);
        generate::generate(
            &mut m,
            ids,
            cli.max_tokens,
            &opts,
            &eos,
            cli.tokens_per_eval,
            &mut kv_cache,
            &mut emit,
        )?
    };
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
