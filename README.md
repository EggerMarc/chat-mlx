# chat-mlx (playground)

Standalone MLX inference for MiniCPM5-1B (and other Llama-family models) on
Apple Silicon. Deliberately **not** a chat-rs provider yet — this is a place to
get inference working, profile it, and optimize before porting the core into
`providers/mlx`.

## Layers (what we own vs what MLX owns)

```
mlx-rs        -> Array + math kernels + nn::{Linear,Rope,RmsNorm,Embedding} + fast::sdpa   [the framework]
src/model.rs  -> the architecture: compose those modules into a MiniCPM5/Llama decoder      [OURS]
src/generate.rs -> prefill + autoregressive decode loop (KV cache, EOS, max_tokens)         [OURS]
src/sampler.rs  -> greedy / temperature sampling from logits                                [OURS]
src/prompt.rs   -> ChatML formatting                                                        [OURS]
src/config.rs   -> parse HF config.json / generation_config.json                            [OURS]
src/main.rs     -> CLI + HF download wiring                                                 [OURS]
```

`nn/` doesn't exist as a folder: mlx-rs already ships those bricks. Struct field
names in `model.rs` mirror HF tensor keys (`model.layers.N.self_attn.q_proj`, …)
so `load_safetensors` maps the official weights with no manual remapping.

## Run

First build is heavy (compiles the MLX C++/Metal backend; needs Xcode CLT + cmake).

```bash
cargo run --release -- --prompt "Explain RoPE in one sentence."
# options: --system, --max-tokens, --temp, --model, --quantize
```

Default model: `openbmb/MiniCPM5-1B` (bf16). The 4-bit `openbmb/MiniCPM5-1B-MLX`
needs pre-quantized loading (TODO); for now load bf16 and pass `--quantize` to
quantize at runtime.

## Status / TODO

- [x] First successful generate on bf16 weights (MiniCPM5-1B, greedy, coherent output)
- [x] top-p / top-k / temperature sampling (host-side, seeded)
- [x] live streaming output (decode_stream)
- [x] tokens/sec timing (prefill vs decode)
- [ ] load pre-quantized `*-MLX` safetensors directly
- [ ] release-build perf pass (top-k via select_nth_unstable; on-device sampler)
- [ ] a custom fused Metal kernel experiment
- [ ] port the `model.rs` + `generate.rs` core into `providers/mlx`
