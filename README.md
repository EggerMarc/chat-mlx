# chat-mlx

Standalone MLX inference for MiniCPM5-1B (and other Llama-family models) on
Apple Silicon. The engine works today as a CLI; the project is now being grown
into a **[chat-rs](../chat-rs) provider** so it can be driven through
`chat_rs::ChatBuilder` like any other provider — participating in the
tool-calling, structured-output, and streaming chat loop. See
[Roadmap → chat-rs provider](#roadmap--chat-rs-provider) below.

This stays its own repo (`lib + thin bin`): a `chat_mlx` library exposing the
provider, plus the existing CLI as a thin binary over it. It depends on
`chat-core` directly, exactly like the existing local-inference provider
`chat-mistralrs`.

## Layers (what we own vs what MLX owns)

```
mlx-rs        -> Array + math kernels + nn::{Linear,Rope,RmsNorm,Embedding} + fast::sdpa   [the framework]
src/model.rs  -> the architecture: compose those modules into a MiniCPM5/Llama decoder      [OURS]
src/cache.rs  -> KV cache: pre-allocated growable + rotating attention-sink window          [OURS]
src/generate.rs -> prefill + autoregressive decode loop (KV cache, EOS, max_tokens)         [OURS]
src/sampler.rs  -> on-device greedy / temperature / top-k / top-p sampling from logits      [OURS]
src/prompt.rs   -> chat templating (today: simple; becoming family-aware)                   [OURS]
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
# options: --system, --max-tokens, --temp, --top-k, --top-p, --model, --quantize,
#          --tokens-per-eval, --max-context, --sink-tokens, --seed
```

```bash
cargo run --release -- --model Qwen/Qwen3-0.6B --prompt "Explain RoPE in one sentence."
```

Default model: `openbmb/MiniCPM5-1B` (bf16). The 4-bit `openbmb/MiniCPM5-1B-MLX`
needs pre-quantized loading (TODO); for now load bf16 and pass `--quantize` to
quantize at runtime.

Supported families (one config-knob model, no per-family files yet):
`llama` / `minicpm` and `qwen3`. Qwen3 adds per-head QK-Norm, gated on
`model_type == "qwen3"`; everything else (GQA, SwiGLU, RoPE, head_dim,
rope_theta, EOS) comes from `config.json` / `generation_config.json`.

## KV cache

Decode used to slow down quadratically because every token re-`concatenate`d the
whole K/V tensor. `src/cache.rs` now pre-allocates the K/V buffers and grows them
in 256-token chunks, writing each new token into a slice instead of copying the
whole cache. With `--max-context` set (default 4096) it becomes a rotating
attention-sink cache: the first `--sink-tokens` (default 4) are pinned and the
rest of the window is overwritten ring-buffer style, so KV memory is bounded no
matter how long a generation (or thinking trace) runs. `--max-context 0` disables
the cap (unbounded growable). Because RoPE is relative, pinned sink tokens + the
recent window keep correct relative positions to the current query.

## Sampling

Sampling runs entirely on-device and lazily, so it composes into MLX's batched
eval graph rather than forcing a host sync per token: temperature → `categorical`,
top-k via `argpartition` masking, top-p (nucleus) via sorted `cumsum` + `which`.
Determinism comes from `--seed` (seeds MLX's RNG). Sampling method therefore does
not affect decode throughput.

## Perf (MiniCPM5-1B, M-series, decode tok/s)

| build / config | decode tok/s |
| --- | --- |
| debug | ~7.6 |
| release (bf16) | ~27 |
| release + `--quantize` (4-bit), tpe=8 | ~77 |
| release + `--quantize` (4-bit), tpe=16 | ~83 |

Decode is memory-bandwidth bound on the per-token weight read. Quantization is
the dominant lever (~3x). `--tokens-per-eval` batches MLX eval to amortize the
GPU<->host sync: nil effect on bf16 (sync is noise vs the weight read), ~+14% in
the 4-bit regime where decode is fast enough that sync starts to matter.

## Roadmap → chat-rs provider

Goal: `chat-mlx` becomes a `CompletionProvider` / `StreamProvider` for chat-rs.
The closest analog is `chat-mistralrs`, but that is a *thin* wrapper — mistral.rs
hides tokenization, chat-templating, sampling, and tool-call parsing, and it
*rejects* tools and structured output. Because chat-mlx owns the raw token loop,
this provider must implement everything mistral.rs hid. That parsing/templating
work is the bulk of the roadmap.

Dependency: `chat-core = { path = "../chat-rs/core", version = "0.4.2" }` (core is
0.4.2; 0.5.3 is the chat-rs umbrella version).

- **Phase 0 — this README.** Document the path and the GGUF study (below).
- **Phase 1 — provider skeleton + completion.** `lib + thin bin` split; typestate
  `ChatMlxBuilder` → `ChatMlxClient`; `impl CompletionProvider` mapping `Messages`
  → family-aware chat template → token loop → `ChatResponse` (text + usage). Tools
  and structured output rejected with a clear error for now (as mistralrs does).
- **Phase 2 — tool-call parser *families*.** A `ToolCallParser` trait with built-in
  families (Qwen/Hermes `<tool_call>{…}</tool_call>` first), **detected from model
  metadata** (`model_type`, chat template) when possible; otherwise the user
  supplies a **delimiter pattern** that we strip to extract the JSON inside, with a
  generic-JSON scrape as fallback. Tool declarations (`ToolDeclarations::json()`)
  are injected into the prompt in the family's format; parsed calls become
  `PartEnum::Tool` / `complete_reason = ToolCall`.
- **Phase 3 — structured output.** Inject the `schemars::Schema` as a system
  instruction (or a forced synthetic tool); parse the emitted JSON into
  `PartEnum::Structured`.
- **Phase 4 — streaming.** `impl StreamProvider` behind a `stream` feature,
  emitting `TextChunk` / `ReasoningChunk` / `ToolCall` deltas and a final
  `Done(ChatResponse)`.

## GGUF feasibility study — "can we build the blocks on GGUF?"

Open research item, not yet a commitment. Today chat-mlx loads **bf16
safetensors** (`resolve_weights` + `load_safetensors`) and can runtime-quantize
to 4-bit via MLX's group-affine `nn::quantize` (`--quantize`). The question is
whether we can instead load **GGUF** weights directly so the model blocks
(`Attention`, `Mlp`, the `MaybeQuantized` linears) consume them.

Findings:

- **mlx-rs 0.25 has no GGUF reader.** Its safe API loads only `.safetensors` /
  `.npy` (`mlx_rs::utils::io`). No `gguf` symbols appear in mlx-rs or `mlx-sys`.
  MLX's C++ core does have `mlx_load_gguf`, but it is **not bound** in the Rust
  stack.
- **MLX's quantization ≠ GGUF's.** MLX uses a *group-affine* scheme
  (`quantized_matmul`, `dequantize`, `MaybeQuantized` — already used in
  `model.rs`). GGUF uses **k-quants** (Q4_K_M, Q5_K, …) with a different block
  layout. There is no 1:1 mapping; a transcode is required.
- **The blocks are already quant-capable.** The bottleneck is purely the
  loader/transcode path, not the model definition.

Options to evaluate, roughly in increasing effort / payoff:

1. **Pure-Rust GGUF reader → dequantize to bf16/fp16 on load**, then feed the
   existing `Model` unchanged (e.g. a `gguf` crate or candle's GGUF reader).
   Lowest risk; unlocks the huge GGUF model ecosystem. Loses the on-disk-size
   benefit unless we re-quantize to MLX affine after loading.
2. **FFI to `mlx_load_gguf`** via `mlx-sys` if/when the binding is exposed
   upstream. Cleanest if it lands.
3. **Transcode GGUF k-quant blocks → MLX quantized layers** so `MaybeQuantized`
   consumes them directly with no dequant round-trip. Best memory story, most
   work (per-format block decoders).

Pragmatic first step if we pursue this: option 1, behind a `--gguf` flag, reusing
the existing block definitions.

## Status / TODO

- [x] First successful generate on bf16 weights (MiniCPM5-1B, greedy, coherent output)
- [x] top-p / top-k / temperature sampling (now on-device, seeded)
- [x] live streaming output (decode_stream)
- [x] tokens/sec timing (prefill vs decode)
- [x] second model family: Qwen3-0.6B (QK-Norm via config knob)
- [x] KV cache rewrite: growable + rotating attention-sink window (bounded memory)
- [x] on-device sampler (no per-token host sync)
- [ ] **chat-rs provider** (Phases 1–4 above)
- [ ] load pre-quantized `*-MLX` safetensors directly
- [ ] GGUF loader study (above)
- [ ] a custom fused Metal kernel experiment
