# chat-mlx

Local-inference **[chat-rs](https://github.com/eggermarc/chat-rs) provider** (and CLI) for MiniCPM5 /
Llama / Qwen-family models on Apple Silicon, via [MLX](https://github.com/ml-explore/mlx).
It implements `CompletionProvider` + `StreamProvider`, so it drops into
`chat_core::ChatBuilder` and participates in the tool-calling, structured-output,
and streaming chat loop — the same surface as `chat-claude`, `chat-openai`, etc.

It owns the raw token loop (no daemon, no HTTP): tokenization, chat-templating,
sampling, KV cache, tool-call parsing, and JSON-constrained decoding all happen
in-process. The closest sibling is `chat-mistralrs`, but that is a thin wrapper
over mistral.rs and _rejects_ tools/structured output; here we implement the
parsing and templating it hides.

`lib + thin bin`: the `chat_mlx` library exposes the provider; the `chat-mlx`
binary is a small CLI over it. Depends on `chat-core` directly
(`{ version = "0.4.2", features = ["stream"] }`).

## Layout

```
src/engine/         the inference core (no chat-rs types)
  config.rs         parse HF config.json / generation_config.json -> ModelArgs
  model.rs          the architecture: Attention/Mlp/Decoder composed from mlx-rs nn modules
  cache.rs          KV cache: pre-allocated growable + rotating attention-sink window
  sampler.rs        on-device greedy / temperature / top-k / top-p sampling
  generate.rs       prefill + decode loop; plus generate_constrained (logit-masked)
  constraint.rs     LogitMask trait (constrained decoding hook)
  template.rs       ChatML rendering
src/loader.rs       HF download + weight load + tie_lm_head (shared by CLI and builder)
src/builder.rs      MlxBuilder (type-state) -> MlxClient
src/client.rs       MlxClient (Arc<Mutex<Model>>, Clone) + StructuredMode
src/api/            CompletionProvider / StreamProvider impls + request/response mapping
src/parsers/        reasoning (<think>), tool (families + stripper), json (validator+mask), structured
src/main.rs         CLI over the lib
```

mlx-rs ships the `nn` bricks (`Linear`, `Rope`, `RmsNorm`, `Embedding`, fast SDPA);
we compose the architecture. Struct field names in `model.rs` mirror HF tensor
keys so `load_safetensors` maps official weights with no manual remapping.

## Use as a provider

```rust
use chat_core::builder::ChatBuilder;
use chat_core::types::messages::{Messages, content};
use chat_core::parts;
use chat_mlx::MlxBuilder;

let client = MlxBuilder::new().with_model("Qwen/Qwen3-0.6B").build()?;
let mut chat = ChatBuilder::new().with_model(client).build();

let mut msgs = Messages::default();
msgs.push(content::from_user(parts!["Explain RoPE in one sentence."]));
let out = chat.complete(&mut msgs).await?; // ChatOutcome<ChatResponse>
```

Builder knobs: `with_quantize(bool)`, `with_max_context(i32)`, `with_sink_tokens`,
`with_tokens_per_eval`, `with_tool_format` / `with_tool_pattern`,
`with_structured_mode`.

### Tool calling

Register `tools-rs` tools on the `ChatBuilder` as usual. The provider advertises
them in the prompt and parses the model's calls back out. Formats are _families_:

- **Hermes/Qwen** (`<tool_call>{…}</tool_call>`) — auto-detected, the default.
- **Custom pattern** — `MlxBuilder::with_tool_pattern(open, close)`: we strip the
  delimiters and parse the JSON inside.

Parsed calls become `PartEnum::Tool` with `complete_reason = ToolCall`, which the
chat loop executes and feeds back. Streaming hides the in-progress call markup and
surfaces `StreamEvent::ToolCall` / `ToolResult` instead.

### Structured output

`ChatBuilder::with_structured_output::<T>()` (T: `JsonSchema + Deserialize`) works
two ways, selected by `MlxBuilder::with_structured_mode`:

- **`StructuredMode::Prompt`** (default) — inject the schema into the prompt, parse
  the emitted JSON; the chat loop retries on a parse miss.
- **`StructuredMode::Constrained`** — mask logits each decode step so only tokens
  keeping the output a valid-JSON prefix can be sampled; EOS only once a complete
  value is formed. Guarantees well-formed JSON (the schema's types/required fields
  are still validated on the typed deserialize). It enforces JSON _syntax_; full
  schema-level masking would need a grammar engine (llguidance/outlines).

Streaming structured output isn't expressible through chat-core's
`StreamProvider::stream` (it carries no schema), so the provider exposes a native
`MlxClient::stream_structured(messages, options, schema, mode)` that streams the
JSON live and emits a final `StreamEvent::Structured`.

## CLI

First build is heavy (compiles the MLX C++/Metal backend; needs Xcode CLT + cmake).

```bash
cargo run --release -- --prompt "Explain RoPE in one sentence."
# flags: --model --system --max-tokens --temp --top-k --top-p --quantize
#        --tokens-per-eval --max-context --sink-tokens --seed
cargo run --release -- --model Qwen/Qwen3-0.6B --prompt "…"
```

Default model: `openbmb/MiniCPM5-1B` (bf16). The 4-bit `*-MLX` repos need
pre-quantized loading (TODO); for now load bf16 and pass `--quantize` to quantize
at runtime.

### Examples

```bash
cargo run --release --example chat        -- --model Qwen/Qwen3-0.6B --temp 0.7 --top-k 40
cargo run --release --example structured   -- Qwen/Qwen2.5-1.5B-Instruct
```

- `chat` — interactive streaming REPL: multi-turn, gray `<think>` reasoning, the
  `get_weather` tool round-trip shown inline.
- `structured` — streaming structured output in both modes side by side.

## Supported model families

Architecture is config-driven (`config.json`), no per-family source files:

- **Llama / MiniCPM** — GQA, SwiGLU, RoPE; bias per the `attention_bias` flag.
- **Qwen2 / Qwen2.5** (`model_type == "qwen2"`) — adds **QKV bias** (output
  projection unbiased) and **tied embeddings** (no shipped `lm_head.weight`;
  `tie_lm_head` shares the input embedding).
- **Qwen3** (`model_type == "qwen3"`) — per-head **QK-Norm**, no QKV bias.

## KV cache

Decode used to slow down quadratically because every token re-`concatenate`d the
whole K/V tensor. `engine/cache.rs` now pre-allocates the K/V buffers and grows
them in 256-token chunks, writing each new token into a slice. With `--max-context`
(default 4096) it becomes a rotating attention-sink cache: the first
`--sink-tokens` (default 4) are pinned, the rest of the window rotates ring-buffer
style, so KV memory is bounded regardless of generation length. `--max-context 0`
disables the cap. Because RoPE is relative, pinned sinks + the recent window keep
correct relative positions to the current query.

## Sampling

Entirely on-device and lazy, so it composes into MLX's batched eval graph instead
of forcing a per-token host sync: temperature → `categorical`, top-k via
`argpartition` masking, top-p via sorted `cumsum` + `which`. Determinism from
`--seed`. Sampling method doesn't affect decode throughput. Constrained decoding
(`generate_constrained`) adds a per-step additive logit mask before sampling.

## Perf (MiniCPM5-1B, M-series, 256-token decode)

| build / config                         | decode tok/s |
| -------------------------------------- | ------------ |
| debug                                  | ~8           |
| release (bf16)                         | ~28          |
| release + `--quantize` (4-bit), tpe=8  | ~87          |
| release + `--quantize` (4-bit), tpe=16 | ~85          |

Decode is memory-bandwidth bound on the per-token weight read, so quantization is
the dominant lever (~3×). `--tokens-per-eval` is now ~flat: with the sampler moved
on-device there's no per-token GPU↔host sync left to amortize by batching.
Constrained decoding is slower (one token per eval + a vocab-sized mask each step)
— fine for short structured extractions.

## GGUF feasibility study — "can we build the blocks on GGUF?"

Open research item. Today chat-mlx loads **bf16 safetensors** and can
runtime-quantize to 4-bit via MLX's group-affine `nn::quantize` (`--quantize`).
Can we instead load **GGUF** directly so the blocks (`MaybeQuantized` linears)
consume it?

- **mlx-rs 0.25 has no GGUF reader** — only `.safetensors` / `.npy`. No `gguf`
  symbols in mlx-rs or `mlx-sys`; MLX's C++ `mlx_load_gguf` is unbound in Rust.
- **MLX's quantization ≠ GGUF's** — MLX is group-affine; GGUF is k-quants
  (Q4_K_M, …). No 1:1 mapping; a transcode is required.
- **The blocks are already quant-capable** — the gap is purely the loader/transcode.

Options, increasing effort/payoff: (1) pure-Rust GGUF reader → dequantize to
bf16/fp16 on load → feed the existing `Model` (lowest risk; e.g. candle's reader);
(2) FFI to `mlx_load_gguf` if upstream exposes it; (3) transcode k-quant blocks →
MLX quantized layers (best memory, most work). Pragmatic first step: (1) behind a
`--gguf` flag.

## Status

- [x] bf16 generate (MiniCPM5-1B / Qwen3 / Qwen2.5), coherent output
- [x] on-device top-p / top-k / temperature sampling (seeded)
- [x] KV cache rewrite: growable + rotating attention-sink window (bounded memory)
- [x] config-driven QKV bias + tied embeddings (Qwen2.5, Llama)
- [x] **chat-rs provider**: completion, streaming, tool families, structured output
- [x] constrained (valid-JSON) decoding via logit masking
- [ ] load pre-quantized `*-MLX` safetensors directly
- [ ] GGUF loader (study above)
- [ ] more families (Llama-3, Mistral) + a load matrix
- [ ] a custom fused Metal kernel experiment

```

```
