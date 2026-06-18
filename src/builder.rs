use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use chat_core::error::{ChatError, ChatFailure};
use chat_core::types::provider_meta::ProviderMeta;

use crate::client::{MlxClient, StructuredMode};
use crate::loader::{self, Quantize};
use crate::parsers::tool::{self, Pattern, ToolFormat};

/// Typestate marker — no model id set yet, `build()` is not callable.
pub struct WithoutModel;
/// Typestate marker — model id is set, `build()` is callable.
pub struct WithModel;

/// Builder for [`MlxClient`].
///
/// `with_model` sets the Hugging Face repo id (e.g. `Qwen/Qwen3-0.6B`) and
/// transitions the builder so `.build()` becomes callable. `build()` is
/// synchronous: the HF download (hf-hub sync API) and MLX weight load both
/// block.
pub struct MlxBuilder<M = WithoutModel> {
    model_id: Option<String>,
    quant: Option<Quantize>,
    tokens_per_eval: usize,
    max_context: Option<i32>,
    sink_tokens: i32,
    format: Option<Arc<dyn ToolFormat>>,
    structured_mode: StructuredMode,
    description: Option<String>,
    _m: PhantomData<M>,
}

impl Default for MlxBuilder<WithoutModel> {
    fn default() -> Self {
        Self::new()
    }
}

impl MlxBuilder<WithoutModel> {
    pub fn new() -> Self {
        Self {
            model_id: None,
            quant: None,
            tokens_per_eval: 8,
            max_context: Some(4096),
            sink_tokens: 4,
            format: None,
            structured_mode: StructuredMode::default(),
            description: None,
            _m: PhantomData,
        }
    }

    /// Set the Hugging Face repo id. Transitions the builder so `.build()`
    /// becomes callable.
    pub fn with_model(self, id: impl Into<String>) -> MlxBuilder<WithModel> {
        MlxBuilder {
            model_id: Some(id.into()),
            quant: self.quant,
            tokens_per_eval: self.tokens_per_eval,
            max_context: self.max_context,
            sink_tokens: self.sink_tokens,
            format: self.format,
            structured_mode: self.structured_mode,
            description: self.description,
            _m: PhantomData,
        }
    }
}

impl<M> MlxBuilder<M> {
    /// Runtime-quantize the loaded fp weights to the given level (group size 64).
    pub fn with_quantize(mut self, quant: Quantize) -> Self {
        self.quant = Some(quant);
        self
    }

    /// Number of decode steps batched per MLX `eval` (amortises GPU<->host sync).
    pub fn with_tokens_per_eval(mut self, n: usize) -> Self {
        self.tokens_per_eval = n.max(1);
        self
    }

    /// Max tokens retained in the rotating KV cache. `0` disables the cap
    /// (unbounded growable cache).
    pub fn with_max_context(mut self, n: i32) -> Self {
        self.max_context = (n > 0).then_some(n);
        self
    }

    /// Leading tokens pinned as attention sinks when the KV window rotates.
    pub fn with_sink_tokens(mut self, n: i32) -> Self {
        self.sink_tokens = n.max(0);
        self
    }

    pub fn with_description(mut self, d: impl Into<String>) -> Self {
        self.description = Some(d.into());
        self
    }

    /// Override the auto-detected tool-call format with a specific one.
    pub fn with_tool_format(mut self, format: Arc<dyn ToolFormat>) -> Self {
        self.format = Some(format);
        self
    }

    /// Parse tool calls from output using custom delimiters: everything between
    /// `open` and `close` is treated as the call JSON. Overrides detection.
    pub fn with_tool_pattern(mut self, open: impl Into<String>, close: impl Into<String>) -> Self {
        self.format = Some(Arc::new(Pattern {
            open: open.into(),
            close: close.into(),
        }));
        self
    }

    /// Choose how structured output is enforced: prompt-and-parse
    /// ([`StructuredMode::Prompt`], default) or grammar-masked decoding
    /// ([`StructuredMode::Constrained`]).
    pub fn with_structured_mode(mut self, mode: StructuredMode) -> Self {
        self.structured_mode = mode;
        self
    }
}

impl MlxBuilder<WithModel> {
    /// Download (on first use) and load the model, returning a ready client.
    pub fn build(self) -> Result<MlxClient, ChatFailure> {
        let model_id = self.model_id.expect("with_model() sets model_id");

        let loaded = loader::load(&model_id, self.quant).map_err(|e| {
            ChatFailure::from_err(ChatError::Provider(format!(
                "chat-mlx failed to load {model_id}: {e}"
            )))
        })?;

        let format = self
            .format
            .unwrap_or_else(|| tool::detect(&loaded.model_type));

        let meta = Arc::new(ProviderMeta {
            description: self.description,
            ..Default::default()
        });

        Ok(MlxClient {
            model: Arc::new(Mutex::new(loaded.model)),
            tokenizer: Arc::new(loaded.tokenizer),
            args: loaded.args,
            eos: loaded.eos,
            model_id,
            tokens_per_eval: self.tokens_per_eval,
            max_context: self.max_context,
            sink_tokens: self.sink_tokens,
            format,
            template: Arc::new(loaded.chat_template),
            structured_mode: self.structured_mode,
            token_strings: Arc::new(std::sync::OnceLock::new()),
            meta,
        })
    }
}
