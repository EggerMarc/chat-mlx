use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use chat_core::error::{ChatError, ChatFailure};
use chat_core::types::provider_meta::ProviderMeta;

use crate::client::MlxClient;
use crate::loader;

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
    quantize: bool,
    tokens_per_eval: usize,
    max_context: Option<i32>,
    sink_tokens: i32,
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
            quantize: false,
            tokens_per_eval: 8,
            max_context: Some(4096),
            sink_tokens: 4,
            description: None,
            _m: PhantomData,
        }
    }

    /// Set the Hugging Face repo id. Transitions the builder so `.build()`
    /// becomes callable.
    pub fn with_model(self, id: impl Into<String>) -> MlxBuilder<WithModel> {
        MlxBuilder {
            model_id: Some(id.into()),
            quantize: self.quantize,
            tokens_per_eval: self.tokens_per_eval,
            max_context: self.max_context,
            sink_tokens: self.sink_tokens,
            description: self.description,
            _m: PhantomData,
        }
    }
}

impl<M> MlxBuilder<M> {
    /// Runtime-quantize the loaded fp weights to 4-bit (group size 64).
    pub fn with_quantize(mut self, quantize: bool) -> Self {
        self.quantize = quantize;
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
}

impl MlxBuilder<WithModel> {
    /// Download (on first use) and load the model, returning a ready client.
    pub fn build(self) -> Result<MlxClient, ChatFailure> {
        let model_id = self.model_id.expect("with_model() sets model_id");

        let loaded = loader::load(&model_id, self.quantize).map_err(|e| {
            ChatFailure::from_err(ChatError::Provider(format!(
                "chat-mlx failed to load {model_id}: {e}"
            )))
        })?;

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
            meta,
        })
    }
}
