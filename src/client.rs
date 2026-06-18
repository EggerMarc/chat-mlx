use std::sync::{Arc, Mutex};

use chat_core::types::provider_meta::ProviderMeta;
use tokenizers::Tokenizer;

use crate::engine::config::ModelArgs;
use crate::engine::model::Model;

/// Local-inference client backed by a loaded MLX `Model`.
///
/// The model lives behind `Arc<Mutex<…>>`: mlx-rs `Array` is `Send` but not
/// `Sync`, and `Module::forward` needs `&mut`, so the mutex is what makes the
/// client `Sync` (required by `CompletionProvider`) and serialises decode
/// calls. Clones are cheap and share the same loaded weights.
#[derive(Clone)]
pub struct MlxClient {
    pub(crate) model: Arc<Mutex<Model>>,
    pub(crate) tokenizer: Arc<Tokenizer>,
    pub(crate) args: ModelArgs,
    pub(crate) eos: Vec<u32>,
    pub(crate) model_id: String,
    pub(crate) tokens_per_eval: usize,
    pub(crate) max_context: Option<i32>,
    pub(crate) sink_tokens: i32,
    pub(crate) meta: Arc<ProviderMeta>,
}

impl MlxClient {
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn args(&self) -> &ModelArgs {
        &self.args
    }

    pub fn provider_meta(&self) -> &ProviderMeta {
        &self.meta
    }
}
