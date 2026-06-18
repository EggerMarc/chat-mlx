use std::sync::{Arc, Mutex, OnceLock};

use chat_core::types::provider_meta::ProviderMeta;
use tokenizers::Tokenizer;

use crate::engine::config::ModelArgs;
use crate::engine::model::Model;
use crate::parsers::tool::ToolFormat;

/// How structured output is enforced.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StructuredMode {
    /// Inject the schema into the prompt and parse the JSON the model emits.
    /// Relies on the chat loop's retries if the model strays.
    #[default]
    Prompt,
    /// Mask logits during decoding so only tokens keeping the output a valid
    /// JSON prefix can be sampled — guarantees well-formed JSON (the schema's
    /// types/required fields are still validated on the typed deserialize).
    Constrained,
}

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
    pub(crate) format: Arc<dyn ToolFormat>,
    pub(crate) structured_mode: StructuredMode,
    /// Per-token surface strings, built lazily on first constrained decode and
    /// shared across clones. Index = token id, length = model vocab.
    pub(crate) token_strings: Arc<OnceLock<Arc<Vec<String>>>>,
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

    /// The decoded surface string of every token id (built once, then cached).
    /// Used by constrained decoding to test candidate tokens against the JSON
    /// grammar.
    pub(crate) fn token_strings(&self) -> Arc<Vec<String>> {
        self.token_strings
            .get_or_init(|| {
                let vocab = self.args.vocab_size.max(0) as u32;
                let strings = (0..vocab)
                    .map(|id| self.tokenizer.decode(&[id], false).unwrap_or_default())
                    .collect();
                Arc::new(strings)
            })
            .clone()
    }
}
