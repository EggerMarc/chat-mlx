use async_trait::async_trait;

use chat_core::error::ChatFailure;
use chat_core::traits::CompletionProvider;
use chat_core::types::messages::Messages;
use chat_core::types::options::ChatOptions;
use chat_core::types::provider_meta::ProviderMeta;
use chat_core::types::response::ChatResponse;
use chat_core::types::tools::ToolDeclarations;

use crate::api::types::{error, request, response};
use crate::client::MlxClient;
use crate::engine::generate;
use crate::parsers::reasoning;

#[async_trait]
impl CompletionProvider for MlxClient {
    async fn complete(
        &mut self,
        messages: &mut Messages,
        tool_declarations: Option<&dyn ToolDeclarations>,
        options: Option<&ChatOptions>,
        structured_output: Option<&schemars::Schema>,
    ) -> Result<ChatResponse, ChatFailure> {
        let tools = match tool_declarations {
            Some(d) => Some(
                d.json()
                    .map_err(|e| error::provider(format!("tool declarations: {e}")))?,
            ),
            None => None,
        };

        let prepared = request::from_core(
            messages,
            options,
            structured_output,
            tools.as_ref(),
            &*self.format,
        )?;

        let encoding = self
            .tokenizer
            .encode(prepared.prompt, true)
            .map_err(|e| error::invalid(format!("tokenizer encode: {e}")))?;
        let ids = encoding.get_ids();
        let input_tokens = ids.len();

        // Lock the model for the duration of the (synchronous) decode. No await
        // is held across the guard, so the future stays `Send`.
        let mut model = self
            .model
            .lock()
            .map_err(|_| error::provider("model mutex poisoned"))?;
        let mut cache = model.make_cache(self.max_context, self.sink_tokens);

        let stats = generate::generate(
            &mut model,
            ids,
            prepared.max_tokens,
            &prepared.sampler,
            &self.eos,
            self.tokens_per_eval,
            &mut cache,
            |_| {},
        )
        .map_err(|e| error::provider(format!("generation failed: {e}")))?;
        drop(model);

        let raw = self
            .tokenizer
            .decode(&stats.tokens, true)
            .map_err(|e| error::invalid(format!("tokenizer decode: {e}")))?;

        let (reasoning_text, body) = reasoning::split(&raw);
        let (calls, text) = if tools.is_some() {
            let parsed = self.format.parse(&body);
            (parsed.calls, parsed.text)
        } else {
            (Vec::new(), body)
        };

        Ok(response::build(
            &self.model_id,
            reasoning_text,
            text,
            calls,
            input_tokens,
            stats.tokens.len(),
            prepared.max_tokens,
        ))
    }

    fn metadata(&self) -> Option<&ProviderMeta> {
        Some(&self.meta)
    }
}
