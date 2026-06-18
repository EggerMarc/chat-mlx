use async_trait::async_trait;

use chat_core::error::ChatFailure;
use chat_core::traits::CompletionProvider;
use chat_core::types::messages::Messages;
use chat_core::types::options::ChatOptions;
use chat_core::types::provider_meta::ProviderMeta;
use chat_core::types::response::ChatResponse;
use chat_core::types::tools::ToolDeclarations;

use crate::api::types::{error, request, response};
use crate::client::{MlxClient, StructuredMode};
use crate::engine::generate;
use crate::parsers::json::JsonConstraint;
use crate::parsers::{reasoning, structured};

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

        // For constrained structured output, build the JSON grammar mask up
        // front (decoding the vocab is independent of the model lock).
        let constrained =
            structured_output.is_some() && self.structured_mode == StructuredMode::Constrained;
        let mut json_constraint =
            constrained.then(|| JsonConstraint::new(self.token_strings(), self.eos.clone()));

        // Lock the model for the duration of the (synchronous) decode. No await
        // is held across the guard, so the future stays `Send`.
        let mut model = self
            .model
            .lock()
            .map_err(|_| error::provider("model mutex poisoned"))?;
        let mut cache = model.make_cache(self.max_context, self.sink_tokens);

        let stats = match json_constraint.as_mut() {
            Some(con) => generate::generate_constrained(
                &mut model,
                ids,
                prepared.max_tokens,
                &prepared.sampler,
                &self.eos,
                &mut cache,
                con,
                |_| {},
            ),
            None => generate::generate(
                &mut model,
                ids,
                prepared.max_tokens,
                &prepared.sampler,
                &self.eos,
                self.tokens_per_eval,
                &mut cache,
                |_| {},
            ),
        }
        .map_err(|e| error::provider(format!("generation failed: {e}")))?;
        drop(model);

        let raw = self
            .tokenizer
            .decode(&stats.tokens, true)
            .map_err(|e| error::invalid(format!("tokenizer decode: {e}")))?;

        let (reasoning_text, body) = reasoning::split(&raw);

        let mut structured = None;
        let mut calls = Vec::new();
        let mut text = String::new();

        if structured_output.is_some() {
            // Both modes parse the emitted JSON; constrained decoding has
            // already guaranteed it is well-formed. On a parse miss, hand back
            // the raw text so the chat loop's retry can take over.
            match structured::extract(&body) {
                Some(v) => structured = Some(v),
                None => text = body,
            }
        } else if tools.is_some() {
            let parsed = self.format.parse(&body);
            calls = parsed.calls;
            text = parsed.text;
        } else {
            text = body;
        }

        Ok(response::build(
            &self.model_id,
            reasoning_text,
            text,
            calls,
            structured,
            input_tokens,
            stats.tokens.len(),
            prepared.max_tokens,
        ))
    }

    fn metadata(&self) -> Option<&ProviderMeta> {
        Some(&self.meta)
    }
}
