use async_trait::async_trait;
use chat_core::error::ChatError;
use chat_core::traits::StreamProvider;
use chat_core::types::messages::Messages;
use chat_core::types::options::ChatOptions;
use chat_core::types::response::StreamEvent;
use chat_core::types::tools::ToolDeclarations;
use futures::StreamExt;
use futures::stream::BoxStream;
use tokio::sync::mpsc;

use crate::api::types::{request, response};
use crate::client::MlxClient;
use crate::engine::generate;
use crate::parsers::reasoning::{Chunk, ReasoningSplitter};

#[async_trait]
impl StreamProvider for MlxClient {
    async fn stream(
        &mut self,
        messages: &mut Messages,
        tool_declarations: Option<&dyn ToolDeclarations>,
        options: Option<&ChatOptions>,
    ) -> Result<BoxStream<'static, Result<StreamEvent, ChatError>>, ChatError> {
        let prepared = request::from_core(messages, options, None, tool_declarations.is_some())
            .map_err(|f| f.err)?;

        let model = self.model.clone();
        let tokenizer = self.tokenizer.clone();
        let eos = self.eos.clone();
        let model_id = self.model_id.clone();
        let tokens_per_eval = self.tokens_per_eval;
        let max_context = self.max_context;
        let sink_tokens = self.sink_tokens;
        let max_tokens = prepared.max_tokens;
        let sampler = prepared.sampler.clone();
        let prompt = prepared.prompt;

        let (tx, mut rx) = mpsc::unbounded_channel::<Result<StreamEvent, ChatError>>();

        // The decode loop is synchronous and blocking; run it off the async
        // runtime and forward events through the channel. The mutex guard is
        // confined to this blocking task.
        tokio::task::spawn_blocking(move || {
            let encoding = match tokenizer.encode(prompt, true) {
                Ok(e) => e,
                Err(e) => {
                    let _ = tx.send(Err(ChatError::InvalidResponse(format!(
                        "tokenizer encode: {e}"
                    ))));
                    return;
                }
            };
            let ids = encoding.get_ids();
            let input_tokens = ids.len();

            let mut model = match model.lock() {
                Ok(m) => m,
                Err(_) => {
                    let _ = tx.send(Err(ChatError::Provider("model mutex poisoned".into())));
                    return;
                }
            };
            let mut cache = model.make_cache(max_context, sink_tokens);
            let mut decoder = tokenizer.decode_stream(true);
            let mut splitter = ReasoningSplitter::new();

            let result = generate::generate(
                &mut model,
                ids,
                max_tokens,
                &sampler,
                &eos,
                tokens_per_eval,
                &mut cache,
                |id| {
                    if let Ok(Some(piece)) = decoder.step(id) {
                        for chunk in splitter.push(&piece) {
                            let _ = tx.send(Ok(to_event(chunk)));
                        }
                    }
                },
            );

            match result {
                Ok(stats) => {
                    for chunk in splitter.flush() {
                        let _ = tx.send(Ok(to_event(chunk)));
                    }
                    let resp = response::into_core_parts(
                        &model_id,
                        std::mem::take(&mut splitter.reasoning),
                        std::mem::take(&mut splitter.text),
                        input_tokens,
                        stats.tokens.len(),
                        max_tokens,
                    );
                    let _ = tx.send(Ok(StreamEvent::Done(resp)));
                }
                Err(e) => {
                    let _ = tx.send(Err(ChatError::Provider(format!("generation failed: {e}"))));
                }
            }
        });

        let s = async_stream::stream! {
            while let Some(ev) = rx.recv().await {
                yield ev;
            }
        };
        Ok(s.boxed())
    }
}

fn to_event(chunk: Chunk) -> StreamEvent {
    match chunk {
        Chunk::Text(s) => StreamEvent::TextChunk(s),
        Chunk::Reasoning(s) => StreamEvent::ReasoningChunk(s),
    }
}
