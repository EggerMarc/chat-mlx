// chat-core's `ChatFailure` is large; boxing every Result is not worth it. The
// other chat-rs providers (e.g. chat-mistralrs) silence this crate-wide too.
#![allow(clippy::result_large_err)]

pub mod api;
pub mod engine;
pub mod loader;
pub mod parsers;

mod builder;
mod client;

pub use builder::{MlxBuilder, WithModel, WithoutModel};
pub use client::{MlxClient, StructuredMode};
