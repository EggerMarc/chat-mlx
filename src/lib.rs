#![allow(clippy::result_large_err)]

pub mod api;
pub mod engine;
pub mod loader;
pub mod parsers;

mod builder;
mod client;

pub use builder::{MlxBuilder, WithModel, WithoutModel};
pub use client::{MlxClient, StructuredMode};
pub use loader::Quantize;
