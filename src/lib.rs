#![cfg(target_arch = "wasm32")]

mod abi;
mod embed;

pub use abi::{plugin_call, plugkit_alloc, plugkit_free};
