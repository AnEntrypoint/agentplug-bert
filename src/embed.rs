use std::sync::{Mutex, OnceLock};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, PositionEmbeddingType};
use tokenizers::Tokenizer;

use crate::abi::{elog, return_json};

static MODEL_SAFETENSORS: &[u8] = include_bytes!("../weights/bge-small-en-v1.5.safetensors");
static TOKENIZER_JSON: &[u8] = include_bytes!("../weights/bge-tokenizer.json");

const EMBED_MODEL_NAME: &str = "BAAI/bge-small-en-v1.5";
const EMBED_DIM: usize = 384;
const MAX_TOKENS: usize = 512;
const BGE_QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";
const QUERY_CACHE_CAP: usize = 64;
const QUERY_CACHE_TTL_MS: i64 = 600_000;
const PLAIN_CACHE_MAX_TEXT: usize = 4096;

#[link(wasm_import_module = "env")]
extern "C" {
    fn host_now_ms() -> u64;
    fn host_random_fill(ptr: *mut u8, len: u32) -> u32;
}

fn custom_getrandom(buf: &mut [u8]) -> Result<(), getrandom::Error> {
    let rc = unsafe { host_random_fill(buf.as_mut_ptr(), buf.len() as u32) };
    if rc == 0 {
        Err(getrandom::Error::UNSUPPORTED)
    } else {
        Ok(())
    }
}
getrandom::register_custom_getrandom!(custom_getrandom);

fn now_ms() -> i64 {
    unsafe { host_now_ms() as i64 }
}

struct EmbedCtx {
    tokenizer: Tokenizer,
    model: BertModel,
    device: Device,
}

static CTX: OnceLock<Result<EmbedCtx, String>> = OnceLock::new();

fn bge_small_config() -> Config {
    Config {
        vocab_size: 30522,
        hidden_size: 384,
        num_hidden_layers: 12,
        num_attention_heads: 12,
        intermediate_size: 1536,
        hidden_act: HiddenAct::Gelu,
        hidden_dropout_prob: 0.1,
        max_position_embeddings: 512,
        type_vocab_size: 2,
        initializer_range: 0.02,
        layer_norm_eps: 1e-12,
        pad_token_id: 0,
        position_embedding_type: PositionEmbeddingType::Absolute,
        use_cache: true,
        classifier_dropout: None,
        model_type: Some("bert".to_string()),
    }
}

fn init_ctx() -> Result<EmbedCtx, String> {
    // See rs-plugkit's embed.rs for the full root-cause writeup: gemm's
    // wasm32 SIMD dispatch is a runtime AtomicBool that defaults false
    // regardless of the -C target-feature=+simd128 compile flag; every real
    // host loading this module (wasmtime via agentplug-runner, or a browser
    // engine) has simd128 support, so force-enabling is safe unconditionally.
    gemm::set_wasm_simd128(true);

    let tokenizer = Tokenizer::from_bytes(TOKENIZER_JSON).map_err(|e| format!("tokenizer load: {e}"))?;
    let device = Device::Cpu;
    let vb = VarBuilder::from_slice_safetensors(MODEL_SAFETENSORS, DType::F32, &device)
        .map_err(|e| format!("varbuilder safetensors: {e}"))?;
    let config = bge_small_config();
    let model = BertModel::load(vb, &config).map_err(|e| format!("bert init: {e}"))?;
    elog(&format!("agentplug-bert: model loaded ({EMBED_MODEL_NAME}, dim={EMBED_DIM})"));
    Ok(EmbedCtx { tokenizer, model, device })
}

fn ctx() -> Result<&'static EmbedCtx, &'static str> {
    CTX.get_or_init(init_ctx).as_ref().map_err(|_| "embed init failed")
}

fn l2_normalize(v: &mut [f32]) {
    let mut s = 0f32;
    for x in v.iter() {
        s += *x * *x;
    }
    let n = s.sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

macro_rules! step {
    ($label:expr, $expr:expr) => {
        match $expr {
            Ok(v) => v,
            Err(e) => {
                elog(&format!("agentplug-bert embed_text step '{}' failed: {}", $label, e));
                return None;
            }
        }
    };
}

pub fn embed_text(text: &str) -> Option<Vec<f32>> {
    let cacheable = text.len() <= PLAIN_CACHE_MAX_TEXT;
    if cacheable {
        if let Some(v) = cache_get(&PLAIN_CACHE, text) {
            return Some(v);
        }
    }
    let v = embed_text_uncached(text)?;
    if cacheable {
        cache_put(&PLAIN_CACHE, text, &v);
    }
    Some(v)
}

fn embed_text_uncached(text: &str) -> Option<Vec<f32>> {
    let c = match ctx() {
        Ok(c) => c,
        Err(e) => {
            elog(&format!("agentplug-bert embed_text ctx() failed: {e} (text_len={})", text.len()));
            return None;
        }
    };

    let enc = step!("tokenizer.encode", c.tokenizer.encode(text, true));
    let mut ids: Vec<u32> = enc.get_ids().to_vec();
    let mut mask: Vec<u32> = enc.get_attention_mask().to_vec();
    if ids.len() > MAX_TOKENS {
        ids.truncate(MAX_TOKENS);
        mask.truncate(MAX_TOKENS);
    }
    let seq_len = ids.len();
    if seq_len == 0 {
        elog(&format!("agentplug-bert embed_text empty tokenization (text_len={})", text.len()));
        return None;
    }

    let ids_t = step!("Tensor::from_vec(ids)", Tensor::from_vec(ids.clone(), (1, seq_len), &c.device));
    let mask_t = step!("Tensor::from_vec(mask)", Tensor::from_vec(mask.clone(), (1, seq_len), &c.device));
    let token_type_ids = step!("Tensor::zeros(token_type_ids)", Tensor::zeros((1, seq_len), DType::U32, &c.device));

    let t0 = now_ms();
    let hidden_raw = step!("model.forward", c.model.forward(&ids_t, &token_type_ids, Some(&mask_t)));
    let total_ms = now_ms() - t0;
    if total_ms > 1000 {
        elog(&format!("agentplug-bert embed_text SLOW forward={total_ms}ms seq_len={seq_len} text_len={}", text.len()));
    }
    drop(ids_t);
    drop(token_type_ids);
    let hidden = step!("hidden.to_dtype(F32)", hidden_raw.to_dtype(DType::F32));
    drop(hidden_raw);

    let mask_f = step!("mask.to_dtype(F32)", mask_t.to_dtype(DType::F32));
    drop(mask_t);
    let mask_e = step!("mask.unsqueeze(2)", mask_f.unsqueeze(2));
    let masked = step!("hidden.broadcast_mul(mask)", hidden.broadcast_mul(&mask_e));
    drop(hidden);
    drop(mask_e);
    let sum = step!("masked.sum(1)", masked.sum(1));
    drop(masked);
    let denom_s = step!("mask.sum(1)", mask_f.sum(1));
    drop(mask_f);
    let denom = step!("denom.unsqueeze(1)", denom_s.unsqueeze(1));
    drop(denom_s);
    let pooled = step!("sum.broadcast_div(denom)", sum.broadcast_div(&denom));
    drop(sum);
    drop(denom);

    let flat_t = step!("pooled.flatten_all", pooled.flatten_all());
    drop(pooled);
    let flat: Vec<f32> = step!("flat.to_vec1", flat_t.to_vec1());
    drop(flat_t);
    if flat.len() != EMBED_DIM {
        elog(&format!("agentplug-bert embed_text dim mismatch: got={} expected={EMBED_DIM}", flat.len()));
        return None;
    }
    let mut out = flat;
    l2_normalize(&mut out);
    Some(out)
}

/// Sub-batch cap on total padded elements (batch_n * max_len), not just item
/// count -- BERT attention is O(batch_n * heads * max_len^2) per layer, and
/// an unbounded batch padded to its own longest sequence produced a real
/// ~1.8GB single allocation live-witnessed in the original rs-plugkit
/// integration. Ported verbatim as the fix, not rediscovered.
const MAX_SUBBATCH_ITEMS: usize = 32;
const MAX_SUBBATCH_PADDED_ELEMENTS: usize = 32 * 512;

pub fn embed_texts_batch(texts: &[String]) -> Vec<Option<Vec<f32>>> {
    if texts.is_empty() {
        return Vec::new();
    }
    if texts.len() == 1 {
        return vec![embed_text(&texts[0])];
    }

    let mut out: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
    let mut uncached_idx: Vec<usize> = Vec::new();
    for (i, t) in texts.iter().enumerate() {
        let cacheable = t.len() <= PLAIN_CACHE_MAX_TEXT;
        if cacheable {
            if let Some(v) = cache_get(&PLAIN_CACHE, t) {
                out[i] = Some(v);
                continue;
            }
        }
        uncached_idx.push(i);
    }
    if uncached_idx.is_empty() {
        return out;
    }

    let c = match ctx() {
        Ok(c) => c,
        Err(_) => {
            for &i in &uncached_idx {
                out[i] = embed_text(&texts[i]);
            }
            return out;
        }
    };

    let batch_texts: Vec<&str> = uncached_idx.iter().map(|&i| texts[i].as_str()).collect();
    let encodings = match c.tokenizer.encode_batch(batch_texts, true) {
        Ok(e) => e,
        Err(e) => {
            elog(&format!("agentplug-bert embed_texts_batch tokenizer.encode_batch failed: {e}; falling back per-item"));
            for &i in &uncached_idx {
                out[i] = embed_text(&texts[i]);
            }
            return out;
        }
    };

    let mut per_item_ids: Vec<Vec<u32>> = Vec::with_capacity(encodings.len());
    let mut per_item_mask: Vec<Vec<u32>> = Vec::with_capacity(encodings.len());
    for enc in &encodings {
        let mut ids = enc.get_ids().to_vec();
        let mut mask = enc.get_attention_mask().to_vec();
        if ids.len() > MAX_TOKENS {
            ids.truncate(MAX_TOKENS);
            mask.truncate(MAX_TOKENS);
        }
        per_item_ids.push(ids);
        per_item_mask.push(mask);
    }

    let n = per_item_ids.len();
    let mut start = 0usize;
    while start < n {
        let mut end = start;
        let mut sub_max_len = 1usize;
        while end < n && (end - start) < MAX_SUBBATCH_ITEMS {
            let candidate_max_len = sub_max_len.max(per_item_ids[end].len().max(1));
            let candidate_items = end - start + 1;
            if candidate_items * candidate_max_len > MAX_SUBBATCH_PADDED_ELEMENTS && candidate_items > 1 {
                break;
            }
            sub_max_len = candidate_max_len;
            end += 1;
        }
        if end == start {
            end = start + 1;
            sub_max_len = sub_max_len.max(per_item_ids[start].len().max(1));
        }

        let sub_ids = &per_item_ids[start..end];
        let sub_mask = &per_item_mask[start..end];
        let batch_n = sub_ids.len();
        let max_len = sub_max_len;

        let mut ids_flat: Vec<u32> = Vec::with_capacity(batch_n * max_len);
        let mut mask_flat: Vec<u32> = Vec::with_capacity(batch_n * max_len);
        for i in 0..batch_n {
            let ids = &sub_ids[i];
            let mask = &sub_mask[i];
            let len = ids.len();
            ids_flat.extend_from_slice(ids);
            ids_flat.extend(std::iter::repeat(0u32).take(max_len - len));
            mask_flat.extend_from_slice(mask);
            mask_flat.extend(std::iter::repeat(0u32).take(max_len - len));
        }

        let build = || -> Result<Vec<Option<Vec<f32>>>, String> {
            let ids_t = Tensor::from_vec(ids_flat.clone(), (batch_n, max_len), &c.device).map_err(|e| format!("{e}"))?;
            let mask_t = Tensor::from_vec(mask_flat.clone(), (batch_n, max_len), &c.device).map_err(|e| format!("{e}"))?;
            let token_type_ids = Tensor::zeros((batch_n, max_len), DType::U32, &c.device).map_err(|e| format!("{e}"))?;
            let hidden_raw = c.model.forward(&ids_t, &token_type_ids, Some(&mask_t)).map_err(|e| format!("{e}"))?;
            let hidden = hidden_raw.to_dtype(DType::F32).map_err(|e| format!("{e}"))?;
            let mask_f = mask_t.to_dtype(DType::F32).map_err(|e| format!("{e}"))?;
            let mask_e = mask_f.unsqueeze(2).map_err(|e| format!("{e}"))?;
            let masked = hidden.broadcast_mul(&mask_e).map_err(|e| format!("{e}"))?;
            let sum = masked.sum(1).map_err(|e| format!("{e}"))?;
            let denom_s = mask_f.sum(1).map_err(|e| format!("{e}"))?;
            let denom = denom_s.unsqueeze(1).map_err(|e| format!("{e}"))?;
            let pooled = sum.broadcast_div(&denom).map_err(|e| format!("{e}"))?;

            let mut results = Vec::with_capacity(batch_n);
            for row in 0..batch_n {
                let row_t = pooled.get(row).map_err(|e| format!("{e}"))?;
                let flat: Vec<f32> = row_t.to_vec1().map_err(|e| format!("{e}"))?;
                if flat.len() != EMBED_DIM {
                    results.push(None);
                    continue;
                }
                let mut v = flat;
                l2_normalize(&mut v);
                results.push(Some(v));
            }
            Ok(results)
        };

        let sub_uncached = &uncached_idx[start..end];
        match build() {
            Ok(results) => {
                for (j, &i) in sub_uncached.iter().enumerate() {
                    if let Some(v) = &results[j] {
                        let cacheable = texts[i].len() <= PLAIN_CACHE_MAX_TEXT;
                        if cacheable {
                            cache_put(&PLAIN_CACHE, &texts[i], v);
                        }
                    }
                    out[i] = results[j].clone();
                }
            }
            Err(e) => {
                elog(&format!("agentplug-bert embed_texts_batch sub-batch failed: {e}; falling back per-item"));
                for &i in sub_uncached {
                    out[i] = embed_text(&texts[i]);
                }
            }
        }
        start = end;
    }
    out
}

fn vec_to_json(v: Vec<f32>) -> serde_json::Value {
    serde_json::Value::Array(
        v.into_iter().map(|f| serde_json::Number::from_f64(f as f64).map(serde_json::Value::Number).unwrap_or(serde_json::Value::Null)).collect(),
    )
}

struct CacheEntry {
    key: String,
    embedding: Vec<f32>,
    ts_ms: i64,
}

static QUERY_CACHE: Mutex<Vec<CacheEntry>> = Mutex::new(Vec::new());
static PLAIN_CACHE: Mutex<Vec<CacheEntry>> = Mutex::new(Vec::new());

fn cache_get(cache: &Mutex<Vec<CacheEntry>>, key: &str) -> Option<Vec<f32>> {
    let mut guard = cache.lock().ok()?;
    let now = now_ms();
    guard.retain(|e| now - e.ts_ms < QUERY_CACHE_TTL_MS);
    let idx = guard.iter().position(|e| e.key == key)?;
    let entry = guard.remove(idx);
    let emb = entry.embedding.clone();
    guard.push(entry);
    Some(emb)
}

fn cache_put(cache: &Mutex<Vec<CacheEntry>>, key: &str, embedding: &[f32]) {
    let mut guard = match cache.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    let now = now_ms();
    guard.retain(|e| now - e.ts_ms < QUERY_CACHE_TTL_MS && e.key != key);
    while guard.len() >= QUERY_CACHE_CAP {
        guard.remove(0);
    }
    guard.push(CacheEntry { key: key.to_string(), embedding: embedding.to_vec(), ts_ms: now });
}

fn query_cache_get(key: &str) -> Option<Vec<f32>> {
    cache_get(&QUERY_CACHE, key)
}

fn query_cache_put(key: &str, embedding: &[f32]) {
    cache_put(&QUERY_CACHE, key, embedding)
}

pub fn handle_embed(body: &serde_json::Value) -> u64 {
    let text = body.get("text").and_then(|v| v.as_str()).unwrap_or("");
    let is_query = body.get("kind").and_then(|v| v.as_str()) == Some("query");
    if text.trim().is_empty() {
        return return_json(serde_json::json!({"ok": false, "error": "empty_text"}));
    }
    let result = if is_query {
        let trimmed = text.trim();
        if let Some(cached) = query_cache_get(trimmed) {
            Some(cached)
        } else {
            let prefixed = format!("{BGE_QUERY_PREFIX}{trimmed}");
            let v = embed_text(&prefixed);
            if let Some(v) = &v {
                query_cache_put(trimmed, v);
            }
            v
        }
    } else {
        embed_text(text)
    };
    match result {
        Some(v) => return_json(serde_json::json!({"ok": true, "embedding": vec_to_json(v), "dim": EMBED_DIM})),
        None => return_json(serde_json::json!({"ok": false, "error": "embed_failed"})),
    }
}

pub fn handle_embed_batch(body: &serde_json::Value) -> u64 {
    let texts: Vec<String> = body
        .get("texts")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        .unwrap_or_default();
    if texts.is_empty() {
        return return_json(serde_json::json!({"ok": true, "embeddings": []}));
    }
    let results = embed_texts_batch(&texts);
    let embeddings: Vec<serde_json::Value> =
        results.into_iter().map(|r| r.map(vec_to_json).unwrap_or(serde_json::Value::Null)).collect();
    return_json(serde_json::json!({"ok": true, "embeddings": embeddings}))
}
