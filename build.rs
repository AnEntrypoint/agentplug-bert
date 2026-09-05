use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use half::f16;
use safetensors::tensor::{Dtype, SafeTensors, View};

struct ConvertedTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

impl View for ConvertedTensor {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.bytes)
    }

    fn data_len(&self) -> usize {
        self.bytes.len()
    }
}

fn f32_bytes_to_f16_bytes(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(4) {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        out.extend_from_slice(&f16::from_f32(value).to_le_bytes());
    }
    out
}

fn convert_weights_to_f16(source: &Path, target: &Path) {
    let bytes = std::fs::read(source).unwrap_or_else(|e| panic!("read {}: {e}", source.display()));
    let parsed = SafeTensors::deserialize(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", source.display()));
    let mut converted: BTreeMap<String, ConvertedTensor> = BTreeMap::new();
    for (name, tensor) in parsed.tensors() {
        let (dtype, bytes) = match tensor.dtype() {
            Dtype::F32 => (Dtype::F16, f32_bytes_to_f16_bytes(tensor.data())),
            other => (other, tensor.data().to_vec()),
        };
        converted.insert(name, ConvertedTensor { dtype, shape: tensor.shape().to_vec(), bytes });
    }
    safetensors::tensor::serialize_to_file(converted, &None, target).unwrap_or_else(|e| panic!("write {}: {e}", target.display()));
}

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let source = manifest_dir.join("weights").join("bge-small-en-v1.5.safetensors");
    let target = out_dir.join("bge-small-en-v1.5.f16.safetensors");
    println!("cargo:rerun-if-changed={}", source.display());
    convert_weights_to_f16(&source, &target);
    let shrunk = std::fs::metadata(&target).map(|m| m.len()).unwrap_or(0);
    let original = std::fs::metadata(&source).map(|m| m.len()).unwrap_or(0);
    println!("cargo:warning=bert weights converted to f16 for the wasm data segment: {original} -> {shrunk} bytes");
}
