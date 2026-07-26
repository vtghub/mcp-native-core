//! Local sentence-embedding backend for `semantic_search`, built on `candle`
//! (a pure-Rust ML runtime) so this stays a single native binary with no
//! Python/ONNX dependency — consistent with the rest of this project's
//! "written entirely in Rust" positioning.
//!
//! The `Embedder` trait exists so `semantic_search`'s indexing/scoring logic
//! (and its tests) don't need to depend on an actual downloaded model: only
//! `BertEmbedder` talks to candle/hf-hub. Everything downstream — the
//! rotation/quantizer, the vector cache, the ranking — works against the
//! trait.

use std::path::Path;
use std::sync::Arc;

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use tokenizers::{Tokenizer, TruncationParams};

/// Produces a fixed-length, L2-normalized embedding for a piece of text.
/// Every consumer (`Rotation`, `QuantizedVector`, cosine scoring in
/// `semantic_search`) assumes the vectors this returns have unit norm.
pub trait Embedder: Send + Sync {
    fn dim(&self) -> usize;
    fn embed(&self, text: &str) -> Result<Vec<f32>, String>;
}

const DEFAULT_MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
// Comfortably under most small BERT encoders' position-embedding limit
// (512), while keeping a single forward pass cheap for the file/line-range
// chunks `semantic_search` embeds.
const MAX_TOKENS: usize = 256;

/// Reads `MCP_EMBEDDING_MODEL_DIR` (a local directory holding `config.json`,
/// `tokenizer.json`, and `model.safetensors` — for offline/air-gapped setups
/// where reaching the Hugging Face Hub isn't possible) if set; otherwise
/// downloads `MCP_EMBEDDING_MODEL_ID` (default
/// `sentence-transformers/all-MiniLM-L6-v2`) from the Hub on first use.
/// Either way, loading happens once (lazily, on the first `semantic_search`
/// call) and is reused for the rest of the process's lifetime.
pub async fn load_default_embedder() -> Result<Arc<dyn Embedder>, String> {
    let embedder = if let Ok(dir) = std::env::var("MCP_EMBEDDING_MODEL_DIR") {
        BertEmbedder::load_from_dir(Path::new(&dir))?
    } else {
        let model_id = std::env::var("MCP_EMBEDDING_MODEL_ID").unwrap_or_else(|_| DEFAULT_MODEL_ID.to_string());
        BertEmbedder::load_from_hub(&model_id).await?
    };
    Ok(Arc::new(embedder))
}

/// A BERT-family sentence encoder (mean-pooled, L2-normalized) running
/// locally on CPU via `candle`.
pub struct BertEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    dim: usize,
}

impl BertEmbedder {
    fn new(model: BertModel, mut tokenizer: Tokenizer, device: Device, dim: usize) -> Self {
        // `embed()` encodes one chunk of text at a time with padding
        // disabled, so truncation is the only length control needed — it
        // caps a chunk far longer than MAX_TOKENS instead of overflowing the
        // model's position embeddings.
        tokenizer
            .with_truncation(Some(TruncationParams { max_length: MAX_TOKENS, ..Default::default() }))
            .expect("static truncation config is always valid");
        tokenizer.with_padding(None);
        Self { model, tokenizer, device, dim }
    }

    /// Loads model files from a local directory (`MCP_EMBEDDING_MODEL_DIR`),
    /// bypassing the Hugging Face Hub entirely.
    pub fn load_from_dir(dir: &Path) -> Result<Self, String> {
        Self::load_from_files(&dir.join("config.json"), &dir.join("tokenizer.json"), &dir.join("model.safetensors"))
    }

    /// Downloads `config.json`, `tokenizer.json`, and `model.safetensors`
    /// for `model_id` from the Hugging Face Hub (cached locally by `hf-hub`
    /// after the first call) and loads them.
    pub async fn load_from_hub(model_id: &str) -> Result<Self, String> {
        let client = hf_hub::HFClient::new().map_err(|e| format!("Failed to create Hugging Face client: {e}"))?;
        let (owner, name) = hf_hub::split_id(model_id);
        let repo = client.model(owner, name);

        let config_path = repo
            .download_file()
            .filename("config.json")
            .send()
            .await
            .map_err(|e| format!("Failed to download config.json for {model_id}: {e}"))?;
        let tokenizer_path = repo
            .download_file()
            .filename("tokenizer.json")
            .send()
            .await
            .map_err(|e| format!("Failed to download tokenizer.json for {model_id}: {e}"))?;
        let weights_path = repo
            .download_file()
            .filename("model.safetensors")
            .send()
            .await
            .map_err(|e| format!("Failed to download model.safetensors for {model_id}: {e}"))?;

        Self::load_from_files(&config_path, &tokenizer_path, &weights_path)
    }

    fn load_from_files(config_path: &Path, tokenizer_path: &Path, weights_path: &Path) -> Result<Self, String> {
        let device = Device::Cpu;

        let config_str = std::fs::read_to_string(config_path).map_err(|e| format!("Failed to read {}: {e}", config_path.display()))?;
        let config: BertConfig =
            serde_json::from_str(&config_str).map_err(|e| format!("Failed to parse {}: {e}", config_path.display()))?;

        let tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|e| format!("Failed to load {}: {e}", tokenizer_path.display()))?;

        // Safety: `weights_path` is either a file `load_from_hub` just
        // downloaded itself or one the operator pointed
        // `MCP_EMBEDDING_MODEL_DIR` at directly — not attacker-controlled
        // input, the same trust assumption `fast_search`/`parse_structure`
        // make about `mmap`ing files the caller names.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path.to_path_buf()], DTYPE, &device)
                .map_err(|e| format!("Failed to load {}: {e}", weights_path.display()))?
        };
        let model = BertModel::load(vb, &config).map_err(|e| format!("Failed to build BERT model: {e}"))?;

        let dim = config.hidden_size;
        Ok(Self::new(model, tokenizer, device, dim))
    }
}

impl Embedder for BertEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        let encoding = self.tokenizer.encode(text, true).map_err(|e| format!("Tokenization failed: {e}"))?;
        let ids = encoding.get_ids();
        if ids.is_empty() {
            return Ok(vec![0.0; self.dim]);
        }

        let input_ids = Tensor::new(ids, &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| format!("Failed to build input tensor: {e}"))?;
        let token_type_ids = input_ids.zeros_like().map_err(|e| format!("Failed to build token-type tensor: {e}"))?;

        let output = self
            .model
            .forward(&input_ids, &token_type_ids, None)
            .map_err(|e| format!("BERT forward pass failed: {e}"))?;

        // Mean-pool over the sequence dimension. No attention mask needed:
        // `embed()` encodes one sequence at a time with padding disabled, so
        // every position in `output` is a real (non-padding) token.
        let (_batch, seq_len, _hidden) = output.dims3().map_err(|e| format!("Unexpected model output shape: {e}"))?;
        let pooled = (output.sum(1).map_err(|e| e.to_string())? / seq_len as f64).map_err(|e| e.to_string())?;

        let norm = pooled
            .sqr()
            .and_then(|t| t.sum_keepdim(1))
            .and_then(|t| t.sqrt())
            .map_err(|e| format!("Failed to compute embedding norm: {e}"))?;
        let normalized = pooled.broadcast_div(&norm).map_err(|e| format!("Failed to normalize embedding: {e}"))?;

        normalized
            .squeeze(0)
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| format!("Failed to extract embedding: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;
    use candle_transformers::models::bert::{HiddenAct, PositionEmbeddingType};
    use tokenizers::models::wordlevel::WordLevelBuilder;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    /// Builds a tiny (untrained, random-weight) BERT plus a matching
    /// whitespace/word-level tokenizer entirely in-process — no network, no
    /// bundled model file. This can't validate embedding *quality* (the
    /// weights are random), but it does exercise the real candle
    /// tokenize -> forward -> mean-pool -> normalize path this module ships,
    /// which is what network-restricted environments (this one included)
    /// can't otherwise verify against the real hosted model.
    fn tiny_embedder() -> BertEmbedder {
        let vocab_words = ["[UNK]", "[CLS]", "[SEP]", "fn", "main", "hello", "world", "fast", "search"];
        let vocab: ahash::AHashMap<String, u32> =
            vocab_words.iter().enumerate().map(|(i, w)| (w.to_string(), i as u32)).collect();
        let word_level = WordLevelBuilder::default()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .expect("static test vocab is valid");

        let mut tokenizer = Tokenizer::new(word_level);
        tokenizer.with_pre_tokenizer(Some(Whitespace {}));

        let config = BertConfig {
            vocab_size: vocab_words.len(),
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 16,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout_prob: 0.0,
            max_position_embeddings: 32,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: false,
            classifier_dropout: None,
            model_type: None,
        };

        let device = Device::Cpu;
        // A fresh VarMap fabricates randomly-initialized tensors for every
        // parameter `BertModel::load` asks for, standing in for a real
        // safetensors checkpoint — the shapes and forward-pass wiring are
        // real, only the weights are untrained.
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DTYPE, &device);
        let model = BertModel::load(vb, &config).expect("tiny config should build a valid model");

        BertEmbedder::new(model, tokenizer, device, config.hidden_size)
    }

    #[test]
    fn embed_returns_a_unit_normalized_vector_of_the_configured_dimension() {
        let embedder = tiny_embedder();
        let vec = embedder.embed("fn main hello world").expect("embedding a known-vocab sentence should succeed");

        assert_eq!(vec.len(), embedder.dim());
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "embedding should be L2-normalized, got norm {norm}");
    }

    #[test]
    fn embed_handles_out_of_vocabulary_text_via_the_unk_token() {
        let embedder = tiny_embedder();
        let vec = embedder.embed("completely unknown vocabulary here").expect("OOV text should still embed via [UNK]");
        assert_eq!(vec.len(), embedder.dim());
        assert!(vec.iter().all(|x| x.is_finite()));
    }

    // Not a real test: run once with `cargo test --release -- --ignored
    // write_tiny_model_fixture_to_disk` to materialize the same tiny model
    // this module's tests build in-memory as an on-disk
    // config.json/tokenizer.json/model.safetensors directory, so
    // `MCP_EMBEDDING_MODEL_DIR` can point the *actual running binary* at it
    // for a real end-to-end check with no network access — this sandbox
    // can't reach huggingface.co to exercise `load_from_hub` for real.
    #[test]
    #[ignore]
    fn write_tiny_model_fixture_to_disk() {
        let dir = std::env::var("MCP_TEST_FIXTURE_DIR").expect("set MCP_TEST_FIXTURE_DIR to an output directory");
        let dir = std::path::Path::new(&dir);
        std::fs::create_dir_all(dir).unwrap();

        let vocab_words = ["[UNK]", "[CLS]", "[SEP]", "fn", "main", "hello", "world", "fast", "search"];
        let vocab: ahash::AHashMap<String, u32> =
            vocab_words.iter().enumerate().map(|(i, w)| (w.to_string(), i as u32)).collect();
        let word_level = WordLevelBuilder::default().vocab(vocab).unk_token("[UNK]".to_string()).build().unwrap();
        let mut tokenizer = Tokenizer::new(word_level);
        tokenizer.with_pre_tokenizer(Some(Whitespace {}));
        tokenizer.save(dir.join("tokenizer.json"), true).unwrap();

        let config = serde_json::json!({
            "vocab_size": vocab_words.len(),
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 16,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "pad_token_id": 0,
            "position_embedding_type": "absolute",
            "use_cache": false,
            "classifier_dropout": null,
            "model_type": null
        });
        std::fs::write(dir.join("config.json"), serde_json::to_string_pretty(&config).unwrap()).unwrap();

        let bert_config = BertConfig {
            vocab_size: vocab_words.len(),
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 16,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout_prob: 0.0,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: false,
            classifier_dropout: None,
            model_type: None,
        };
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DTYPE, &device);
        BertModel::load(vb, &bert_config).unwrap();
        varmap.save(dir.join("model.safetensors")).unwrap();

        eprintln!("wrote fixture model to {}", dir.display());
    }
}
