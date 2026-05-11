//! LLM-tutorial: Educational Language Model Implementation
//!
//! A complete GPT-2 style transformer implemented from scratch in Rust
//! for educational purposes. Named after Shakespeare's witty fool from
//! *Twelfth Night*.
//!
//! # Modules
//!
//! - [`tokenizer`] - Byte Pair Encoding (BPE) tokenization
//! - [`tensor`] - Multi-dimensional arrays and operations
//! - [`model`] - GPT-2 model architecture (forward pass only)
//! - [`gpt2_trainable`] - Trainable GPT-2 with backward pass
//! - [`train`] - Data loading for training
//! - [`training_logger`] - Training metrics and logging
//!
//! # Example: Tokenization
//!
//! ```rust,no_run
//! use llm_tutorial::BPETokenizer;
//!
//! // Train a tokenizer
//! let text = std::fs::read_to_string("corpus.txt").unwrap();
//! let mut tokenizer = BPETokenizer::new(1024);
//! tokenizer.train(&text, 1024);
//!
//! // Encode and decode
//! let ids = tokenizer.encode("Hello, world!");
//! let decoded = tokenizer.decode(&ids);
//! assert_eq!(decoded, "Hello, world!");
//! ```
//!
//! # Example: Tensor Operations
//!
//! ```rust
//! use llm_tutorial::Tensor;
//!
//! // Create a 2x2 matrix
//! let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
//! let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
//!
//! // Matrix multiplication
//! let c = a.matmul(&b);
//! assert_eq!(c.shape, vec![2, 2]);
//! ```
//!
//! # Example: Model Architecture
//!
//! ```rust
//! use llm_tutorial::{GPT2, Config};
//!
//! // Create a tiny model
//! let config = Config::tiny(512); // 512 vocab size
//! let model = GPT2::new(&config);
//!
//! // Forward pass: tokens → logits
//! let tokens = vec![vec![1, 2, 3, 4]]; // batch_size=1, seq_len=4
//! let logits = model.forward(&tokens);
//! assert_eq!(logits.shape, vec![1, 4, 512]); // [batch, seq, vocab]
//! ```

mod model;
pub mod tensor;
pub mod tokenizer;

pub use crate::model::{Config, Embedding, GPT2};
pub use crate::tensor::Tensor;
pub use crate::tokenizer::{BPETokenizer, TokenizerStats};
