# RustBert

A BERT model implementation in Rust using the [Burn](https://burn.dev/) deep learning framework.

## Features

- Full BERT model architecture implementation
- Load pre-trained weights from HuggingFace Hub
- Support for both GPU (WGPU) and CPU (NdArray) backends
- Simple command-line interface

## Requirements

- Rust (latest stable)
- For GPU support: A Vulkan-compatible GPU (it can be used with AMD apus)

## Usage

### Run with CPU backend (default)

```bash
cargo run -- cpu
```

### Run with GPU backend (WGPU)

```bash
cargo run
```

### Load a specific HuggingFace model

```bash
cargo run -- cpu "bert-base-uncased"
cargo run -- "bert-base-uncased"  # GPU
```

Available models include:

- `bert-base-uncased`
- `bert-large-uncased`
- `bert-base-cased`
- And any other BERT model available on HuggingFace Hub

## Project Structure

- `src/config.rs` - BERT configuration
- `src/model.rs` - BERT model implementation
- `src/weight_loader.rs` - HuggingFace weight loading
- `src/main.rs` - CLI interface

## Implementation Details

The BERT model includes:

- Word, position, and token type embeddings
- Multi-head self-attention
- Intermediate and output layers with GELU activation
- Layer normalization and dropout

## License

MIT
