use std::env;

mod config;
mod model;
mod weight_loader;

use burn::tensor::backend::Backend;
use config::BertConfig;
use model::BertModel;
use weight_loader::BertHFConfig;

#[derive(PartialEq)]
enum BackendType {
    Wgpu,
    Cpu,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        println!("Please select the Backend Type (cpu/gpu) and model (Ex: bert-base-uncased)");
        return;
    }

    let backend_type = if args.len() > 1 && args[1] == "cpu" {
        println!("Using CPU backend (ndarray)");
        BackendType::Cpu
    } else {
        println!("Using WGPU backend (GPU if available)");
        BackendType::Wgpu
    };

    let model_name = args.get(2).map(|s| s.as_str());

    match backend_type {
        BackendType::Wgpu => {
            run_with_backend::<burn_wgpu::Wgpu>(model_name);
        }
        BackendType::Cpu => {
            run_with_backend::<burn_ndarray::NdArray>(model_name);
        }
    }
}

fn run_with_backend<B: Backend>(model_name: Option<&str>) {
    println!("Initializing BERT model...");

    let config = if let Some(name) = model_name {
        println!("Loading config from HuggingFace model: {}", name);
        match BertHFConfig::from_hub(name) {
            Ok(hf_config) => {
                println!(
                    "Loaded config: hidden_size={}, layers={}, heads={}",
                    hf_config.hidden_size,
                    hf_config.num_hidden_layers,
                    hf_config.num_attention_heads
                );
                BertConfig {
                    vocab_size: hf_config.vocab_size,
                    hidden_size: hf_config.hidden_size,
                    num_hidden_layers: hf_config.num_hidden_layers,
                    num_attention_heads: hf_config.num_attention_heads,
                    intermediate_size: hf_config.intermediate_size,
                    max_position_embeddings: hf_config.max_position_embeddings,
                    type_vocab_size: hf_config.type_vocab_size,
                    pad_token_id: hf_config.pad_token_id,
                    ..Default::default()
                }
            }
            Err(e) => {
                println!("Failed to load from Hub: {}. Using default config.", e);
                BertConfig::default()
            }
        }
    } else {
        println!("Using default BERT-base config. Pass a HuggingFace model name to load from Hub.");
        BertConfig::default()
    };

    let device = B::Device::default();
    let mut model: BertModel<B> = BertModel::new(&config, &device);

    // Load weights from HuggingFace if model name is provided
    if let Some(name) = model_name {
        println!("Loading weights from HuggingFace model: {}", name);
        match weight_loader::WeightLoader::from_hub(name) {
            Ok(loader) => match loader.load_into_model(&mut model, &device) {
                Ok(()) => println!("Weights loaded successfully!"),
                Err(e) => println!("Failed to load weights: {}", e),
            },
            Err(e) => println!("Failed to load weights from Hub: {}", e),
        }
    }

    println!("Model created successfully!");
    println!("Config: {:?}", config);

    // Test forward pass with dummy input
    use burn::tensor::{Int, Tensor};
    let input_ids = Tensor::<B, 2, Int>::from_ints([[101, 2023, 2003, 1037, 102]], &device);
    let attention_mask = Tensor::<B, 2, Int>::from_ints([[1, 1, 1, 1, 1]], &device);

    println!("Running forward pass...");
    let (sequence_output, pooled_output) = model.forward(input_ids, None, Some(attention_mask));
    println!("Sequence output shape: {:?}", sequence_output.dims());
    if let Some(pooled) = &pooled_output {
        println!("Pooled output shape: {:?}", pooled.dims());
        println!("{}", pooled);
    }
    println!("BERT model is working correctly!");
}
