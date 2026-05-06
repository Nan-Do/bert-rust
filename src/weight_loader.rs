use std::collections::HashMap;
use std::path::Path;

use burn::module::{Param, ParamId};
use burn::nn::{Embedding, LayerNorm, Linear};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use serde_json::Value;

#[derive(Debug, Clone)]
pub struct BertHFConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub pad_token_id: usize,
}

impl BertHFConfig {
    pub fn from_json(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let json: Value = serde_json::from_str(&content)?;

        Ok(Self {
            vocab_size: json["vocab_size"].as_u64().unwrap_or(30522) as usize,
            hidden_size: json["hidden_size"].as_u64().unwrap_or(768) as usize,
            num_hidden_layers: json["num_hidden_layers"].as_u64().unwrap_or(12) as usize,
            num_attention_heads: json["num_attention_heads"].as_u64().unwrap_or(12) as usize,
            intermediate_size: json["intermediate_size"].as_u64().unwrap_or(3072) as usize,
            max_position_embeddings: json["max_position_embeddings"].as_u64().unwrap_or(512)
                as usize,
            type_vocab_size: json["type_vocab_size"].as_u64().unwrap_or(2) as usize,
            pad_token_id: json["pad_token_id"].as_u64().unwrap_or(0) as usize,
        })
    }

    pub fn from_hub(model_name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(model_name.to_string());
        let config_path = repo.get("config.json")?;
        Self::from_json(&config_path)
    }
}

pub struct WeightLoader {
    // Maps weight name → (f32 values, shape)
    weights: HashMap<String, (Vec<f32>, Vec<usize>)>,
    // Prefix used in the weight file, e.g. "bert." for bert-base-uncased
    prefix: String,
}

impl WeightLoader {
    pub fn from_hub(model_name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(model_name.to_string());

        let weights_path = if let Ok(path) = repo.get("model.safetensors") {
            path
        } else if let Ok(path) = repo.get("pytorch_model.bin") {
            path
        } else {
            return Err("No weight file found".into());
        };

        Self::from_path(&weights_path)
    }

    pub fn from_path(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let weights = if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            load_safetensors(path)?
        } else {
            return Err("Only safetensors format is supported".into());
        };

        println!("Layers:");
        for key in weights.keys() {
            println!("{key}");
        }

        let prefix = if weights.contains_key("bert.embeddings.word_embeddings.weight") {
            "bert.".to_string()
        } else {
            String::new()
        };

        println!("Weight prefix detected: '{}'", prefix);
        Ok(Self { weights, prefix })
    }

    fn full_key(&self, name: &str) -> String {
        format!("{}{}", self.prefix, name)
    }

    fn load_tensor_2d<B: Backend>(
        &self,
        name: &str,
        device: &B::Device,
    ) -> Result<Tensor<B, 2>, Box<dyn std::error::Error>> {
        let key = self.full_key(name);
        let (values, shape) = self
            .weights
            .get(&key)
            .ok_or_else(|| format!("Weight '{}' not found", key))?;
        Ok(Tensor::<B, 2>::from_data(
            TensorData::new(values.clone(), shape.clone()),
            device,
        ))
    }

    fn load_tensor_1d<B: Backend>(
        &self,
        name: &str,
        device: &B::Device,
    ) -> Result<Tensor<B, 1>, Box<dyn std::error::Error>> {
        let key = self.full_key(name);
        let (values, shape) = self
            .weights
            .get(&key)
            .ok_or_else(|| format!("Weight '{}' not found", key))?;
        Ok(Tensor::<B, 1>::from_data(
            TensorData::new(values.clone(), shape.clone()),
            device,
        ))
    }

    pub fn load_into_model<B: Backend>(
        &self,
        model: &mut crate::model::BertModel<B>,
        device: &B::Device,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("Loading weights into model...");

        // Embeddings
        self.load_embedding(
            "embeddings.word_embeddings.weight",
            model.embeddings_mut().word_embeddings_mut(),
            device,
        )?;
        self.load_embedding(
            "embeddings.position_embeddings.weight",
            model.embeddings_mut().position_embeddings_mut(),
            device,
        )?;
        self.load_embedding(
            "embeddings.token_type_embeddings.weight",
            model.embeddings_mut().token_type_embeddings_mut(),
            device,
        )?;
        self.load_layer_norm(
            "embeddings.LayerNorm",
            model.embeddings_mut().layer_norm_mut(),
            device,
        )?;

        // Encoder layers
        let n_layers = model.encoder().layers().len();
        for i in 0..n_layers {
            let prefix = format!("encoder.layer.{}", i);
            self.load_attention_weights(
                &prefix,
                model.encoder_mut().layers_mut()[i].attention_mut(),
                device,
            )?;
            self.load_intermediate_weights(
                &prefix,
                model.encoder_mut().layers_mut()[i].intermediate_mut(),
                device,
            )?;
            self.load_output_weights(
                &prefix,
                model.encoder_mut().layers_mut()[i].output_mut(),
                device,
            )?;
        }

        // Pooler
        if let Some(pooler) = model.pooler_mut() {
            self.load_linear(
                "pooler.dense.weight",
                "pooler.dense.bias",
                pooler.dense_mut(),
                device,
            )?;
        }

        println!("Weights loaded successfully!");
        Ok(())
    }

    fn load_embedding<B: Backend>(
        &self,
        name: &str,
        embedding: &mut Embedding<B>,
        device: &B::Device,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match self.load_tensor_2d::<B>(name, device) {
            Ok(tensor) => {
                embedding.weight = Param::initialized(ParamId::new(), tensor);
                println!("  Loaded {}", name);
            }
            Err(e) => println!("  Warning: {}", e),
        }
        Ok(())
    }

    fn load_layer_norm<B: Backend>(
        &self,
        name: &str,
        layer_norm: &mut LayerNorm<B>,
        device: &B::Device,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match self.load_tensor_1d::<B>(&format!("{}.gamma", name), device) {
            Ok(tensor) => {
                layer_norm.gamma = Param::initialized(ParamId::new(), tensor);
                println!("  Loaded {}.gamma → gamma", name);
            }
            Err(e) => println!("  Warning: {}", e),
        }
        match self.load_tensor_1d::<B>(&format!("{}.beta", name), device) {
            Ok(tensor) => {
                layer_norm.beta = Param::initialized(ParamId::new(), tensor);
                println!("  Loaded {}.beta → beta", name);
            }
            Err(e) => println!("  Warning: {}", e),
        }
        Ok(())
    }

    // HuggingFace Linear stores weight as [d_out, d_in]; Burn uses [d_in, d_out], so transpose.
    fn load_linear<B: Backend>(
        &self,
        weight_name: &str,
        bias_name: &str,
        linear: &mut Linear<B>,
        device: &B::Device,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match self.load_tensor_2d::<B>(weight_name, device) {
            Ok(tensor) => {
                linear.weight = Param::initialized(ParamId::new(), tensor.transpose());
                println!("  Loaded {}", weight_name);
            }
            Err(e) => println!("  Warning: {}", e),
        }
        match self.load_tensor_1d::<B>(bias_name, device) {
            Ok(tensor) => {
                linear.bias = Some(Param::initialized(ParamId::new(), tensor));
                println!("  Loaded {}", bias_name);
            }
            Err(e) => println!("  Warning: {}", e),
        }
        Ok(())
    }

    fn load_attention_weights<B: Backend>(
        &self,
        prefix: &str,
        attention: &mut crate::model::BertAttention<B>,
        device: &B::Device,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let self_prefix = format!("{}.attention.self", prefix);
        let out_prefix = format!("{}.attention.output", prefix);

        // Q, K, V projections and output identity initialisation (block scope for the borrow)
        {
            let mha = attention.self_attention_mut().attention_mut();

            self.load_linear::<B>(
                &format!("{}.query.weight", self_prefix),
                &format!("{}.query.bias", self_prefix),
                &mut mha.query,
                device,
            )?;
            self.load_linear::<B>(
                &format!("{}.key.weight", self_prefix),
                &format!("{}.key.bias", self_prefix),
                &mut mha.key,
                device,
            )?;
            self.load_linear::<B>(
                &format!("{}.value.weight", self_prefix),
                &format!("{}.value.bias", self_prefix),
                &mut mha.value,
                device,
            )?;

            // Burn's MHA includes its own output projection which has no direct HF counterpart
            // because in HF BERT the output projection lives in BertSelfOutput.dense.
            // Set MHA.output to identity so BertSelfOutput.dense performs the actual projection.
            let d_model = mha.output.weight.val().dims()[0];
            let identity: Vec<f32> = (0..d_model * d_model)
                .map(|i| if i / d_model == i % d_model { 1.0 } else { 0.0 })
                .collect();
            mha.output.weight = Param::initialized(
                ParamId::new(),
                Tensor::<B, 2>::from_data(
                    TensorData::new(identity, vec![d_model, d_model]),
                    device,
                ),
            );
            mha.output.bias = Some(Param::initialized(
                ParamId::new(),
                Tensor::<B, 1>::zeros([d_model], device),
            ));
        }

        // BertSelfOutput: attention output dense + LayerNorm
        {
            let self_output = attention.self_output_mut();
            self.load_linear::<B>(
                &format!("{}.dense.weight", out_prefix),
                &format!("{}.dense.bias", out_prefix),
                self_output.dense_mut(),
                device,
            )?;
            self.load_layer_norm::<B>(
                &format!("{}.LayerNorm", out_prefix),
                self_output.layer_norm_mut(),
                device,
            )?;
        }

        Ok(())
    }

    fn load_intermediate_weights<B: Backend>(
        &self,
        prefix: &str,
        intermediate: &mut crate::model::BertIntermediate<B>,
        device: &B::Device,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.load_linear::<B>(
            &format!("{}.intermediate.dense.weight", prefix),
            &format!("{}.intermediate.dense.bias", prefix),
            intermediate.dense_mut(),
            device,
        )
    }

    fn load_output_weights<B: Backend>(
        &self,
        prefix: &str,
        output: &mut crate::model::BertOutput<B>,
        device: &B::Device,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.load_linear::<B>(
            &format!("{}.output.dense.weight", prefix),
            &format!("{}.output.dense.bias", prefix),
            output.dense_mut(),
            device,
        )?;
        self.load_layer_norm::<B>(
            &format!("{}.output.LayerNorm", prefix),
            output.layer_norm_mut(),
            device,
        )
    }
}

fn load_safetensors(
    path: &Path,
) -> Result<HashMap<String, (Vec<f32>, Vec<usize>)>, Box<dyn std::error::Error>> {
    let data = std::fs::read(path)?;
    let tensors = safetensors::SafeTensors::deserialize(&data)?;

    let mut weights = HashMap::new();
    for (name, tensor) in tensors.tensors() {
        let shape: Vec<usize> = tensor.shape().to_vec();
        let raw = tensor.data();

        let floats: Vec<f32> = match tensor.dtype() {
            safetensors::Dtype::F32 => raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            safetensors::Dtype::F16 => raw
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
            _ => continue,
        };
        weights.insert(name.to_string(), (floats, shape));
    }
    Ok(weights)
}

fn f16_to_f32(value: u16) -> f32 {
    let sign = ((value >> 15) & 1) as f32;
    let exp = ((value >> 10) & 0x1F) as i32;
    let frac = (value & 0x3FF) as f32;

    if exp == 0 {
        (sign * -2.0 + 1.0) * 2.0f32.powi(-14) * (frac / 1024.0)
    } else if exp == 31 {
        if frac == 0.0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (sign * -2.0 + 1.0) * 2.0f32.powi(exp - 15) * (1.0 + frac / 1024.0)
    }
}
