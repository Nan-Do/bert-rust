use burn::{
    module::Module,
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear,
        LinearConfig,
        attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
    },
    tensor::{Device, Tensor, activation, backend::Backend},
};

use crate::config::BertConfig;

#[derive(Module, Debug)]
pub struct BertEmbeddings<B: Backend> {
    word_embeddings: Embedding<B>,
    position_embeddings: Embedding<B>,
    token_type_embeddings: Embedding<B>,
    layer_norm: LayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> BertEmbeddings<B> {
    pub fn new(config: &BertConfig, device: &Device<B>) -> Self {
        let word_embeddings =
            EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device);
        let position_embeddings =
            EmbeddingConfig::new(config.max_position_embeddings, config.hidden_size).init(device);
        let token_type_embeddings =
            EmbeddingConfig::new(config.type_vocab_size, config.hidden_size).init(device);
        let layer_norm = LayerNormConfig::new(config.hidden_size)
            .with_epsilon(config.layer_norm_eps)
            .init(device);
        let dropout = DropoutConfig::new(config.hidden_dropout_prob).init();

        Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
        }
    }

    pub fn word_embeddings(&self) -> &Embedding<B> {
        &self.word_embeddings
    }

    pub fn word_embeddings_mut(&mut self) -> &mut Embedding<B> {
        &mut self.word_embeddings
    }

    pub fn position_embeddings(&self) -> &Embedding<B> {
        &self.position_embeddings
    }

    pub fn position_embeddings_mut(&mut self) -> &mut Embedding<B> {
        &mut self.position_embeddings
    }

    pub fn token_type_embeddings(&self) -> &Embedding<B> {
        &self.token_type_embeddings
    }

    pub fn token_type_embeddings_mut(&mut self) -> &mut Embedding<B> {
        &mut self.token_type_embeddings
    }

    pub fn layer_norm(&self) -> &LayerNorm<B> {
        &self.layer_norm
    }

    pub fn layer_norm_mut(&mut self) -> &mut LayerNorm<B> {
        &mut self.layer_norm
    }

    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, burn::tensor::Int>,
        token_type_ids: Option<Tensor<B, 2, burn::tensor::Int>>,
    ) -> Tensor<B, 3> {
        let seq_length = input_ids.dims()[1];
        let device = input_ids.device();

        let word_embeds = self.word_embeddings.forward(input_ids);
        let position_ids = Tensor::arange(0..seq_length as i64, &device).reshape([1, seq_length]);
        let position_embeds = self.position_embeddings.forward(position_ids);

        let token_type_embeds = if let Some(ids) = token_type_ids {
            self.token_type_embeddings.forward(ids)
        } else {
            Tensor::zeros(
                [1, seq_length, self.word_embeddings.weight.dims()[1]],
                &device,
            )
        };

        let embeddings = word_embeds + position_embeds + token_type_embeds;
        let embeddings = self.layer_norm.forward(embeddings);
        self.dropout.forward(embeddings)
    }
}

#[derive(Module, Debug)]
pub struct BertSelfAttention<B: Backend> {
    attention: MultiHeadAttention<B>,
}

impl<B: Backend> BertSelfAttention<B> {
    pub fn new(config: &BertConfig, device: &Device<B>) -> Self {
        let d_model = config.hidden_size;

        let attention = MultiHeadAttentionConfig::new(d_model, config.num_attention_heads)
            .with_dropout(config.attention_probs_dropout_prob)
            .init(device);

        Self { attention }
    }

    pub fn attention_mut(&mut self) -> &mut MultiHeadAttention<B> {
        &mut self.attention
    }

    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 2, burn::tensor::Bool>>,
    ) -> Tensor<B, 3> {
        let input = MhaInput::new(hidden_states.clone(), hidden_states.clone(), hidden_states);

        let input = if let Some(mask) = attention_mask {
            input.mask_pad(mask)
        } else {
            input
        };

        self.attention.forward(input).context
    }
}

#[derive(Module, Debug)]
pub struct BertSelfOutput<B: Backend> {
    dense: Linear<B>,
    layer_norm: LayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> BertSelfOutput<B> {
    pub fn new(config: &BertConfig, device: &Device<B>) -> Self {
        let dense = LinearConfig::new(config.hidden_size, config.hidden_size).init(device);
        let layer_norm = LayerNormConfig::new(config.hidden_size)
            .with_epsilon(config.layer_norm_eps)
            .init(device);
        let dropout = DropoutConfig::new(config.hidden_dropout_prob).init();

        Self {
            dense,
            layer_norm,
            dropout,
        }
    }

    pub fn dense_mut(&mut self) -> &mut Linear<B> {
        &mut self.dense
    }

    pub fn layer_norm_mut(&mut self) -> &mut LayerNorm<B> {
        &mut self.layer_norm
    }

    pub fn forward(&self, hidden_states: Tensor<B, 3>, input_tensor: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden_states = self.dense.forward(hidden_states);
        let hidden_states = self.dropout.forward(hidden_states);
        self.layer_norm.forward(hidden_states + input_tensor)
    }
}

#[derive(Module, Debug)]
pub struct BertAttention<B: Backend> {
    self_attention: BertSelfAttention<B>,
    self_output: BertSelfOutput<B>,
}

impl<B: Backend> BertAttention<B> {
    pub fn new(config: &BertConfig, device: &Device<B>) -> Self {
        Self {
            self_attention: BertSelfAttention::new(config, device),
            self_output: BertSelfOutput::new(config, device),
        }
    }

    pub fn self_attention_mut(&mut self) -> &mut BertSelfAttention<B> {
        &mut self.self_attention
    }

    pub fn self_output_mut(&mut self) -> &mut BertSelfOutput<B> {
        &mut self.self_output
    }

    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 2, burn::tensor::Bool>>,
    ) -> Tensor<B, 3> {
        let self_outputs = self
            .self_attention
            .forward(hidden_states.clone(), attention_mask);
        self.self_output.forward(self_outputs, hidden_states)
    }
}

#[derive(Module, Debug)]
pub struct BertIntermediate<B: Backend> {
    dense: Linear<B>,
}

impl<B: Backend> BertIntermediate<B> {
    pub fn new(config: &BertConfig, device: &Device<B>) -> Self {
        let dense = LinearConfig::new(config.hidden_size, config.intermediate_size).init(device);
        Self { dense }
    }

    pub fn dense_mut(&mut self) -> &mut Linear<B> {
        &mut self.dense
    }

    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        activation::gelu(self.dense.forward(hidden_states))
    }
}

#[derive(Module, Debug)]
pub struct BertOutput<B: Backend> {
    dense: Linear<B>,
    layer_norm: LayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> BertOutput<B> {
    pub fn new(config: &BertConfig, device: &Device<B>) -> Self {
        let dense = LinearConfig::new(config.intermediate_size, config.hidden_size).init(device);
        let layer_norm = LayerNormConfig::new(config.hidden_size)
            .with_epsilon(config.layer_norm_eps)
            .init(device);
        let dropout = DropoutConfig::new(config.hidden_dropout_prob).init();

        Self {
            dense,
            layer_norm,
            dropout,
        }
    }

    pub fn dense_mut(&mut self) -> &mut Linear<B> {
        &mut self.dense
    }

    pub fn layer_norm_mut(&mut self) -> &mut LayerNorm<B> {
        &mut self.layer_norm
    }

    pub fn forward(&self, hidden_states: Tensor<B, 3>, input_tensor: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden_states = self.dense.forward(hidden_states);
        let hidden_states = self.dropout.forward(hidden_states);
        self.layer_norm.forward(hidden_states + input_tensor)
    }
}

#[derive(Module, Debug)]
pub struct BertLayer<B: Backend> {
    attention: BertAttention<B>,
    intermediate: BertIntermediate<B>,
    output: BertOutput<B>,
}

impl<B: Backend> BertLayer<B> {
    pub fn new(config: &BertConfig, device: &Device<B>) -> Self {
        Self {
            attention: BertAttention::new(config, device),
            intermediate: BertIntermediate::new(config, device),
            output: BertOutput::new(config, device),
        }
    }

    pub fn attention(&self) -> &BertAttention<B> {
        &self.attention
    }

    pub fn attention_mut(&mut self) -> &mut BertAttention<B> {
        &mut self.attention
    }

    pub fn intermediate(&self) -> &BertIntermediate<B> {
        &self.intermediate
    }

    pub fn intermediate_mut(&mut self) -> &mut BertIntermediate<B> {
        &mut self.intermediate
    }

    pub fn output(&self) -> &BertOutput<B> {
        &self.output
    }

    pub fn output_mut(&mut self) -> &mut BertOutput<B> {
        &mut self.output
    }

    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 2, burn::tensor::Bool>>,
    ) -> Tensor<B, 3> {
        let attention_output = self
            .attention
            .forward(hidden_states, attention_mask.clone());
        let intermediate_output = self.intermediate.forward(attention_output.clone());
        self.output.forward(intermediate_output, attention_output)
    }
}

#[derive(Module, Debug)]
pub struct BertEncoder<B: Backend> {
    layers: Vec<BertLayer<B>>,
}

impl<B: Backend> BertEncoder<B> {
    pub fn new(config: &BertConfig, device: &Device<B>) -> Self {
        let layers = (0..config.num_hidden_layers)
            .map(|_| BertLayer::new(config, device))
            .collect();
        Self { layers }
    }

    pub fn layers(&self) -> &Vec<BertLayer<B>> {
        &self.layers
    }

    pub fn layers_mut(&mut self) -> &mut Vec<BertLayer<B>> {
        &mut self.layers
    }

    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 2, burn::tensor::Bool>>,
    ) -> Tensor<B, 3> {
        let mut hidden_states = hidden_states;
        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states, attention_mask.clone());
        }
        hidden_states
    }
}

#[derive(Module, Debug)]
pub struct BertPooler<B: Backend> {
    dense: Linear<B>,
}

impl<B: Backend> BertPooler<B> {
    pub fn new(config: &BertConfig, device: &Device<B>) -> Self {
        let dense = LinearConfig::new(config.hidden_size, config.hidden_size).init(device);
        Self { dense }
    }

    pub fn dense_mut(&mut self) -> &mut Linear<B> {
        &mut self.dense
    }

    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 2> {
        let first_token_tensor = hidden_states.narrow(1, 0, 1).squeeze(1);
        let pooled_output = self.dense.forward(first_token_tensor);
        activation::tanh(pooled_output)
    }
}

#[derive(Module, Debug)]
pub struct BertModel<B: Backend> {
    embeddings: BertEmbeddings<B>,
    encoder: BertEncoder<B>,
    pooler: Option<BertPooler<B>>,
}

impl<B: Backend> BertModel<B> {
    pub fn new(config: &BertConfig, device: &Device<B>) -> Self {
        let pooler = if config.add_pooling_layer {
            Some(BertPooler::new(config, device))
        } else {
            None
        };

        Self {
            embeddings: BertEmbeddings::new(config, device),
            encoder: BertEncoder::new(config, device),
            pooler,
        }
    }

    pub fn embeddings(&self) -> &BertEmbeddings<B> {
        &self.embeddings
    }

    pub fn embeddings_mut(&mut self) -> &mut BertEmbeddings<B> {
        &mut self.embeddings
    }

    pub fn encoder(&self) -> &BertEncoder<B> {
        &self.encoder
    }

    pub fn encoder_mut(&mut self) -> &mut BertEncoder<B> {
        &mut self.encoder
    }

    pub fn pooler(&self) -> Option<&BertPooler<B>> {
        self.pooler.as_ref()
    }

    pub fn pooler_mut(&mut self) -> Option<&mut BertPooler<B>> {
        self.pooler.as_mut()
    }

    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, burn::tensor::Int>,
        token_type_ids: Option<Tensor<B, 2, burn::tensor::Int>>,
        attention_mask: Option<Tensor<B, 2, burn::tensor::Int>>,
    ) -> (Tensor<B, 3>, Option<Tensor<B, 2>>) {
        let embedding_output = self.embeddings.forward(input_ids, token_type_ids);

        let attention_mask = attention_mask.map(|mask| mask.bool());

        let sequence_output = self.encoder.forward(embedding_output, attention_mask);

        let pooled_output = if let Some(pooler) = &self.pooler {
            Some(pooler.forward(sequence_output.clone()))
        } else {
            None
        };

        (sequence_output, pooled_output)
    }
}
