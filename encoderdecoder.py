from transformers import FlaubertConfig, EncoderDecoderConfig, TFEncoderDecoderModel, CamembertConfig, TFFlaubertModel
import tensorflow as tf
import torch


#encoder = TFFlaubertModel.from_pretrained('flaubert/flaubert_base_cased')

#decoder_config = encoder.config.to_dict()
#decoder_config.update({"is_decoder": False})
#decoder = TFFlaubertModel.from_pretrained(decoder_config)
class Encoderdecoder(tf.keras.Model):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = TFFlaubertModel.from_pretrained("flaubert/flaubert_base_cased", from_pt=True)
        #encoder_config = FlaubertConfig()
        decoder_config = FlaubertConfig(is_encoder=False, emb_dim=768)
        self.decoder = TFFlaubertModel(decoder_config)
        #self.embedding = tf.keras.layers.Embedding(tokenizer.vocab_size, self.decoder.config.hidden_size)
        #self.config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
        #self.model = TFEncoderDecoderModel(encoder=self.encoder, decoder=self.decoder)
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(309)
        ])
        self.optimizer = tf.keras.optimizers.Adam()
        self.inputlayer = tf.keras.layers.InputLayer(input_shape=[None])
            
    def call(self, inputs, targets, training):
        #encoder_out = self.encoder(inputs, training=training)
       #decoder_input = self.embedding(targets)
        #decoder_out = self.decoder(encoder_out.last_hidden_state, training = training)
        x = self.inputlayer(inputs)
        x = self.encoder(x)
        output = self.decoder(None, inputs_embeds=x.last_hidden_state)
        logits = self.dense(output.last_hidden_state)
        
        
        if training:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(targets,logits)
            loss = tf.reduce_mean(loss)
            return loss
        else:
            return logits
            
        
        
