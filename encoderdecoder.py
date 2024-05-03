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
        #decoder_config = FlaubertConfig(is_encoder=False)
        self.decoder = TFFlaubertModel.from_pretrained("flaubert/flaubert_base_cased", from_pt=True)
        #self.embedding = tf.keras.layers.Embedding(tokenizer.vocab_size, self.decoder.config.hidden_size)
        #self.config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
        #self.model = TFEncoderDecoderModel(self.config)
        self.dense = tf.keras.layers.Dense(tokenizer.vocab_size)
        self.optimizer = tf.keras.optimizers.Adam()
        self.inputlayer = tf.keras.layers.InputLayer(input_shape=[None])
            
    def call(self, inputs, targets, training):
        #encoder_out = self.encoder(inputs, training=training)
       #decoder_input = self.embedding(targets)
        #decoder_out = self.decoder(encoder_out.last_hidden_state, training = training)
        x = self.inputlayer(inputs)
        x = self.encoder(x)
        output = self.decoder(x)
        logits = self.dense(output)
        
        if training:
            loss = tf.nn.sparse_softmax_cross_entropy(targets,logits)
            loss = tf.reduce_mean(loss)
            return loss
        else:
            return logits
            
        
        