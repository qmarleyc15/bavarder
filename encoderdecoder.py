from transformers import FlaubertTokenizer, FlaubertModel
import tensorflow as tf


encoder = TFFlaubertModel('flaubert/flaubert_base_cased')

decoder_config = encoder.config.to_dict()
decoder_config.update({"is_decoder": False})
decoder = TFFlaubertModel(decoder_config)
class encoderdecoder():
    def __init__(self, tokenizer, encoder, decoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = tf.keras.layers.Embedding(tokenizer.vocab_size, decoder.config.hidden_size)
        self.dense = tf.keras.layers.Dense(tokenizer.vocab_size)
        
    def call(self, inputs, targets, training):
        encoder_out = self.encoder(inputs, targets)
        decoder_input = self.embedding(targets)
        decoder_out = self.decoder(encoder_out, decoder_input, training = training)
        logits = self.dense(decoder_out)
        
        if training:
            loss = tf.nn.sparse_softmax_cross_entropy(targets,logits)
            loss = tf.reduce_mean(loss)
            return loss
        else:
            return logits
            
        
        
