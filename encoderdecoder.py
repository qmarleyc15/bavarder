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
        encoder_config = FlaubertConfig(vocab_size=tokenizer.vocab_size)
        self.encoder = TFFlaubertModel(encoder_config)
        self.decoder = tf.keras.layers.LSTM(309)
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Reshape((-1, 1)),
            tf.keras.layers.Dense(2*tokenizer.vocab_size, activation='relu'),
            tf.keras.layers.Dense(tokenizer.vocab_size),
        ])
        self.optimizer = tf.keras.optimizers.Adam()
            
    def call(self, inputs, targets, training):
        #encoder_out = self.encoder(inputs, training=training)
       #decoder_input = self.embedding(targets)
        #decoder_out = self.decoder(encoder_out.last_hidden_state, training = training)
        x = self.inputlayer(inputs)
        x = self.encoder(x)
        print('1')
        output = self.decoder(None, inputs_embeds=x.last_hidden_state)
        print('2')
        logits = self.dense(output.last_hidden_state)
        print('3')
        
        
        if training:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(targets,logits)
            loss = tf.reduce_mean(loss)
            print(loss)
            return loss
        else:
            return logits
            
        
        
