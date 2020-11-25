import tensorflow as tf
import numpy as np
from tensorflow.keras import Model


class Speaker_Model(tf.keras.Model):
    def __init__(self, num_speakers, vocab_size):
        """
        :param vocab_size: The number of speakers to be encoded and learned 
        """

        super(Model, self).__init__()

        self.num_speakers = num_speakers 
        self.vocab_size = vocab_size
        self.speaker_embed_size = 50 # TODO
        self.word_embed_size = 50 # TODO
        self.batch_size = 50 #TODO 
        self.learning_rate = 0.015
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        # TODO: initialize embeddings and forward pass weights (weights, biases)
        # Note: You can now use tf.keras.layers!
        # - use tf.keras.layers.Dense for feed forward layers: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        # - and use tf.keras.layers.GRU or tf.keras.layers.LSTM for your RNN 
        self.lstm = tf.keras.layers.LSTM(80, return_sequences=True, return_state=True)
        self.speaker_embeddings = tf.random.normal([self.num_speakers, self.speaker_embed_size], stddev=.1, dtype=tf.float32)
        self.word_embeddings = tf.random.normal([self.vocab_size, self.word_embed_size], stddev=.1, dtype=tf.float32)
        self.dense1 = tf.keras.layers.Dense(150, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.vocab_size, activation='softmax')

    def call(self, inputs, initial_state):
        embedding_layer = tf.nn.embedding_lookup(self.e, inputs) # 1 x embedding_sz
        output, final_memory_output, final_carry_output = self.lstm(embedding_layer, initial_state=initial_state) # 1 x rnn_size
        final_state = (final_memory_output, final_carry_output)
        output = self.dense1(output)
        output = self.dense2(output)
        return output,final_state
