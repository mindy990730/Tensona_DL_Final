import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
# from preprocess import get_data
from attention import Atten_Head


class Decoder_Model(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(Decoder_Model, self).__init__()

        self.embed_size = 0
        self.batch_size = 0
        self.hidden_size = 0

        self.beam_size = 200
        self.max_length = 20
        # self.personas = 

        self.embedding = self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(self.embed_size, return_sequences = True, return_state = True)
        self.dense = tf.keras.layers.Dense(self.hidden_size, activation='softmax')
        # self.attention = Attention
        
    def call(self, input, hidden, encoder_outputs):
        '''
        input           -> (1 x Batch Size)
        speakers        -> (1 x Batch Size, Addressees of inputs to Encoder)
        hidden          -> (Num. Layers * Num. Directions x Batch Size x Hidden Size)
        encoder_outputs -> (Max Sentence Length, Batch Size, Hidden Size)
        '''

        features = self.embedding(input)
        attention_weights = Atten_Head(hidden, encoder_outputs)
        output, _,_ = self.lstm(features)
        output = self.dense(output)

        return output, attention_weights


    def beam_search(self, max_decoding_length,beam_size, data):
        all_lists = []
        best_lists = []
        # for i in range(1,max_decoding_length):
		# 	for k in range(beam_size):
        #         if list ends in 'EOS':
        #           all_lists.append(list)
        #           preserve the top-B unfinished hypotheses

        # rerank the generated N-best list using a scoring function that linearly combines 
        # a length penalty and the log likelihood of the source given the target

        return best_lists

        

    def decode(self, inputs):
        pass