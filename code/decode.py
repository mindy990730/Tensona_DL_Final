import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
import os
from lstm_model import *

class decoder_model(tf.keras.Model):
    def __init__(self, params, num_vocab, num_characters, is_speaker):

        super(decoder_model, self).__init__()
        self.is_speaker = is_speaker
        self.params = params
        hidden_sz = params.hidden_sz

        self.persona_embedding = tf.random.normal([num_characters, self.params.embed_size], stddev=.1, dtype=tf.float32)
        self.lstm_t_1 = tf.keras.layers.LSTM(hidden_sz, activation='relu', dropout=self.params.dropout, return_state=True, return_sequences=True)
        self.lstm_t_2 = tf.keras.layers.LSTM(hidden_sz, activation='relu', dropout=self.params.dropout, return_state=True, return_sequences=True)
        self.lstm_t_3 = tf.keras.layers.LSTM(hidden_sz, activation='relu', dropout=self.params.dropout, return_state=True, return_sequences=True)
        self.lstm_t_4= tf.keras.layers.LSTM(hidden_sz, activation='relu', return_state=True, return_sequences=True)
        self.dense = tf.keras.layers.Dense(num_vocab, activation='softmax')

        if not self.is_speaker:
            self.speaker_linear = tf.keras.layers.Dense(self.params.embed_size)
            self.addressee_linear = tf.keras.layers.Dense(self.params.embed_size)
            self.s_a_linear = tf.keras.layers.Dense(self.params.embed_size, activation='tanh')

        self.sentence_max_length = 20
        self.beam_size = 200
        

    def call(self, encoded_output, initial_state, word_embedding, speaker, addressee):
        """

        Forward pass of target LSTM that calculates the probablity of next words. 

        :param encoded_output: output of lstm_source, shape = (batch_size, sentence_max_length, embed_size)
        
        :param initial_state: last hidden_state of lstm_source or previous calls of lstm_target
        :param word_embedding: target embedding input, shape = (batch_size, 1, embed_size)
        :param speaker: list of speakers in this batch, shape = (batch_size)
        :param addressee: list of addressee in this batch, shape = (batch_size)
        :param is_speaker: True if speaker model, False if speaker-addressee model

        :return probs: probilities of vocab for this batch of words, shape = (batch_size, num_vocab)
        :       initial_state: the last hidden state of this call; to be used in the next call
        """
        # Concatenate embeddings 
        for i in range(self.params.batch_size): # in each sentence of this bacth
            word_embeddings_in_s = tf.slice(encoded_output, [i, 0, 0], [1, 1, self.params.embed_size]) # shape = (1, 1, embed_size)
            for s in range(1, tf.shape(encoded_output)[1], 1):
                w_e = tf.slice(encoded_output, [i, s, 0], [1, 1, self.params.embed_size])
                word_embeddings_in_s = tf.concat((word_embeddings_in_s, w_e), axis=2)
            # Now word_embeddings_in_s shape = (1, 1, sentence_max_length * embed_size)
            if i == 0:
                reshaped_encoder_output = word_embeddings_in_s
            else:
                reshaped_encoder_output = tf.concat((reshaped_encoder_output, word_embeddings_in_s), axis=0)

        # Now reshaped_encoder_output shape = (batch_size, 1, sentence_max_length * embed_size)
        word_embedding = tf.expand_dims(word_embedding, axis=1)
        lstm_t_input = tf.concat([reshaped_encoder_output, word_embedding], axis=2) # shape = (batch_size, 1, (1+sentence_max_length) * embed_size)

        speaker_embed = tf.nn.embedding_lookup(self.persona_embedding, speaker) # shape = (batch_size, embed_size)
        speaker_embed = tf.expand_dims(speaker_embed, axis=1) # shape = (batch_size, 1, embed_size)
        
        if self.is_speaker:
            lstm_t_input = tf.concat([lstm_t_input, speaker_embed], axis=2) # shape = (batch_size, 1, (2+sentence_max_length) * embed_size)
        else:
            addressee_embed = tf.nn.embedding_lookup(self.persona_embedding, addressee) # shape = (batch_size, embed_size)
            addressee_embed = tf.expand_dims(addressee_embed, axis=1)  # shape = (batch_size, 1, embed_size)
            
            addressee_embed = self.addressee_linear(addressee_embed) # shape = (batch_size, 1, embed_size)
            speaker_embed = self.speaker_linear(speaker_embed) # shape = (batch_size, 1, embed_size)
            
            combined_embed = tf.concat([speaker_embed, addressee_embed], axis=2) # shape = (batch_size, 1, 2 * embed_size)
            combined_embed = self.s_a_linear(combined_embed) # shape = (batch_size, 1, embed_size)
            lstm_t_input = tf.concat([lstm_t_input, combined_embed], axis=2) # shape = (batch_size, 1, (3+sentence_max_length) * embed_size)
        
        output, h, c = self.lstm_t_1(lstm_t_input, initial_state=initial_state)
        initial_state = (h,c)
        output, h, c = self.lstm_t_2(output, initial_state=initial_state)
        initial_state = (h,c)
        output, h, c = self.lstm_t_3(output, initial_state=initial_state)
        initial_state = (h,c)
        output, h, c = self.lstm_t_4(output, initial_state=initial_state)
        initial_state = (h,c)
        probs = self.dense(output) # shape = (batch_size, 1, num_vocab)
        probs = tf.squeeze(probs)
        return probs, initial_state

        # source_ebd = tf.nn.embedding_lookup(self.source_embedding, batched_source) # shape = (batch_size, sentence_max_length, embed_size)
        # probs, initial_state = self.encoder(source_ebd, initial_state=initial_state)
        # start=self.target_embedding(batched_target[:,0])

        # return self.beam_search(start,initial_state)

    def self_beam_search(self, probs, initial_state):
        """
        Performs beam search to produce the N-best lists

        :param initial_state:  a 3-D tensor that contains probabilities calculated for each column of words
                        in target, shape = (sentence_max_length-1, batch_size, num_vocab)
        :param stop_token: a token that signals the end of a single hypothesis
        :param max_decoding_length: maximumum length of generated candidates 
        :param beam_size: the scope of next-word candidates to search for 
        :return: N-best list
        """
        
        print("set up self_beam_search")

        beam_size = self.beam_size
        max_length = self.sentence_max_length
        #need encoder model
        probs=tf.nn.softmax(probs)
        h = initial_state[0]
        c = initial_state[1]

        # initialize list for adding hypothesis & their scores
        best_candidates = []
        top_probs, top_idx = tf.math.top_k(probs, beam_size, sorted = True) #computes top-k entries in each row
        beam_history = tf.expand_dims(top_idx,axis = 2)

        # At each time step, examine all B × B possible next-word candidates
        for i in range (1, max_length):
            for j in range (beam_size):

                # probs,h1,c1=self.decoder(context,h,c,self.tembed(beamHistory[:,k,-1]),speaker_label,addressee_label)
                # probs = tf.nn.log_softmax(probs)
                prob_k,beam_k = tf.math.top_k(probs,beam_size, sorted = True)
                curr_candidates = []

                # add all hypothesis ending with an EOS token to the N-best list.
                hyp = initial_state[i][j]
                if hyp[:-1] == 1:
                    best_candidates.append(hyp)
                else:
                    curr_candidates.append(hyp)

                # preserve the top-B unfinished hypotheses: 
                best_probs, idx = tf.math.top_k(curr_candidates, k=beam_size, sorted = True) #computes top-k entries in each row
                best_candidates.append(best_probs)

        # rerank the generated N-best list using a scoring function that linearly combines 
        # a length penalty and the log likelihood of the source given the target
        top_candidates = [([], 0)]
        for i in len(best_candidates):
            #TODO: figure out correct value & equation
            score = 0 # might be log_probs / length_penalty
            top_candidates.append((best_candidates[i],score))

        top_candidates = sorted(top_candidates, key = lambda v: v[1], reverse = True)
        return top_candidates

    def tf_beam_search1(self, initial_state,beam_width):
        # TODO: figure out correct parameters

        print("set up tf_beam_search1")
        bsd = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=self.decode_cell,
            embedding=self.target_embedding,
            end_token=1,
            initial_state=initial_state,
            beam_width=beam_width,
            output_layer=self.output_layer,
            length_penalty_weight=self.config.length_penalty_weight)

        final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            bsd,output_time_major=False,maximum_iterations=20)
        top_candidates = final_outputs.predicted_ids

        # TODO: figure out correct dimensions (beam search might add -1 when meet end token? and tf will delete start tokens)
        top_candidates = tf.transpose(top_candidates, perm=[0, 2, 1])
        
        pass 
    
    def tf_beam_search2(self,probs, seq_length, beam_width, top_path):
        print("set up tf_beam_search2")
        top_candidates =  tf.nn.ctc_beam_search_decoder(input = probs, 
                sequence_length = seq_length, beam_width=beam_width, 
                top_paths=top_path)
        return top_candidates

# TODO: figure out args
# if __name__ == '__main__':
# 	model = decode_model(args)
# 	model.decode()