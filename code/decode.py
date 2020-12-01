import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
import os
import sys
from lstm_model import *
from encode import *
from numpy.random import randint

class decoder_lstm_model(lstm_model):
    def call(self, batched_source, batched_target, speaker_list, addressee_list, initial_state):
        """
        Runs the decoder model on one batch of source & target inputs

        :param batched_source: 2-D array of size (batch_size, sentence_max_length) that contains the batched, tokenized source scripts 
        :param batched_target: 2-D array of size (batch_size, sentence_max_length) that contains the batched, tokenized target scripts 
        :param speaker_list: 1-D array of size (batch_size) that contains speaker ids
        :param addressee_list: 1-D array of size (batch_size) that contains addressee ids

        :return loss: a tensor that contains the loss of this batch
        :       probs_list: a 2-D tensor that contains probabilities calculated for each column of 
                            words in target, shape = (sentence_max_length-1, batch_size, num_vocab)

        """
     
        source_ebd = tf.nn.embedding_lookup(self.source_embedding, batched_source)
        encoded_outputs, initial_state = self.encoder(source_ebd, initial_state=initial_state)
        losses = []

        for i in range(tf.shape(batched_target)[1]-1):
            target_ebd = tf.nn.embedding_lookup(self.target_embedding, batched_target[:, i]) # shape = (batch_size, 1, embed_size)
            probs, initial_state = self.decoder(encoded_outputs, initial_state, target_ebd, speaker_list, addressee_list)
            labels = tf.squeeze(batched_target[:, i+1]) 
            l = self.loss_func(probs, labels)
            losses.append(l)
        losses = tf.convert_to_tensor(losses) 
        loss = tf.reduce_sum(losses)

        # start with the first column of the batched target
        start=self.target_embedding(batched_target[:,0])
        init_probs,init_state= get_decoder_outputs(encoded_outputs, initial_state, start, speaker_list, addressee_list)
        probs_list = self.beam_search(start,init_probs, init_state, encoded_outputs)
        return losses, probs_list

    def self_beam_search(self,start,init_probs, init_state, encoded_outputs):
        """
        Performs beam search to produce the top N candidates

        :param start: 1-D array of size (sentence_max_length) that serves as the starting column of beam search
        :param probs:  float tensor, word prediction probabilities (batch_size, num_vocab)
        :param initial_state: hidden state from previous executions

        :return top_candidates: top N candidates represented as probabilities
        """

        # initialize first round of beam search
        init_probs = tf.nn.log_softmax(init_probs)# log_probs in the scoring function
        top_probs, candidates = tf.nn.top_k(init_probs, k = self.beam_size,sorted = False) #computes top-k entries in each row
        candidates = tf.expand_dims(candidates,axis = 2)
        
        # At each time step, examine all B × B possible next-word candidates
        for i in range (1, self.sentence_max_length):
            for b in range (self.beam_size):

                #find the top b candidates at this step
                probs,h,c= get_decoder_outputs(encoded_outputs,init_state,self.target_embedding(candidates[:,b,-1]),speaker_list,addressee_list)
                probs = tf.nn.log_softmax(probs)
                prob_k,candidates_k = tf.nn.top_k(probs,k = self.beam_size, sorted = False)
                
                #checks if each candidate ends in EOT
                prob_k *= (candidates[:,b]!=2).all(axis=1)
                prob_k = tf.expand_dims(prob_k,axis = 1)
                prob_k += tf.expand_dims(top_probs[:,b], axis = 1)

                cur_candidates = tf.expand_dims(candidates[:,b], axis = 1)
                candidates_k = tf.concat((cur_candidates,tf.expand_dims(candidates_k)),2)

                if b==0:
                    prob = prob_k
                    beam = candidates_k
                    hs = tf.expand_dims(h,axis = 2)
                    cs = tf.expand_dims(c,axis = 2)

                else:
                    prob = tf.concat((prob,prob_k),1)
                    beam = tf.concat((beam,candidates_k),1)
                    hs = tf.concat((hs,tf.expand_dims(h,axis = 2)),2)
                    cs = tf.concat((cs,tf.expand_dims(h,axis = 2)),2)

                # TODO: preserve the top-B unfinished hypotheses: 
                top_probs,top_idx = tf.nn.top_k(prob,k = self.beam_size,sorted = False)
                # candidates = beam[torch.arange(beam.size(0)).view(-1,1).expand(index.size()).contiguous().view(-1),
                #             index.view(-1),:].view(index.size(0),index.size(1),beam.size(2))
                # if (candidates == 2).any(axis=2).all():
                #     break

                h_list = hs.clone()
                c_list = cs.clone()

            # TODO: rank the candidates
            predicted_path = tf.argmax(top_probs,1)
            # top_candidates = candidates[torch.arange(candidates.size(0)),predicted_path,:]
            return top_candidates

    def get_decoder_outputs(self, encoded_outputs, initial_state, embedding, speaker, addressee):
        probs, state = self.decoder(encoded_outputs, initial_state, embedding, speaker, addressee)
        return probs, state

    def tf_beam_search(self, initial_state, beam_width):
        # TODO: figure out correct parameters

        print("set up tf_beam_search")
        # bsd = tf.contrib.seq2seq.BeamSearchDecoder(
        #     cell=self.decode_cell,
        #     embedding=self.target_embedding,
        #     end_token=1,
        #     initial_state=initial_state,
        #     beam_width=beam_width,
        #     output_layer=output_layer,
        #     length_penalty_weight=self.config.length_penalty_weight)
        # final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
        #     bsd,output_time_major=False,maximum_iterations=20)
        # top_candidates = final_outputs.ids

        top_candidates =  tf.nn.ctc_beam_search_decoder(input = initial_state, 
                sequence_length = self.sentence_max_length, beam_width=self.beam_size, 
                top_paths=self.beam_size)
        print("TOP CANDIDATES = ",top_candidates)

        # TODO: figure out correct dimensions 
        top_candidates = tf.transpose(top_candidates, perm=[0, 2, 1])
        return top_candidates 
    

    def decode(self):
        #TODO: turn id to words
        #TODO: write to output file
        pass

# TODO: figure out how to run this
if __name__ == '__main__':
    if sys.argv[1] == "SPEAKER":
        is_speaker = True
    elif sys.argv[1] == "SPEAKER_ADDRESSEE":
        is_speaker = False


    params = encoder_params()
    
    model = decoder_lstm_model(params, 20,20, is_speaker)

    #fake data
    probs = tf.random.normal([64, 15105], 3, 1, tf.float32)
    initial_state = tf.random.normal([512, 512], 6, 1, tf.float32)
    speaker_list = randint(1, 10, 64)
    addressee_list = randint(1, 10, 64)
    batched_source = (np.random.random ([64,20]) * 10).astype(int)
    batched_target = (np.random.random ([64,20]) * 10).astype(int)

    model.call(batched_source, batched_target, speaker_list, addressee_list, initial_state)
