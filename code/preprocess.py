import json
import pandas as pd
import numpy as np
import tensorflow as tf


class Data():
    def __init__(self, params):
        self.params = params
        self.folder_path = params.data_folder_path # ../data
        self.output_file_name = params.friends_output_file_name # 'friends_transcripts.csv'
        self.EOS = 0 
        self.EOT = 1 
        self.vocab_dict = None
        self.character_dict = None

    def friends_tsv(self, num_seasons):
        """
        To turn Friends json files into a dictionary and a csv file 
        
        :param num_seasons: The number of seasons of Friends transcripts to read (should be 1 ~ 10)
        
        :return friends_data: a dictionary containing seasons, episodes, speakers, scripts
        """
        try:
            if num_seasons < 1:
                raise ValueError("num_seasons must be 1 ~ 10!")
        except:
            exit('num_seasons invalid! ')

        friends_data = dict(season_id=[],
                            episode_id=[],
                            scene_id=[],
                            utterance_id=[],
                            speaker=[],
                            tokens=[],
                            transcript=[]
                            )
        print('Loading seasons...')
        for season_index in range(1, num_seasons):
            season_index = '0%d'%season_index if season_index <10 else str(season_index)
            json_file_path = self.folder_path + '/friends_season_' + str(season_index)+'.json'
            with open(json_file_path, 'r') as json_file:
                season = json.load(json_file)
                season_id = season['season_id']
                # read each episode
                for episode in season['episodes']:
                    episode_id = episode['episode_id']
                    # read each scene
                    for scene in episode['scenes']:
                        scene_id = scene['scene_id']
                    # read each utterance
                        for utterance in scene['utterances']:
                            utterance_id = utterance['utterance_id']
                            speaker = utterance['speakers'][0] if utterance['speakers'] else 'unknown'
                            friends_data['season_id'].append(season_id)
                            friends_data['episode_id'].append(episode_id.split('_')[-1])
                            friends_data['scene_id'].append(scene_id.split('_')[-1])
                            friends_data['utterance_id'].append(utterance_id.split('_')[-1])
                            friends_data['speaker'].append(speaker)
                            friends_data['tokens'].append(utterance['tokens'])
                            friends_data['transcript'].append(utterance['transcript'])
        # save dicitonary to data frame
        friends_df = pd.DataFrame(friends_data)

        # save data frame to .tsv
        friends_df.to_csv(self.output_file_name, sep='\t', index=False)
        print('File saved in ' + self.output_file_name +' !')
        # show sample
        friends_df.head()
        return friends_data

    def cleanup_and_build_dict(self, friends_data):
        """
        To clean up the data into an 2-D array of num_lines * [speaker_id, speaker_scripts, addressee_id, addressee_scripts] (shape = num_lines * 4)
        1) Account for end of season/episode/scene
        2) Account for unknown speakers & empty lines
        
        :param friends_data: a dict containing info of 'Friends', output of friends_csv()
        
        :return spkr_adrs_list: 2-D array with each row = [speaker_id, speaker_scripts, addressee_id, addressee_scripts]
        
        """
        num_lines = len(friends_data['season_id'])
        spkr_adrs_list = []

        character_dict = dict()
        num_characters = 0

        vocab_dict = dict()
        vocab_dict['EOS'] = self.EOS
        vocab_dict['EOT'] = self.EOT
        num_vocab = 2 # EOS=0, EOT=1

        scene_id = friends_data['scene_id']
        speakers = friends_data['speaker']
        tokens = friends_data['tokens']
        
        for i in range(num_lines-1):
            
            # If reached an empty line or the end of a conversation --> continue to the next line
            if len(tokens[i]) == 0 or len(tokens[i+1]) == 0:
                continue  
            
            # If reached the end of a scene --> continue to the next line
            if scene_id[i] != scene_id[i+1]:
                continue 
            
            # tokenize speaker & addressee, add to character_dict
            speaker_name = speakers[i]
            addressee_name = speakers[i+1]
            if speaker_name not in character_dict.keys():
                character_dict[speaker_name] = num_characters
                num_characters += 1
            if addressee_name not in character_dict.keys():
                character_dict[addressee_name] = num_characters
                num_characters += 1

            # tokenize speaker scripts
            speaker_scripts = tokens[i]
            tokenized_speaker_scripts = []
            for sentence in speaker_scripts: 
                tokenized_speaker_scripts.extend(sentence)
            # only keep sentence_max_length number of words at most
            if len(tokenized_speaker_scripts) > self.params.sentence_max_length:
                tokenized_speaker_scripts = tokenized_speaker_scripts[:self.params.sentence_max_length]
            for x in range(len(tokenized_speaker_scripts)):
                word = tokenized_speaker_scripts[x]
                if word not in vocab_dict.keys():
                    vocab_dict[word] = num_vocab
                    num_vocab += 1
                tokenized_speaker_scripts[x] = vocab_dict[word] # tokenize word --> id
            if len(tokenized_speaker_scripts) < self.params.sentence_max_length:
                extension = np.zeros(self.params.sentence_max_length-len(tokenized_speaker_scripts))
                tokenized_speaker_scripts.extend(extension)
            
            # tokenize addressee scripts
            addressee_scripts = tokens[i+1]
            tokenized_addressee_scripts = []
            for sentence in addressee_scripts: 
                tokenized_addressee_scripts.extend(sentence)
            if len(tokenized_addressee_scripts) > self.params.sentence_max_length-2: # account for EOS & EOT
                # only keep sentence_max_length number of words at most
                tokenized_addressee_scripts = tokenized_addressee_scripts[:self.params.sentence_max_length]
            for j in range(len(tokenized_addressee_scripts)):
                word = tokenized_addressee_scripts[j]
                if word not in vocab_dict.keys():
                    vocab_dict[word] = num_vocab
                    num_vocab += 1
                tokenized_addressee_scripts[j] = vocab_dict[word]
            if len(tokenized_addressee_scripts) < self.params.sentence_max_length:
                extension = np.zeros(self.params.sentence_max_length-len(tokenized_addressee_scripts))
                tokenized_addressee_scripts.extend(extension)
            
            # Add EOS & EOT to tokenized_addressee_scripts
            tokenized_addressee_scripts = [self.EOS] + tokenized_addressee_scripts + [self.EOT] 
            spkr_adrs_list.append([character_dict[speaker_name], tokenized_speaker_scripts, character_dict[addressee_name], tokenized_addressee_scripts])
        
        # Update hyperparameters of vocab & character dict
        self.vocab_dict = vocab_dict
        self.character_dict = character_dict
        # print()
        return spkr_adrs_list

    def train_test_split(self, all_data, p_split=0.9):
        """
        Shuffle and split the data into train_data and test_data 
        :param all_data: 2-D array with each row = [speaker_id, speaker_scripts, addressee_id, addressee_scripts]
        :param p_split: percentage of data to be train_data (default=0.9)
        :return train_data: 2-D tensor with each row = [speaker_id, speaker_scripts, addressee_id, addressee_scripts]
        :       test_data: 2-D tensor with each row = [speaker_id, speaker_scripts, addressee_id, addressee_scripts]
        """
        # Convert all data into speakers, speaker_scripts, addressee, addressee_scripts
        speakers = []
        speaker_scripts = []
        addressee = []
        addressee_scripts = []
        # print(all_data, '\n\n')
        for row in range(len(all_data)):
            speakers.append(all_data[row][0])
            speaker_scripts.append(all_data[row][1])
            addressee.append(all_data[row][2])
            addressee_scripts.append(all_data[row][3])
        speakers = tf.squeeze(speakers)
        speaker_scripts = tf.squeeze(speaker_scripts)
        addressee = tf.squeeze(addressee)
        addressee_scripts = tf.squeeze(addressee_scripts)
        # Shuffle data
        indices = tf.range(0, len(all_data), 1)
        indices = tf.random.shuffle(indices)
        # ==================================================================
        # all data of mixed type & cannot be converted to tensor (change into speaker, scripts here and shuffle each separately)
        # all_data = tf.convert_to_tensor(all_data)
        speakers = tf.gather(speakers, indices, axis=0)
        speaker_scripts = tf.gather(speaker_scripts, indices, axis=0)
        addressee = tf.gather(addressee, indices, axis=0)
        addressee_scripts = tf.gather(addressee_scripts, indices, axis=0)
    
        # Split into train and test 
        num_train = int(len(speakers) * 0.9)
        num_test = tf.shape(speakers)[0] - num_train
        
        train_speakers = tf.cast(tf.slice(speakers, [0], [num_train]), dtype=tf.int64)
        test_speakers = tf.cast(tf.slice(speakers, [num_train], [num_test]), dtype=tf.int64)
        
        train_speaker_scripts = tf.cast(tf.slice(speaker_scripts, [0, 0], [num_train, self.params.sentence_max_length]), dtype=tf.int64)
        test_speaker_scripts = tf.cast(tf.slice(speaker_scripts, [num_train, 0], [num_test, self.params.sentence_max_length]), dtype=tf.int64)
        
        train_addressees = tf.cast(tf.slice(speakers, [0], [num_train]), dtype=tf.int64)
        test_addressees = tf.cast(tf.slice(speakers, [num_train], [num_test]), dtype=tf.int64)

        train_addressee_scripts = tf.cast(tf.slice(addressee_scripts, [0, 0], [num_train, self.params.sentence_max_length]), dtype=tf.int64)
        test_addressee_scripts = tf.cast(tf.slice(addressee_scripts, [num_train, 0], [num_test, self.params.sentence_max_length]), dtype=tf.int64)
        print('number of train data = ', len(train_addressees))
        test_data = [test_speakers, test_speaker_scripts, test_addressees, test_addressee_scripts]
        train_data = [train_speakers, train_speaker_scripts, train_addressees, train_addressee_scripts]

        return train_data, test_data

    def read_batch(self, data, start_index, mode='train'):
        """
        Use the helper functions in this file to read and parse training and test data, then pad the corpus.
        Then vectorize your train and test data based on your vocabulary dictionaries.

        :param data: train or test data, a 2-D array of size (train_sz/test_sz, 4) each row with form [speaker_id, speaker_scripts, addressee_id, addressee_scripts]
        :param start_index: starting index of rows in spkr_adrs_list.
        :param mode: #TODO
        :return sources: 2-D array of size (batch_size, sentence_max_length) that contains the batched, tokenized source scripts 
        :       targets: 2-D array of size (batch_size, sentence_max_length) that contains the batched, tokenized target scripts 
        :       speakers: 1-D array of size (batch_size) that contains speaker ids
        :       addressees: 1-D array of size (batch_size) that contains addressee ids
        """
        speakers = data[0][start_index:(start_index + self.params.batch_size)]
        sources = data[1][start_index:(start_index + self.params.batch_size), :]
        addressees = data[2][start_index:(start_index + self.params.batch_size)]
        targets = data[3][start_index:(start_index + self.params.batch_size), :]
        # print('speakers.shape = ', speakers.shape)
        # print('sources.shape = ', sources.shape)
        # print('addressees shape = ', addressees.shape)
        # print('targets shape = ', targets.shape)
        # sources = np.zeros((self.params.batch_size, self.params.sentence_max_length))
        # targets = np.zeros((self.params.batch_size, self.params.sentence_max_length))
        # speakers = np.zeros(self.params.batch_size)
        # addressees = np.zeros(self.params.batch_size)
        # max_source_len = 0
        # max_target_len = 0
        # END = 0
        # for i in range(self.params.batch_size):
        #     entry = spkr_adrs_list[start_index + i]
        #     source_i = entry[1]
        #     target_i = entry[3]
        #     # if max_source_len < len(source_i):
        #     #     max_source_len = len(source_i) 
        #     # if max_target_len < len(target_i):
        #     #     max_target_len = len(target_i)
        #     sources[i, :len(source_i)] = source_i
        #     targets[i, :len(target_i)] = target_i
        #     speakers[i] = entry[0]
        #     addressees[i] = entry[2]
        return sources, targets, speakers, addressees
