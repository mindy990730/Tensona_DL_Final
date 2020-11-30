import requests
import json
import pandas as pd
from tqdm.notebook import tqdm


class Data():
    def __init__(self, folder_path, output_file_name):
        self.data = 0
        self.folder_path = folder_path # ../json
        self.output_file_name = output_file_name # 'friends_transcripts.csv'



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
        # friends_data['season_id'] = dict()
        # for i in range(1, num_seasons+1):
        #     friends_data['season_id']
        # loop through each season
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

    def cleanup_speaker_addressee(self, friends_data):
        """
        To clean up the data into an 2-D array of num_lines * [speaker_id, speaker_scripts, addressee_id, addressee_scripts] (shape = num_lines * 4)
        :param friends_data: output of friends_csv()
        :return spkr_adrs_list: [speaker_id, speaker_scripts, addressee_id, addressee_scripts]
        :       character_dict: a dictionary mapping characters (both speaker & addressee) to id
        """
        num_lines = len(friends_data['season_id'])
        spkr_adrs_list = []
        character_dict = dict()
        count_characters = 0

        season_id = friends_data['season_id']
        episode_id = friends_data['episode_id']
        scene_id = friends_data['scene_id']
        utterance_id = friends_data['utterance_id']
        speakers = friends_data['speaker']
        tokens = friends_data['tokens']
        transcripts = friends_data['transcript']

        for i in range(num_lines-1):
            
            # If reached an empty line or the end of a conversation --> continue to the next line
            if len(tokens[i]) == 0 or len(tokens[i+1]) == 0:
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
        s = data[1]
        t = data[3]
        sources = s[start_index:(start_index + self.params.batch_size)]
        targets = t[start_index:(start_index + self.params.batch_size)]
        speakers = data[0][start_index:(start_index + self.params.batch_size)]
        addressees = data[2][start_index:(start_index + self.params.batch_size)]

        return sources, targets, speakers, addressees
