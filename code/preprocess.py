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
                character_dict['speaker_name'] = count_characters
                count_characters += 1
            if addressee_name not in character_dict.keys():
                character_dict['addressee_name'] = count_characters
                count_characters += 1
            speaker_scripts = tokens[i]
            addressee_scripts = tokens[i+1]
            spkr_adrs_list.append([character_dict['speaker_name'], speaker_scripts, character_dict['addressee_name'], addressee_scripts])
        return spkr_adrs_list, character_dict 
