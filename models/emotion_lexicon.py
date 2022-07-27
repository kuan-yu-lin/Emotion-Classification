'''
Author: Tzu-Ju Lin
This file is used for BERT[emo] amd MLP[bert+emo]
The final value of "df_train" is a n x 12 dataframe, the first column is emotion (int),
second column is the sentences (string), the following ten columns emotions in the NRC-dataset (binary)'''


#import library
import pandas as pd
import numpy as np
import re
from preprocessing import emo_label, get_data

# get the original training data
# variable "data_file_path" should be change into the file path
data_file_path = None
df_train = get_data(data_file_path, deli=',', header=None)

# get the file path of the emotion lexicon
# variable "emo_path should be change into the file path
emo_path = None
emo_df = pd.read_csv(emo_path)


# get the new target integer labels
train_emo = emo_label(df_train.iloc[:, 0])

# null column with the length of dataset
null_col = [0]*len(df_train)

# create a new dataframe that includes the new features from NRC emotion lexicon dataset
df_train = pd.DataFrame({'emotion': train_emo, 'sentence': df_train.iloc[:, 1], 'Positive': null_col, 'Negative': null_col, 'Anger': null_col,
                        'Anticipation': null_col, 'Disgust': null_col, 'Fear': null_col, 'Joy': null_col, 'Sadness': null_col, 'Surprise': null_col, 'Trust': null_col})


# convert the emo_df into a dictionary form. Words are keys and the values is a dictionary that stores the binary emotion features
emotion_lexicon = {}
emo_df_col = ["Positive", "Negative", "Anger", "Anticipation",
              "Disgust", "Fear", "Joy", "Sadness", "Surprise", "Trust"]
for i in range(len(emo_df)):
    emotion_lexicon[emo_df.Word[i]] = {}
    for col, j in zip(emo_df_col, range(1, 11)):
        emotion_lexicon[emo_df.Word[i]][col] = emo_df.iloc[i][j]


# renew emo_df, check each token in each sentence in the training data and then renew the binary features
for i in range(len(df_train)):
    example = df_train.iloc[i]
    for word in re.findall(r"[\w']+|[.,!?;]", example.sentence):
        if word in emotion_lexicon.keys():
            for items in emo_df_col:
                if df_train.at[i, items] == 0:
                    df_train.at[i, items] = df_train.at[i,
                                                        items] + emotion_lexicon[word][items]
