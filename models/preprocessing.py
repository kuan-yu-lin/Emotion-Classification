'''
Author: Tzu-Ju Lin
functions used for reading data for training and testing models
'''
import pandas as pd

'''
emo_dict
input: a list of text labels from ISEAR dataset
output: a list of int
'''
emo_dict = {'joy': 0, "fear": 1, "shame": 2,
            "sadness": 3, "disgust": 4, "guilt": 5, "anger": 6}


def emo_label(ls):
    emo_dict = {'joy': 0, "fear": 1, "shame": 2,
                "sadness": 3, "disgust": 4, "guilt": 5, "anger": 6}
    convert_emo = []

    for emotion in ls:
        if emotion in emo_dict.keys():
            convert_emo.append(emo_dict[emotion])
    return convert_emo


'''
get_data
parameter:
    file: file path of the data
    deli: delimeter
    header(optional): None or [0]
    emo: binary, if True, the emotion labels will be converted to int with emo_lable
output: a dataframe with labels at position[0] and sentences at position[1]
'''


def get_data(file, deli, header=[0], emo=False):
    df = pd.read_csv(file, delimiter=deli, header=header)
    if emo == True:
        df_emo = emo_label(df[0])
        df = pd.DataFrame({'emotion': df_emo, 'sentence': df[1]})
    else:
        df = pd.DataFrame({'emotion': df[0], 'sentence': df[1]})
    return df
