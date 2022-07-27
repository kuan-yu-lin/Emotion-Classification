'''Author: Tzu-Ju Lin'''

from preprocessing import get_data
from bert_encoder import encode
from transformers import BertTokenizer, BertModel, BertConfig
import torch
import pandas as pd
device = torch.device(f"cuda:{2}" if torch.cuda.is_available() else "cpu")

'''
bert-based feature extraction 
'''


class bert_feature:

    def __init__(self, MAX_LEN=256, batch_size=32):
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower_case=True)
        self.input_ids = None
        self.attention_mask = None
        self.embeddings = None
        self.MAX_LEN = MAX_LEN
        self.batch_size = batch_size

    '''
    fit
    pass in the training dataframe to make the data readable for BERT
    input: df_train(first column)
    '''

    def fit(self, df_train):

        input_ids = []
        attention_masks = []
        # this line can be changed to any list of sentence
        for sentence in df_train['sentence'].tolist():
            dictionary = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=self.MAX_LEN,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            input_ids.append(dictionary['input_ids'])
            attention_masks.append(dictionary['attention_mask'])
        self.input_ids = input_ids
        self.attention_mask = attention_masks
    '''
    train bert-base-uncased to get the text embeddings
    '''

    def train(self):
        train_input_ids = torch.cat(self.input_ids, dim=0)
        train_attention_masks = torch.cat(self.attention_mask, dim=0)
        config = BertConfig.from_pretrained(
            "bert-base-uncased", output_hidden_states=True)
        model = BertModel.from_pretrained("bert-base-uncased", config=config)
        with torch.no_grad():
            outputs = model(train_input_ids,
                            attention_mask=train_attention_masks)
        self.embeddings = outputs[2][1:]

    '''
    get_embedding
    nr_layer: 0-11, bert-base outputs 12 layers in total. This parameter allows user to 
            deciede which layer they want
    This functions returns the specified layer
    '''

    def get_embedding(self, nr_layer):
        cls_embeddings = []
        for i in range(12):
            cls_embeddings.append(self.embeddings[i][:, 0, :].numpy())

        return cls_embeddings[nr_layer]

# demonstration function


def demo():
    # please change this variable to the path of the feature-extraction data
    data_path = None

    df_train = get_data(data_path, deli=',', header=None, emo=True)
    model = bert_feature()
    model.fit(df_train)
    model.get_embedding()
    # in order to get the last embedding, the parameter is set to 11
    last_embed = model.get_embedding(11)
    return last_embed


demo()
