"""
Author: Tzu-Ju Lin
"""

from keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


def encode(df, max_len, tokenizer, batch_size):
    # encode training data with special tokens and add padding
    encoded_sent = []
    for sent in df.sentence:
        sent = tokenizer.tokenize(sent)
        sent = ["[CLS]"] + sent + ["[SEP]"]
        sent = tokenizer.convert_tokens_to_ids(sent)
        encoded_sent.append(sent)
    encoded_sent = pad_sequences(
        encoded_sent, maxlen=max_len, dtype="long", truncating="post", padding="post")

    # get attention masks
    attention_masks = []
    for sent in encoded_sent:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)

    encoded_sent = torch.tensor(encoded_sent)
    attention_masks = torch.tensor(attention_masks)
    encoded_target = torch.tensor(df.emotion)

    data = TensorDataset(encoded_sent, attention_masks, encoded_target)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader
