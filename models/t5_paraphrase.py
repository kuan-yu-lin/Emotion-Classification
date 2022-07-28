'''
Author: Kuan-Yu Lin
'''
from transformers import AutoModelWithLMHead, AutoTokenizer
import csv
import os

# get the fine-tune T5 model for paraphrasing
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-small-finetuned-quora-for-paraphrasing")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-small-finetuned-quora-for-paraphrasing")

def paraphrase(text, max_length=128):

    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)
    generated_ids = model.generate(input_ids=input_ids, num_return_sequences=5, num_beams=5, max_length=max_length, no_repeat_ngram_size=2, repetition_penalty=3.5, length_penalty=1.0, early_stopping=True)
    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

    return preds

def load_train_data(filename):
    '''
    function to load the training data

    argument: string
        - file name of training data (file type: csv)

    return: a list
        - encode each row of data into a list with two items, i.e. label and data entry.
    '''
    train_file_name = './data/' + filename
    train_path = os.path.abspath(train_file_name)

    row_lst = []
    with open(train_path, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        for row in datareader:
            row_lst.append(row)

    return row_lst

def init_t5(filename):    
    '''
    function to produce the augmented training data by paraphrasing with T5 model
    
    argument: string
        - file name of training data (file type: csv)
    '''
    row_lst = load_train_data(filename)
    
    emo_lst = []
    eve_lst = []
    for i in range(len(row_lst)):
        emo_lst.append(row_lst[i][0])
        eve_lst.append("paraphrase: " + row_lst[i][1])

    new_eve_lst = []    
    for i, e in enumerate(eve_lst):
        preds = paraphrase(e)
        for pred in preds:
            new_eve = []
            new_eve.append(emo_lst[i])
            new_eve.append(pred)
            new_eve_lst.append(new_eve)

    train_para_file_name = './data/train_para.csv'
    train_para_path = os.path.abspath(train_para_file_name)

    with open(train_para_path, 'w', newline='') as csvfile:
        datawriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(new_eve_lst)):
            datawriter.writerow(new_eve_lst[i])
