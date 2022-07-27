'''
Author: Kuan-Yu Lin
'''
from transformers import AutoModelWithLMHead, AutoTokenizer
import csv

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-small-finetuned-quora-for-paraphrasing")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-small-finetuned-quora-for-paraphrasing")

def paraphrase(text, max_length=128):

    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)
    generated_ids = model.generate(input_ids=input_ids, num_return_sequences=5, num_beams=5, max_length=max_length, no_repeat_ngram_size=2, repetition_penalty=3.5, length_penalty=1.0, early_stopping=True)
    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

    return preds

row_lst = []
with open('emotion_data/train.csv', newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=',')
    for row in datareader:
        row_lst.append(row)

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


#print('0: ', emo_lst[0])
#print('0: ', eve_lst[0])
#print('1: ', emo_lst[1])
#print('1: ', eve_lst[1])
#print('2: ', emo_lst[2])
#print('2: ', eve_lst[2])

print(new_eve_lst[0])
print(new_eve_lst[1])

with open('emotion_data/train_para.csv', 'w', newline='') as csvfile:
    datawriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(new_eve_lst)):
        datawriter.writerow(new_eve_lst[i])
