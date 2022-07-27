'''Author: Tzu-Ju Lin'''

import tensorflow as tf
import torch
from transformers import BertTokenizer,  AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup
import pandas as pd
import io
import numpy as np
import time
import datetime
import random
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from preprocessing import get_data
from bert_encoder import encode

# detect GPU
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

'''
class BERT-pure
reference: https://colab.research.google.com/drive/1ywsvwO6thOVOrfagjjfuxEf6xVRxbUNO
'''


class BERT_pure:

    def __init__(self, epoch=2, MAX_LENGTH=256, num_label=7, batch_size=32):
        self.epoch = epoch
        self.MAX_LEN = MAX_LENGTH
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower_case=True)
        self.num_label = num_label
        self.batch_size = batch_size

        # encoded for train
        self.train_dataloader = None

        # encoded for validation
        self.val_dataloader = None

        # encoded for test
        self.test_dataloader = None

        # predict results
        self.predictions = None
        self.true_labels = None

        self.model = None

    '''
    fit
    convert the data into Tensor readable format
    input:training dataframe and validatation dataframe
    '''

    def fit(self, df_train, df_val):

        self.train_dataloader = encode(
            df_train, max_len=self.MAX_LEN, tokenizer=self.tokenizer, batch_size=self.batch_size)
        self.val_dataloader = encode(
            df_val,  max_len=self.MAX_LEN, tokenizer=self.tokenizer, batch_size=self.batch_size)

    '''
    flat accuracy
    accuracy function used for evaluation during training phase
    '''

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    '''
    format_time
    for presenting training time during the training process
    '''

    def format_time(elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    '''
    train
    make sure to fit the training and validation data before training
    '''

    def train(self):
        # initialize the model
        # bert-base-uncased treat all case equally
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=self.num_label, output_attentions=False, output_hidden_states=False)
        model.cuda()

        # set up parameters of the model
        optimizer = AdamW(model.parameters(),
                          lr=2e-5,
                          eps=1e-8
                          )
        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        total_steps = len(self.train_dataloader)*self.epoch
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)
        # loss tracking list
        loss_values = []

        # training process
        for epoch_i in range(self.epoch):
            print(" ")
            print(
                '======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epoch))
            print('Training...')
            t0 = time.time()
            total_loss = 0
            model.train()
            for step, batch in enumerate(self.train_dataloader):
                if step % 40 == 0 and not step == 0:
                    elapsed = self.format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                        step, len(self.train_dataloader), elapsed))
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                model.zero_grad()
                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)
                # track the training loss
                loss = outputs[0]
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            avg_train_loss = total_loss / len(self.train_dataloader)
            loss_values.append(avg_train_loss)
            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(
                self.format_time(time.time() - t0)))

            # Validation
            print("")
            print("Running Validation...")
            t0 = time.time()
            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            for batch in self.val_dataloader:

                # Add batch to GPU
                batch = tuple(t.to(device) for t in batch)

                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch

                # Telling the model not to compute or store gradients, saving memory and
                # speeding up validation
                with torch.no_grad():
                    outputs = model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask)
                logits = outputs[0]
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                tmp_eval_accuracy = self.flat_accuracy(logits, label_ids)
                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1
            print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
            print("  Validation took: {:}".format(
                self.format_time(time.time() - t0)))

        print("")
        print(loss_values)
        print("Training complete!")
        self.model = model

    '''
    testing
    input: df_test
    return: true label of df_test and a list of predicted label, both can be used to evaluate the model
    '''

    def test(self, test_dataframe):
        print("Predicting labels")

        # set model to evaluation mode

        self.model.eval()

        pred, true = [], []

        # convert the test data into Tensordataset
        self.test_dataloader = encode(
            test_dataframe, max_len=self.MAX_LEN, tokenizer=self.tokenizer, batch_size=self.batch_size)

        # predict
        for batch in self.test_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = self.model(b_input_ids, token_type_ids=None,
                                     attention_mask=b_input_mask)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            pred.append(logits)
            true.append(label_ids)

        # get the true labels into one single list
        true_label = []
        for sublist in true:
            true_label += sublist.tolist()

        # get the predictions from the model output
        predictions = []
        for batch in pred:
            for prediction in batch:
                prediction = prediction.tolist()
                max = 0
                max_idx = 0
                for i in range(len(prediction)):
                    if prediction[i] > max:
                        max = prediction[i]
                        max_idx = i
                predictions.append(max_idx)
        self.true_labels = true_label
        self.predictions = predictions

    '''
    evaluate
    prints out the macro f1 score, classification report and confusion matrix of the testing
    '''

    def evaluate(self):
        cm = confusion_matrix(self.true_label, self.predictions)
        report = classification_report(self.true_label, self.predictions, target_names=[
                                       'joy', "fear", "shame", "sadness", "disgust", "guilt", "anger"])
        f1 = f1_score(self.true_label, self.predictions, average='macro')

        print("The model get a f1-score of: ",  f1)
        print("The overall evaluation of the model is as following:")
        print(report)
        print("The confusion matrix:")
        print(cm)


def demo():

    # get file path of
    train_path = None
    val_path = None
    test_path = None

    # read the data
    df_train = get_data(train_path, deli=',', emo=True)
    df_val = get_data(val_path, deli=',', emo=True)
    df_test = get_data(test_path, deli=',', emo=True)

    model = BERT_pure()
    model.fit(df_train, df_val)
    model.train()
    model.test(df_test)
    model.evaluate()


demo()
