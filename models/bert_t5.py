'''
Author: Kuan-Yu Lin
'''
from t5_paraphrase import init_t5 
from bert_main import BERT_pure
from preprocessing import get_data
import os

if __name__ == "__main__":

    # create the augmented training data
    init_t5('train.csv')
    
    # get file path of
    train_data_file_name = './data/train_para.csv'
    test_data_file_name = './data/test.csv'
    val_data_file_name = './data/val.csv'
    
    train_path = os.path.abspath(train_data_file_name)
    val_path = os.path.abspath(test_data_file_name)
    test_path = os.path.abspath(val_data_file_name)

    # read the data
    df_train = get_data(train_path, deli=',', emo=True)
    df_val = get_data(val_path, deli=',', emo=True)
    df_test = get_data(test_path, deli=',', emo=True)

    model = BERT_pure()
    model.fit(df_train, df_val)
    model.train()
    model.test(df_test)
    model.evaluate()
