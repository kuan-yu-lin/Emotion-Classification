# Emotion-Classification

###### University Stuttgart
###### CL Team Lab Project, Sommer 2022

## Member Information

* [Kuan-Yu Lin](https://github.com/kuan-yu-lin)
* [Tzu-Ju (Jenny) Lin](https://github.com/TzuJuLin)

## Project Information

### Description

With the gaining popularity of emotion classification, the following project deals with different approaches of implementing emotion classification. The ISEAR dataset, which includes scenarios regarding seven emotion, joy, anger, fear, shame, disgust, guilt and sadness, was used in this project. We implemented single layer perceptron, BERT with three different inputs and a Multi-layer perceptron that combines maching learning and lexicon-based approaches in our project. 

### Dataset

The ISEAR dataset can be found [here](https://www.unige.ch/cisa/research/materials-and-online-research/research-material/).

|Labels|
|------|
|joy|
|anger|
|fear|
|shame|
|disgust|
|sadness|

### Results from experiments

|         Emotions        | f1-score |
|:-----------------------:|:--------:|
|    BERT<sub>pure</sub>  |   0.71   |
|     BERT<sub>T5</sub>   |   0.69   |
|     BERT<sub>emo</sub>  |   0.70   |
| MLP<sub>BERT+pure</sub> |   0.60   |

## Run

### Baseline

1. download the dataset to the folder "data"
2. name the training data with "train.txt" and testing data with test.txt
3. run
```
    python3 ./baseline/main.py
```
### BERT<sub>pure</sub>

1. download the dataset and insert the correct datapath in main.py 
2. run
```
    python3 ./models/bert_main.py
```
### BERT<sub>T5</sub>

1. download the dataset and preprocess it with bert_t5.py
2. insert the correct datapath in main.py
3. run
```
    python3 ./models/bert_t5.py
    python3 ./models/main.py
```
### BERT<sub>emo</sub>

1. download the dataset and preprocess it with emotion_lexicon.py
2. insert the correct datapath in main.py 
3. run
```
    python3 ./models/bert_main.py
```
### MLP + BERT<sub>emo</sub>

1. download the dataset and preprocess it with emotion_lexicon.py
2. take the original dataset and do feature extraction with bert_feature.py
3. concatenate the results from 1 and 2, save as a new dataset
4. insert the datapath of the concatenated in MLP.py
5. run
```
    python3 ./model/emotion_lexicon.py
    python3 ./model/bert_feature.py
    python3 ./models/MLP.py
```


