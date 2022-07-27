# Emotion_Classification

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

    python3 ./baseline/main.py

### BERT<sub>pure</sub>

    python3 ./models/bert_main.py

### BERT<sub>T5</sub>

    # data augmentation
    python3 ./models/t5_paraphrase.py

    python3 ./models/bert_main.py

### BERT<sub>emo</sub>

    python3 ./models/bert_main.py

### MLP + BERT<sub>emo</sub>

    python3 ./models/bert_main.py



