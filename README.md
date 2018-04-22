# A Deep bi-LSTM Semantic Role Labeleler for the Portuguese language

This software is a state-of-the-art semantic role labeler for the Portuguese language (PropBank-Br corpus). It relies on a deep bidirectional long short-term memory neural network architecture.

## System's accuracy

| Semantic role                    | Precision | Recall | F-Score |
|--------------------------|-----------|--------|---------|
| A0         |  81.82 |  86.90 |  84.28 |
| A1         |  71.15 |  72.29 |  71.71 |
| A2         |  52.73 |  42.03 |  46.77 |
| A3         |  28.57 |  40.00 |  33.33 |
| A4         | 100.00 |  50.00 |  66.67 |
| AM-ADV     |  42.86 |  50.00 |  46.15 |
| AM-CAU     |  50.00 |  33.33 |  40.00 |
| AM-DIS     |  44.44 |  28.57 |  34.78 |
| AM-EXT     |   0.00 |   0.00 |   0.00 |
| AM-LOC     |  54.17 |  72.22 |  61.90 |
| AM-MED     |   0.00 |   0.00 |   0.00 |
| AM-MNR     |  34.78 |  47.06 |  40.00 |
| AM-NEG     |  90.00 |  94.74 |  92.31 |
| AM-PNC     |  42.86 |  66.67 |  52.17 |
| AM-PRD     | 100.00 |  33.33 |  50.00 |
| AM-TMP     |  66.67 |  73.47 |  69.90 |
| Overall    | 67.62    |  68.75 | 68.18 |

## Framework setup

Make sure that you have a python 2.7 installed. After cloning this repository, you should download the resources using data_utils.py. Specifically, it will download pre-tained word embeddings and network weigths to be used as an off-the-self semantic role labeler. 

You should also install the required dependencies listed below:

* unicodecsv
* logging
* numpy
* pandas
* tensorflow
* keras
* h5py

The next step is to pre-process the original dataset. Run the prepare_features.py file. It will generate 20 folds from the original training data. This setting is to replicate the paper results.

The you can execute train.py to start training the model.


## Making predictions:

To come 

# How to cite

The paper is currently under review and soon we will have the final bib item





