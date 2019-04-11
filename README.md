# NDSC2019

Classification task for National Data Science Challenge (NDSC) 2019
<https://www.kaggle.com/c/ndsc-beginner>

Worked under Team Four Eyes, with team members 
[@oonyoontong]( https://github.com/oonyoontong ), 
[@ashiswin]( https://github.com/ashiswin ) 
and 
[@jarvin95]( https://github.com/jarvin95 ).

Text-only approach, classifying items using their **Item Titles** only.

Experimented with:

1. Multinomial Naive Bayes, Linear SVC with TFIDF Vectorization
2. ULMfit Language Model with AWD-LSTM Classifier (using fastai)
3. Word embeddings with Bi-LSTM stack (using Keras)

Best performance: *Word embeddings with Bi-LSTM stack* with **74.7%** in public leaderboard

## Requirements

Python3

- tensorflow-gpu
- scikit-learn
- keras
- torch
- fastai
- nltk
