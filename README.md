# NDSC2019
Classification task for National Data Science Challenge (NDSC) 2019
https://www.kaggle.com/c/ndsc-beginner

Text-only approach (item titles)
Experimented with:
1. Multinomial Naive Bayes, Linear SVC with TFIDF Vectorization
2. ULMfit Language Model with AWD-LSTM Classifier (using fastai)
3. Word embeddings with Bi-LSTM stack (using Keras)

Best performance: **Word embeddings with Bi-LSTM stack**

## Requirements
tensorflow-gpu
scikit-learn
keras
torch
fastai
nltk
