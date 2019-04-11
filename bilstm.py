import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import re
from collections import Counter
import text_utils
import tensorflow as tf

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, CuDNNLSTM, Bidirectional, Activation, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras import regularizers

# import fastai # pip install fastai
# from fastai import *
# from fastai.text import * 

# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

# # nltk.download('stopwords')
# # nltk.download('punkt')
# stopwords = stopwords.words('english')


data_path = './data/'
models_dir = './models'

seed = 100

MAX_NUM_WORDS = 15000
MAX_SEQUENCE_LENGTH = 30 # Longest sequence was 27 words
EMBEDDING_DIM = 128
NUM_SPLITS = 5
MAX_EPOCHS = 30

OPTIMIZER = 'adam'
PATIENCE_LR = 10


def retrieve_text(cat, mode):
    df = pd.read_csv(os.path.join(data_path, 'train_{}.csv'.format(cat)))
    seq_titles = df['title'].values
    labels = df['Category'].values
    classes = np.unique(labels)
    if mode == 'train':
        return df, seq_titles, labels, classes
    elif mode == 'test':
        test_df = pd.read_csv(os.path.join(data_path, 'test_{}.csv'.format(cat)))
        test_seq_titles = test_df['title'].values
        return seq_titles, test_df, test_seq_titles, classes
    


def process_text(seq_titles, tokenizer):
    # Tokenize and pad input titles
    tokenizer.fit_on_texts(seq_titles)
    sequences = tokenizer.texts_to_sequences(seq_titles)
    index = np.argmax([len(seq) for seq in sequences])
    print("Longest title length: {}".format(len(sequences[index])))
    word_index = tokenizer.word_index
    print("Found {} unique tokens".format(len(word_index)))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return tokenizer, data, word_index


def get_model(num_classes):
    model = Sequential()
    model.add(Embedding(MAX_NUM_WORDS, 
                        EMBEDDING_DIM,
                        input_length=MAX_SEQUENCE_LENGTH
                       ))
    model.add(
        Bidirectional(
            CuDNNLSTM(
                units=64, 
                return_sequences=True
            ),
            input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
        ))
    model.add(
        Bidirectional(
            CuDNNLSTM(
                units=64, 
                return_sequences=True
            ),
            input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
        ))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='linear', input_shape=(3840,), activity_regularizer=regularizers.l1(1e-4)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', input_shape=(1024,)))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_callbacks(name_weights):
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=PATIENCE_LR, 
        verbose=1, 
        mode='auto'
        )
    mcp_save = ModelCheckpoint(
        name_weights, 
        save_best_only=True, 
        monitor='val_loss', 
        mode='min'
        )
    reduce_lr_loss = ReduceLROnPlateau(
        monitor='loss', 
        factor=0.2, 
        patience=PATIENCE_LR, 
        verbose=1, 
        min_delta=1e-4, 
        mode='min'
        )
    return [early_stopping, mcp_save, reduce_lr_loss]


def train(folds, cat, data, labels, num_classes):
    for fold_i, (train_index, val_index) in enumerate(folds):
        print('\nFold ', fold_i)
        model = get_model(num_classes)
        print(model.summary())

        X_train, X_val = data[train_index], data[val_index]
        y_train, y_val = labels[train_index], labels[val_index]
        
        callbacks = get_callbacks(name_weights="{}_fold_{}_weights.h5".format(fold_i, cat))
        # fit the model
        history = model.fit(
            X_train, 
            y_train, 
            epochs=MAX_EPOCHS, 
            verbose=1,
            validation_data=(X_val, y_val), 
            callbacks=callbacks
            )
    

def test(model, test_data, classes):
    preds = model.predict(test_data)
    indices = preds.argmax(1)
    pred_classes = classes[indices]
    print("First 5 predictions: {}".format(preds[:5]))
    print("First 5 predictions classes: {}".format(pred_classes[:5]))
    return pred_classes


def run():
    parser = argparse.ArgumentParser(description='NDSC Text Classifier')
    parser.add_argument('--mode', type=str, default='train', metavar='N',
                        help='train or test (for submission mode) (default: train)')
    parser.add_argument('--cat', type=str, default='beauty_image', metavar='N',
                        help='beauty_image, fashion_image or mobile_image (default: beauty_image)')
    parser.add_argument('--fold', type=int, default=4, metavar='N',
                        help='folds 0-4 (default: 4)')
    args = parser.parse_args()

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    
    if args.mode == 'train':
        # Train
        df, seq_titles, labels, classes = retrieve_text(args.cat, args.mode)
        num_classes = len(classes)
        labelencoder = LabelEncoder()
        labels = labelencoder.fit_transform(labels)
        onehot_labels = to_categorical(labels)

        # Tokenize and pad input titles
        tokenizer.fit_on_texts(seq_titles)
        sequences = tokenizer.texts_to_sequences(seq_titles)
        index = np.argmax([len(seq) for seq in sequences])
        print("Longest title length: {}".format(len(sequences[index])))
        word_index = tokenizer.word_index
        print("Found {} unique tokens".format(len(word_index)))
        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        # tokenizer, data, word_index = process_text(seq_titles, tokenizer)

        print("Shape of data tensor: {}".format(data.shape))
        print("Shape of label tensor: {}".format(labels.shape))

        skf = StratifiedKFold(n_splits=NUM_SPLITS, shuffle=True, random_state=seed)
        folds = list(skf.split(data, labels))
        
        train(folds, args.cat, data, labels, num_classes)

    elif args.mode == 'test':
        # Test
        seq_titles, test_df, test_seq_titles, classes = retrieve_text(args.cat, args.mode)
        num_classes = len(classes)
        best_weights_path = "{}_fold_{}_weights.h5".format(args.fold, args.cat)
        model = get_model(num_classes)
        model.load_weights(best_weights_path)
        
        tokenizer.fit_on_texts(seq_titles)
        test_sequences = tokenizer.texts_to_sequences(test_seq_titles)
        test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
        print(test_sequences[:5])
        pred_classes = test(model, test_data, classes)

        print("Shape of test data tensor: {}".format(test_data.shape))

        # Write predictions to csv file
        output_csv_path = 'submission_{}_BILSTM.csv'.format(args.cat)

        with open(output_csv_path, 'a') as f:
            for i in range(len(pred_classes)):
                pred_class = pred_classes[i]
                row = '{},{}\n'.format(test_df['itemid'][i],pred_class)
                f.write(row)
    

if __name__=='__main__':
    # python3 bilstm.py --mode train --cat <cat>
    # python3 bilstm.py --mode test --cat <cat> --fold 4
    print("Using GPU:", tf.test.is_gpu_available())
    run()
