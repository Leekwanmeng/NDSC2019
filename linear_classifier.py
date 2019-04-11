import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
from collections import Counter
import text_utils

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

import nltk # pip install nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')
stopwords = stopwords.words('english')

seed = 100
data_path = './data/'
feature_length = 10000

#Clean Texts
def str_clean(text, ps):
    punct = '():[]?.,|_^-&><;!"/%'  
    table = str.maketrans(punct, ' '*len(punct), "0123456789$#'=")
    cleaned_comment = []
    for word in text.split():
        cleaned_comment.extend(word.translate(table).split())
        cleaned_comment = [ps.stem(word) for word in cleaned_comment]
    return " ".join(cleaned_comment)



def run(clf, output_csv_path):
    parser = argparse.ArgumentParser(description='NDSC Text Classifier')
    parser.add_argument('--mode', type=str, default='train', metavar='N',
                        help='train or test (for submission mode) (default: train)')
    args = parser.parse_args()

    print("\nRunning {}...".format(clf.__name__))
    clf = clf()
    ps = PorterStemmer()

    for cat in ['beauty_image', 'fashion_image', 'mobile_image']:
        df = pd.read_csv(os.path.join(data_path, 'train_' + cat + '.csv'))
        # df2 = pd.read_csv('./translations.txt', sep=';')
        # df['title'] = df2['title']

        # Text cleaning
        df['title'] = text_utils.clean_text(df['title'], stopwords)

        # Vector
        vectorizer = TfidfVectorizer(strip_accents='unicode',
            analyzer='word', token_pattern=r'\w{1,}', lowercase=True,
            max_features=feature_length
            )

        if args.mode == 'train':
            _, _, X_train, y_train, X_val, y_val = text_utils.data_split(df, seed)
            train_vectors = vectorizer.fit_transform(X_train)
            val_vectors = vectorizer.transform(X_val)
            print("Feature size:", train_vectors.shape)
            # Train
            clf.fit(train_vectors, y_train)
            predicted = clf.predict(val_vectors)
            print("Accuracy for {}: {:.2f}%".format(cat, accuracy_score(y_val, predicted)))

        elif args.mode == 'test':
            # Only for train on ALL for submission testing
            X_train, y_train = df['title'].values, df['Category'].values
            train_vectors = vectorizer.fit_transform(X_train)

            test_df = pd.read_csv('./test_' + cat + '.csv')
            X_test = test_df['title'].values
            test_vectors = vectorizer.transform(X_test)

            predicted = clf.predict(test_vectors)
            print(predicted)
            print(test_df['itemid'].values)
            with open(output_csv_path, 'a') as f:
                for i in range(len(predicted)):
                    row = '{},{}\n'.format(test_df['itemid'][i],predicted[i])
                    f.write(row)
        else:
            raise Exception("Please enter mode as 'train' or 'test'")

if __name__=='__main__':
    # python3 text_classifier.py --mode train
    # python3 text_classifier.py --mode test

    # MNB beauty:71%, fashion:51%, mobile:72%
    # SGD beauty:74%, fashion:58%, mobile:79%
    # SVC beauty:76%, fashion:60%, mobile:80%

    for idx, clf in enumerate([MultinomialNB, SGDClassifier, LinearSVC]):
        output_csv_path = 'submission{}.csv'.format(idx)
        run(clf, output_csv_path)