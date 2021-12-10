from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd
from nltk import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from nltk import tokenize

Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
filepath = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
with open(filepath, 'r') as file:
    data = file.read().replace('\n', '')
    tokenizer = RegexpTokenizer(r'\w+')
    sentences = tokenize.sent_tokenize(data)

# Read in full dataset
data = pd.read_csv('data/sentences.csv',
                   sep='\t',
                   encoding='utf8',
                   index_col=0,
                   names=['lang', 'text'])

# Filter by text length
len_cond = [True if 20 <= len(s) <= 200 else False for s in data['text']]
data = data[len_cond]

# Filter by text language
lang = ['deu', 'eng', 'fra', 'ita', 'por', 'spa']
data = data[data['lang'].isin(lang)]

# Select 50000 rows for each language
data_trim = pd.DataFrame(columns=['lang', 'text'])

for l in lang:
    lang_trim = data[data['lang'] == l].sample(50000, random_state=100)
    data_trim = data_trim.append(lang_trim)

# Create a random train, valid, test split
data_shuffle = data_trim.sample(frac=1)

train = data_shuffle[0:210000]
valid = data_shuffle[210000:270000]
test = data_shuffle[270000:270100]


def get_trigrams(corpus, n_feat=200):
    """
    Returns a list of the N most common character trigrams from a list of sentences
    params
    ------------
        corpus: list of strings
        n_feat: integer
    """

    # fit the n-gram model
    vectorizer = CountVectorizer(analyzer='char',
                                 ngram_range=(3, 3)
                                 , max_features=n_feat)

    X = vectorizer.fit_transform(corpus)

    # Get model feature names
    feature_names = vectorizer.get_feature_names()

    return feature_names


# obtain trigrams from each language
features = {}
features_set = set()

for l in lang:
    # get corpus filtered by language
    corpus = train[train.lang == l]['text']

    # get 200 most frequent trigrams
    trigrams = get_trigrams(corpus)

    # add to dict and set
    features[l] = trigrams
    features_set.update(trigrams)

# create vocabulary list using feature set
vocab = dict()
for i, f in enumerate(features_set):
    vocab[f] = i

# train count vectoriser using vocabulary
vectorizer = CountVectorizer(analyzer='char',
                             ngram_range=(3, 3),
                             vocabulary=vocab)

# create feature matrix for training set
corpus = train['text']
X = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names()

train_feat = pd.DataFrame(data=X.toarray(), columns=feature_names)

# Scale feature matrix
train_min = train_feat.min()
train_max = train_feat.max()
train_feat = (train_feat - train_min) / (train_max - train_min)

# Add target variable
train_feat['lang'] = list(train['lang'])

# create feature matrix for validation set
corpus = valid['text']
X = vectorizer.fit_transform(corpus)

valid_feat = pd.DataFrame(data=X.toarray(), columns=feature_names)
valid_feat = (valid_feat - train_min) / (train_max - train_min)
valid_feat['lang'] = list(valid['lang'])

# create feature matrix for test set
corpus = test['text']
X = vectorizer.fit_transform(sentences)

# test_feat = pd.DataFrame(data=X.toarray(), columns=feature_names)
test_feat = pd.DataFrame(data=X.toarray(), columns=feature_names)
test_feat = (test_feat - train_min) / (train_max - train_min)
test_feat['lang'] = sentences

# Fit encoder
encoder = LabelEncoder()
encoder.fit(['deu', 'eng', 'fra', 'ita', 'por', 'spa'])


def encode(y):
    """
    Returns a list of one hot encodings
    Params
    ---------
        y: list of language labels
    """

    y_encoded = encoder.transform(y)
    y_dummy = np_utils.to_categorical(y_encoded)

    return y_dummy


# Get training data
x = train_feat.drop('lang', axis=1)
print('Shape of X is ', x.shape)
inp = x.shape[1]
y = encode(train_feat['lang'])
print('Shape of y is ', y.shape)

# Define model
model = Sequential()
model.add(Dense(500, input_dim=inp, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(x, y, epochs=4, batch_size=100)

x_test = test_feat.drop('lang', axis=1)
y_test = test_feat['lang']

# Get predictions on test set
# labels = model.predict_classes(x_test)
labels = np.argmax(model.predict(x_test), axis=-1)
predictions = encoder.inverse_transform(labels)
print(predictions)