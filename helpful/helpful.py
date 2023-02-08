# -*- coding: utf-8 -*-
"""helpful.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FKlQTjkJWvcmR7ErMvK6gpYxVunIxvSr
"""

__author__="Toluwa Fayemi"
__copyright__='toluwa.fayemi@gmail.com'
__date__="2021/06/16"

"""
Using Amazon Review data from 2018 (https://nijianmo.github.io/amazon/index.html), the following
code demonstrates the implementation of a data-pre-processing, text-vectorization, word-embedding,
strategy, as well as the implementation of a neural network architecure designed for the given
problem.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, Dense, Input, Dropout, LSTM, Activation, Attention
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import numpy as np
import pandas as pd

import re, sys, string, math, urllib, zipfile, io
import matplotlib.pyplot as plt

#downloading required nltk packages
nltk.download('stopwords')
nltk.download('wordnet')

#load dataset
dataset_url_1='http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Industrial_and_Scientific_5.json.gz' #industrial&scientific
dataset_url_2='http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Luxury_Beauty_5.json.gz' #luxury beauty
dataset_url_3='http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Tools_and_Home_Improvement_5.json.gz' #tools&home_improvement
dataset_url_4='http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Office_Products_5.json.gz' #office products

datasets=[dataset_url_1,dataset_url_2,dataset_url_3,dataset_url_4]

def build_dataset(arr):
#extracts dataframe from url and adds subsection to master dataframe
  masterset=pd.DataFrame()
  for url in arr:
    df=pd.read_json(
        url,
        lines=True,
        nrows=34000
    )
    masterset=masterset.append(df)
  return masterset

#build dataset, purge duplicates, replace n/a vote values with 0, and handle large number strings
dataset=build_dataset(datasets)
dataset.drop_duplicates(subset=['reviewText', 'asin', 'reviewerID'], inplace=True)
dataset.loc[dataset['vote'].isnull(),['vote']]=0
dataset.vote=dataset.vote.astype(str).str.replace(',','').astype(float)

#determining total helpfulness votes for products in given subset of review
asins=set(dataset.asin)
for asin in asins:
  dataset.loc[dataset['asin']==asin, ['total_votes']]=dataset.loc[dataset['asin']==asin].vote.sum()

#purge results in which the value of total_votes is 0. they're useless to us.
dataset=dataset[dataset.total_votes != 0]

#calculate and record the vote ratio
dataset['vote_ratio']=dataset.apply(lambda x: x['vote']/x['total_votes'],axis=1)



"""# Functions """

def mapper(url):
  #creates dictionary maps of words to indices, indices to words (deprecated), and words to gloVe vectors
  word_idx,idx_word,gloVe={},{},{}
  words=set()
  with open('./glove.6B.100d.txt','r') as f:
    for l in f:
      l=(l.strip().split())
      words.add(l[0])
      gloVe[l[0]]=np.array(l[1:],dtype=np.float64)
    idx=1
    for word in sorted(words):
      word_idx[word]=idx
      idx_word[idx]=word
    return word_idx,idx_word,gloVe

def sentence_idx(review_arr,word_idx,maxlen):
  m=len(review_arr)
  idx_matrix=np.zeros((m,maxlen))
  for i in range(m):
    tokens=str(review_arr[i]).split()
    j=0
    for token in tokens:
      if word_idx.get(token) is not None:
        idx_matrix[i,j]=word_idx.get(token)
        j=j+1
      if j>=maxlen:
        break
    return idx_matrix

def embedding_layer():
  max_vocab=len(word_idx)+1
  sample=gloVe.get("the")
  word_dim=sample.shape[0]
  embedding_matrix = np.zeros((max_vocab,word_dim))

  for word, idx in word_idx.items():
    embedding_matrix[idx, :]=gloVe.get(word)
  layer=Embedding(max_vocab, word_dim, trainable=False)
  layer.build((None))
  layer.set_weights([embedding_matrix])

  return layer

stop_words,lemmatizer=set(stopwords.words("english")),WordNetLemmatizer()

def clean_text(txt):
  #clean, stem and lemmatize strings in text
  txt=txt.translate(str.maketrans('','',string.punctuation)).lower()
  txt=[lemmatizer.lemmatize(token) for token in txt.split(" ")]
  txt=[lemmatizer.lemmatize(token,"v") for token in txt]
  txt=[word for word in txt if not word in stop_words]
  txt=" ".join(txt)
  return txt

word_idx,idx_word,gloVe=mapper('./glove.6B.100d.txt')

"""# Mutli-Class Classification"""

def labeler(x):
  #assign a label to each review given it's helpfulness ratio
  #4 - Not Helpful
  if x['vote_ratio']==0:
    return 3
  #3 - Somewhat Helpful
  elif 0 < x['vote_ratio'] <= 0.3:
    return 2
  #2 - Helpful
  elif 0.3 < x['vote_ratio'] <= 0.7:
    return 1
  #1 - Very Helpful
  elif x['vote_ratio']>=0.7:
    return 0

dataset['helpful']=dataset.apply(labeler, axis=1)

#separate the different classes of helpfulness into distinct subsets
very_helpful=dataset.loc[dataset['helpful']==0]
helpful_=dataset.loc[dataset['helpful']==1]
somewhat_helpful=dataset.loc[dataset['helpful']==2]
not_helpful=dataset.loc[dataset['helpful']==3]

classes=[very_helpful,helpful_,somewhat_helpful,not_helpful]

balanced_set=pd.DataFrame()

for clss in classes:
  balanced_set=balanced_set.append(clss.sample(10000, replace=True))

#split data into features and labels
x_raw,y_=np.array(balanced_set.reviewText.values), np.array(balanced_set.helpful.values)
if len(x_raw)!=len(y_):
  raise ValueError("!MISMATCHED FEATURES & LABELS!")

x_clean=np.vectorize(clean_text)(x_raw.astype(str))

#split cleaned data into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x_clean,y_,test_size=0.2)

def build_model(shape):
  input_layer=Input(shape,dtype='int32')
  emb_layer=embedding_layer()
  embeddings=emb_layer(input_layer)

  x=LSTM(32, return_sequences=True)(embeddings)
  x=Dropout(rate=0.5)(x)
  x=LSTM(32,return_sequences=False)(x)
  x=Dropout(rate=0.5)(x)
  x=Dense(4,activation='relu')(x)
  x=Activation('sigmoid')(x)

  return Model(inputs=input_layer, outputs=x)

model=build_model((50,))

model.summary()

checkpoint = ModelCheckpoint(filepath='model/helpfulness_prediction_model.hdf5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
handbrake = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

x_train=sentence_idx(x_train,word_idx, 50)

from keras.optimizers import SGD
opt = SGD(learning_rate=0.05)

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history1=model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[checkpoint,handbrake], shuffle=True)

#Plotting Accuracy
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.title('Multi Classification Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc="lower right")
plt.show()

#Plotting Loss
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('Multi Classification Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc="lower right")
plt.show()

model.save('./MultiClassificationModel_1')





"""# Binary Classification"""

def labeler(x):
  #assign a label to each review given it's helpfulness ratio
  #0 - Not Helpful
  if 0<=x['vote_ratio']<0.20:
    return 0
  #1 - Helpful
  elif 0.2<x['vote_ratio']:
    return 1

binary_dataset=dataset
binary_dataset['helpful']=dataset.apply(labeler,axis=1)

_helpful_=binary_dataset.loc[binary_dataset['helpful']==1].sample(10000, replace=True)
_not_helpful=binary_dataset.loc[binary_dataset['helpful']==0].sample(10000)

balanced_binary=_helpful_.append(_not_helpful)

#split data into features and labels
x_binary_raw,y_binary=np.array(balanced_binary.reviewText.values), np.array(balanced_binary.helpful.values)
if len(x_binary_raw)!=len(y_binary):
  raise ValueError("!MISMATCHED FEATURES & LABELS!")

x_binary_clean=np.vectorize(clean_text)(x_binary_raw.astype(str))

#split cleaned data into training and testing sets
x_binary_train,x_binary_test,y_binary_train,y_binary_test=train_test_split(x_binary_clean,y_binary,test_size=0.2)

def binary_model(input_shape):
  input_layer=Input(input_shape,dtype='int32')
  emb_layer=embedding_layer()
  embeddings=emb_layer(input_layer)

  x=LSTM(32, return_sequences=True)(embeddings)
  x=Dropout(rate=0.5)(x)
  x=LSTM(32,return_sequences=False)(x)
  x=Dropout(rate=0.5)(x)
  x=Dense(1,activation='relu')(x)
  x=Activation('relu')(x)

  return Model(inputs=input_layer, outputs=x)

x_binary_idx=sentence_idx(x_binary_train, word_idx,50)

binary_model=binary_model((50,))

binary_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history_2=binary_model.fit(x_binary_idx,y_binary_train, epochs=50, batch_size=32, validation_split=0.2, shuffle=True)

#Plotting Accuracy
plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.title('Binary Classification Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc="lower right")
plt.show()

#Plotting Loss
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('Binary Classification Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc="lower right")
plt.show()

model.save('./BinaryClassificationModel_1')