# -*- coding: utf-8 -*-
"""helpful2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1O9-B6b0f4y8MOCybou6OPBAKkHy9oa4_
"""



import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, Dense, Input, Dropout, LSTM, Activation, Attention
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
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

stop_words,lemmatizer=set(stopwords.words("english")),WordNetLemmatizer()
def clean_text(txt):
  #clean, stem and lemmatize strings in text
  txt=txt.translate(str.maketrans('','',string.punctuation)).lower()
  txt=[lemmatizer.lemmatize(token) for token in txt.split(" ")]
  txt=[lemmatizer.lemmatize(token,"v") for token in txt]
  txt=[word for word in txt if not word in stop_words]
  txt=" ".join(txt)
  return txt

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

def labeler(x):
  #assign a label to each review given it's helpfulness ratio
  #0 - Not Helpful
  if 0<=x['vote_ratio']<0.20:
    return 0
  #1 - Helpful
  elif 0.2<x['vote_ratio']:
    return 1

balanced_table=dataset
balanced_table['helpful']=dataset.apply(labeler,axis=1)

_helpful_=balanced_table.loc[balanced_table['helpful']==1].sample(20000, replace=True)
_not_helpful=balanced_table.loc[balanced_table['helpful']==0].sample(20000)

balanced_table=_helpful_.append(_not_helpful)

#split data into features and labels
x_raw,y_raw=np.array(balanced_table.reviewText.values), np.array(balanced_table.helpful.values)
if len(x_raw)!=len(y_raw):
  raise ValueError("!MISMATCHED FEATURES & LABELS!")

x_clean=np.vectorize(clean_text)(x_raw.astype(str))

#split cleaned data into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x_clean,y_raw,test_size=0.2)

x_train,x_val,y_train,y_val=train_test_split(x_train, y_train, test_size=0.2)

train_dataset=tf.data.Dataset.from_tensor_slices((x_train,y_train))

val_dataset=tf.data.Dataset.from_tensor_slices((x_val,y_val))

test_dataset=tf.data.Dataset.from_tensor_slices((x_test,y_test))

embedding_url = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding_url, input_shape=[], 
                           dtype=tf.string, trainable=True)

model=tf.keras.Sequential()
model.add(hub_layer)

model.add(tf.keras.layers.Dense(16,activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.summary()

import keras.backend as K

def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy',f1_metric])

history=model.fit(train_dataset.shuffle(10000).batch(512),epochs=10,validation_data=val_dataset.batch(512),verbose=1)

results = model.evaluate(test_dataset.batch(512), verbose=2)

import matplotlib.pyplot as plt
#Plotting Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('BOW Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc="lower right")
plt.show()

#Plotting Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('BOW Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc="lower right")
plt.show()

#Plotting Loss
plt.plot(history.history['f1_metric'])
plt.plot(history.history['val_f1_metric'])
plt.title('BOW Model F1 Scores')
plt.ylabel('F1 Score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc="lower right")
plt.show()

model.save('./BOWClassification')

review="wow! This product is great!"
model.predict([review])

tf.keras.utils.plot_model(model)
