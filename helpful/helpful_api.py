import flask
import string
import nltk
from nltk.corpus import stopwords
from flask import Flask
from flask_restful import Api, Resource, reqparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

app=Flask(__name__)
#API=Api(app)
model=None

#model = keras.models.load_model('/content/drive/MyDrive/data/AmazonModel_1')

def load_model():
    global model
    model=keras.models.load_model('./BOWClassification')

def gloVe_maps(file):
  words_to_idx, idx_to_words, words_to_gloVe={}, {}, {}
  words=set()
  with open(file, 'r') as f:
    for l in f:
      l=(l.strip().split())
      words.add(l[0])
      words_to_gloVe[l[0]]=np.array(l[1:],dtype=np.float64)
    idx=1
    for word in sorted(words):
      words_to_idx[word]=idx
      idx_to_words[idx]=word
      idx+=1

    return words_to_idx,idx_to_words,words_to_gloVe

def sentence_indices(x, words_to_idx,maxlen):
  """
  Create a list of indices for the for each sentence
  """
  m=len(x)
  idx_matrix=np.zeros((m,maxlen))
  for i in range(m):
    tokens=str(x[i]).split()
    j=0
    for t in tokens:
      if words_to_idx.get(t) is not None:
        idx_matrix[i,j]=words_to_idx.get(t)
        j=j+1
      if j>= maxlen:
        break

  return idx_matrix

def prepare_data(txt):
    lemmatizer=WordNetLemmatizer()
    stop_words=set(stopwords.words("english"))
    txt=txt.lower()
    txt=txt.translate(str.maketrans('','', string.punctuation))
    txt=[lemmatizer.lemmatize(token) for token in txt.split(" ")]
    txt=[lemmatizer.lemmatize(token,"v") for token in txt]
    txt=[word for word in txt if not word in stop_words]
    processed_review=" ".join(txt)

    '''
    Text-to-indice
    '''
    processed_review=[processed_review]
    #review_idx=sentence_indices(processed_review,words_to_idx,50)

    return processed_review

@app.route("/predict", methods=["POST"])
def predict():
  data={"success":False}

  if flask.request.method=="POST":
    review=flask.request.form['review']
    review=prepare_data(review)
    pred=model.predict(review)
    if pred[0] > 0.5:
        data["prediction"]="HELPFUL!"
    if pred[0]<=0.5:
        data["prediction"]="NOT HELPFUL!"
    data["success"]=True

  return flask.jsonify(data)



if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    #words_to_idx,idx_to_word,word_to_gloVe=gloVe_maps("/Users/toluwafayemi/Downloads/glove.6B/glove.6B.100d.txt")
    app.run()
