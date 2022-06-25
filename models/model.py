from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pickle
import logging

import numpy as np
import pandas as pd

def train_model(x, y):
    
    # CountVectorizer
    # 문서를 토큰 리스트로 변환한다.
    # 각 문서에서 토큰의 출현 빈도를 센다.
    # 각 문서를 BOW 인코딩 벡터로 변환한다.
    vect = CountVectorizer()
    vect.fit(x)
    
    # the document-term matrix
    X_dtm = vect.transform(x)
    
    # Classification model by a Multinomial Naive Bayes model
    nb = MultinomialNB()
    nb.fit(X_dtm, y)
    
    return vect, nb

def eval_model(vect, nb, x, y_true):
    X_dtm = vect.transform(x)
    y_pred = nb.predict(X_dtm)
    
    acc = metrics.accuracy_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    
    scores =[acc, recall, precision, f1]
    
    result = "Acc : {:.4%} | recall : {:.4%} | precision : {:.4%} | f1 : {:.4%}".format(*scores)
    
    print(result)
    return None


def save_model(model, filename):
    with open(filename, 'wb') as w:
        pickle.dump(model, w)
        

def predict(x, vect, nb):
    x_dtm = vect.transform(x)
    y_pred = nb.predict(x_dtm)
    return y_pred


def test_predict(file_path, model_path, save_path):
    test_df = pd.read_csv(file_path)
    test_x = test_df['번역'].values
    
    with open(model_path, 'rb') as f:
        vect, nb = pickle.load(f)
    #vect, nb = pickle.loads(model_path)
    
    y_pred = predict(test_x, vect, nb)
    test_df['분류'] = y_pred
    
    test_df.to_csv(save_path, index=False)
    
