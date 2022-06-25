import os
import logging

import string 
from nltk.corpus import stopwords 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]


def text_process(mess):
    """ 
    Takes in a string of text, then performs the following:
    
    1. Remove all punctuation
    2. Remove all stopwords 
    3. Returns a list of the cleaned text
    """ 
    #STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure'] # Check characters to see if they are in punctuation 
    STOPWORDS = stopwords + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure'] # Check characters to see if they are in punctuation 
    nopunc = [char for char in mess if char not in string.punctuation] # Join the characters again to form the string.
    nopunc = ''.join(nopunc) # Now just remove any stopwords 
    nopunc = nopunc.lower()
    nopunc = ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])
    return nopunc

def split_text_label(df, sns_column, label_column):
    texts = np.array(df[sns_column].apply(text_process).values)

    label_dict = {'ham': 0, 'spam' :1}
    labels = np.array(df[label_column].map(label_dict).values)
    return texts, labels


class Data_Processor(object):
    def __init__(self, data_path = './data/sms_dataset.csv', sns_column='번역', label_column = '분류'):
        # dataset 
        self.data_path = data_path
        self.sns_column = sns_column
        self.label_column = label_column
        self.encoding = None
        
        # load dataset
        if self.encoding is None:
            self.df = pd.read_csv(data_path)
        else:
            self.df = pd.read_csv(data_path, encoding=self.encoding)
            
            
    def data_preproces(self, val_split=True):
        texts, labels = split_text_label(self.df, self.sns_column, self.label_column)
        
        if val_split:
            train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.05)
            return train_texts, val_texts, train_labels, val_labels
        else:
            return texts, labels