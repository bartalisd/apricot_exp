import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB,CategoricalNB
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import re
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

def analysis(train, test):
    missing_values=train.isnull().sum()
    percent_missing = train.isnull().sum()/train.shape[0]*100
    value = {
        'missing_values ':missing_values,
        'percent_missing %':percent_missing
    }
    frame=pd.DataFrame(value)
    train=train.drop_duplicates(subset=['text', 'target'], keep='first')
    train['text_length'] = train.text.apply(lambda x: len(x.split()))
    test['text_length'] = test.text.apply(lambda x: len(x.split()))
    list_= []
    for i in train.text:
        list_ += i
    list_= ''.join(list_)
    allWords=list_.split()
    vocabulary= set(allWords)
    return frame, train, vocabulary
    
def data_cleaning(train, test):
    stopwords.words('english')
    pstem = PorterStemmer()
    def clean_text(text):
        text= text.lower()
        text= re.sub('[0-9]', '', text)
        text  = "".join([char for char in text if char not in string.punctuation])
        tokens = word_tokenize(text)
        tokens=[pstem.stem(word) for word in tokens]
        # This line below must be enabled to remove stop words (is it because original model needed the stop words too?)
        tokens=[word for word in tokens if word not in stopwords.words('english')]
        text = ' '.join(tokens)
        return text
    train["clean"]=train["text"].apply(clean_text)
    test["clean"]=test["text"].apply(clean_text)
    X=train['clean']
    Y=train['target']
    return X, Y

def tfidf(X_train, X_test, max_num_features):
    tfidf = TfidfVectorizer(sublinear_tf=True,max_features=max_num_features, min_df=1, norm='l2',  ngram_range=(1,2))
    features = tfidf.fit_transform(X_train).toarray()
    features_test = tfidf.transform(X_test).toarray()
    features_p=pd.DataFrame(features)
    features_t=features_p.transpose()
    features_test_p=pd.DataFrame(features_test)
    features_test_t=features_test_p.transpose()
    return features_t, features_test_t
