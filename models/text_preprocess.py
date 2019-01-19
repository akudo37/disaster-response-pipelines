#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text preprocess function and class
- tokenize: custom function
- TextLengthExtractor: custom transformer class 
"""
# import libraries
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def tokenize(text):
    '''Tokenize text.

    Replace URL, case normalize, tokenize, remove stop words, and lemmatize.

    Parameters
    ----------
    text : string
        text string to be tokenized

    Returns
    -------
    tokenized words: list
    '''

    # replace URL with placeholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    urls = re.findall(url_regex, text)
    for url in urls:
        text = text.replace(url, 'urlplaceholder')

    # case normalize, clean, and tokenize
    text = re.sub(r'[^0-9a-zA-Z]', ' ', text.lower())
    words = text.split()

    # remove stop words
    stop_removed = [
            word for word in words if word not in stopwords.words('english')]

    # lemmatize (nouns then verbs)
    n_lemmed = [WordNetLemmatizer().lemmatize(word) for word in stop_removed]

    n_v_lemmed = [
            WordNetLemmatizer().lemmatize(word, pos='v') for word in n_lemmed]

    return n_v_lemmed


class TextLengthExtractor(BaseEstimator, TransformerMixin):
    '''Calculator of string cell length.'''

    def fit(self, X, y=None):
        '''Just return self.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, 1]
            The training input samples.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        '''

        return self

    def transform(self, X):
        '''Calculate string length of each cell in X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, 1]
            The training or testing input samples.

        Returns
        -------
        lengths : dataframe of shape = [n_samples, 1]
            Returns lengths of string cells.
        '''

        return pd.Series(X).str.len().to_frame()