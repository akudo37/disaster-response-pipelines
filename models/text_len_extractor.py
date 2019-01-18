#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text length extractor (custom transformer)
"""
# import libraries
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
