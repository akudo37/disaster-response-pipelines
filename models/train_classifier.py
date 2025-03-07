#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML pipeline that trains classifier and saves
"""
# import libraries
import sys
from sqlalchemy import create_engine
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

# custom function and class
from text_preprocess import tokenize, TextLengthExtractor


def load_data(database_filepath):
    '''Load cleaned data from database into dataframe.

    Parameters
    ----------
    database_filepath : string
        path to loading database(db) file

    Returns
    -------
    X: dataframe in shape = [n_samples, n_features]
        The input samples.

    Y: dataframe in shape = [n_samples, n_outputs]
        The target values.

    category_names: list in shape = [n_outputs,]
        The column names of target values.
    '''

    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('response', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    # extract category names
    category_names = Y.columns.values

    return X, Y, category_names


def build_model():
    '''Build machine learning model.

    Count vectorize, Tfidf transform, and then classify with LinearSVC.
    Added message length as an additional feature.
    Grid search is applied.

    Returns
    -------
    model: pipeline
    '''

    pipeline = Pipeline([
        ('feature', FeatureUnion([
            ('token', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('length', Pipeline([
                # custom transformer imported for using message length
                ('extract', TextLengthExtractor()),

                # min max scaling before using in classifier
                ('scl', MinMaxScaler())
            ]))
        ])),
        ('clf', MultiOutputClassifier(
                OneVsRestClassifier(LinearSVC()), n_jobs=-1))
    ])

    # grid search parameters (using optimized result from preliminary work)
    parameters = {
        'feature__token__vect__ngram_range': [(1, 1)],
        'feature__token__tfidf__use_idf': [True],
        'clf__estimator__estimator__C': [1]
    }

    cv = GridSearchCV(pipeline, parameters, verbose=2)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''Evaluate model.

    Predict output for test data.
    Output f1 score, precision and recall for the test set for each category.

    Parameters
    ----------
    model: pipeline
        Evaluating model.

    X_test: dataframe in shape = [n_samples, n_features]
        The testing input samples.

    Y_test: dataframe in shape = [n_samples, n_outputs]
        The target values.

    category_names: list in shape = [n_outputs,]
        The column names of target values.
    '''

    Y_pred = model.predict(X_test)

    for i, name in enumerate(category_names):
        print(name, classification_report(Y_test.values[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    '''Save model to pickle file.

    Parameters
    ----------
    model: pipeline
        Evaluated model.

    model_filepath: string
        path to saving pickle file
    '''

    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
