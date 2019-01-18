#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETL pipeline that cleans data and stores in database
"""
# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''Load message file and categories file.

    Load, merge, split 'categories' and convert to 0 or 1, and replace columns.

    Parameters
    ----------
    messages_filepath : string
        path to message data csv file

    categories_filepath : string
        path to category data csv file

    Returns
    -------
    merged data: dataframe

    '''

    # load message dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages.drop_duplicates('id'),
                  categories, how='right', on='id')

    # create a dataframe of the 36 individual category columns
    categories = pd.DataFrame(
            categories['categories'].str.split(';', expand=True))

    # select the first row of the categories dataframe
    row = categories.iloc[0, :]

    # use this row to extract a list of new column names for categories.
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(
                categories[column], downcast='integer')

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    return df


def clean_data(df):
    '''Clean data.

    Remove duplicated rows, and clean over range category cells.

    Parameters
    ----------
    df: dataframe
        merged data

    Returns
    -------
    cleaned data: dataframe

    '''

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # fix over range category cells (turn to 0 if not in [0, 1])
    for col in df.columns[4:]:
        df[col] = df[col].apply(lambda x: 0 if x not in [0, 1] else x)

    return df


def save_data(df, database_filename):
    '''Save data.

    Save the clean dataset into an sqlilte databese.

    Parameters
    ----------
    df: dataframe
        cleaned data

    database_filename: str
        path to saving databese(db) file
    '''

    engine = create_engine('sqlite:///' + database_filename)

    df.to_sql('response', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
