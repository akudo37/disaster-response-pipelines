import json
import plotly
import pandas as pd
import numpy as np

#from nltk.stem import WordNetLemmatizer
#from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

# necessary for loading dependencies in the same name as the model is built
import sys
sys.path.append('../models')

# using my own tokenizer instead of the original provided
# also using custom transformer for getting message length
from text_preprocess import tokenize, TextLengthExtractor


app = Flask(__name__)

# using my own tokenizer (imported) instead of the original provided as follows
'''def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
'''

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('response', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    #genre_counts = df.groupby('genre').count()['message']
    #genre_names = list(genre_counts.index)

    # graph 1: message hit count distribution by category
    category_names = list(df.columns.values)[4:]
    category_hit_counts = [df[col].sum() for col in list(category_names)]
    
    category_names = list(np.array(category_names)[np.argsort(category_hit_counts)])[::-1] # sort
    category_hit_counts = sorted(category_hit_counts, reverse=True) # sort

    # graph 2: message hit count distribution by genre
    df_melt = df.melt(id_vars=['id', 'genre'
                              ], value_vars=category_names, var_name='category', value_name='hit_count')
    genre_hit_counts = df_melt.groupby('genre').sum()['hit_count']
    genre_names = list(genre_hit_counts.index)

    genre_names = list(np.array(genre_names)[np.argsort(genre_hit_counts)])[::-1] # sort
    genre_hit_counts = sorted(genre_hit_counts, reverse=True) # sort

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        # graph 1
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_hit_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Category Hits',
                'yaxis': {
                    'title': "Hit Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': '45'
                },
                'margin': {
                    'l': '80',  # default 80
                    'r': '80',  # default 80
                    't': '100',  # default 100
                    'b': '150',  # default 80
                    'pad': '0'  # default 0
                }
            }
        },
        
        # graph 2
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_hit_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Genre Hits',
                'yaxis': {
                    'title': "Hit Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
