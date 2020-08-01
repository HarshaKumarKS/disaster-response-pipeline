import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from collections import Counter
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')

app = Flask(__name__)


def tokenize(text):
    '''
    Tokenize and clean text.
    Input:
        text: message
    Output:
        clean: tokenized, cleaned text
    '''
    # Normalize
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())
    # Tokenize
    words = word_tokenize(text)
    # Remove Stopwords
    words = [w for w in words if w not in stopwords.words('english')]
    # remove short words
    words = [w for w in words if len(w) > 2]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    clean = [lemmatizer.lemmatize(w, pos='n').strip() for w in words]
    clean = [lemmatizer.lemmatize(w, pos='v').strip() for w in clean]
    return clean

def compute_word_counts(messages, load=True, filepath='../data/counts.npz'):
    '''
    input: (
        messages: list or numpy array
        load: Boolean value if load or run model
        filepath: filepath to save or load data
            )
    Function computes the top 20 words in the dataset with counts of each term
    output: (
        top_words: list
        top_counts: list
            )
    '''
    if load:
        # load arrays
        data = np.load(filepath)
        return list(data['top_words']), list(data['top_counts'])
    else:
        # get top words
        counter = Counter()
        for message in messages:
            tokens = tokenize(message)
            for token in tokens:
                counter[token] += 1
        # top 20 words
        top = counter.most_common(20)
        top_words = [word[0] for word in top]
        top_counts = [count[1] for count in top]
        # save arrays
        np.savez(filepath, top_words=top_words, top_counts=top_counts)
        return list(top_words), list(top_counts)

# load data
#engine = create_engine('sqlite:///../data/DisasterResponse.db')
engine = create_engine('sqlite:///C:/Users/Osama/Desktop/OneDrive/Online Education/Data_Scientist_Nanodegree/Project - 5 Disaster_Response/data/DisasterResponse.db')
df = pd.read_sql_table('df', engine)

# load model
#model = joblib.load("../models/classifier.pkl")
model = joblib.load("C:/Users/Osama/Desktop/OneDrive/Online Education/Data_Scientist_Nanodegree/Project - 5 Disaster_Response/models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create visuals
    # Top ten categories
    top_category_count = df.iloc[:,4:].sum().sort_values(ascending=False)[1:11]
    top_category_names = list(top_category_count.index)

    # Top 20 word counts
    word_counts_path = "C:/Users/Osama/Desktop/OneDrive/Online Education/Data_Scientist_Nanodegree/Project - 5 Disaster_Response/data/word_counts.npz"
    word_counts = compute_word_counts(df['message'].tolist(), False, word_counts_path)
    top_20_words = word_counts[0]
    top_20_counts = word_counts[1]

    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=top_category_names,
                    y=top_category_count
                )
            ],

            'layout': {
                'title': 'Top Ten Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_20_words,
                    y=top_20_counts
                )
            ],

            'layout': {
                'title': 'Top 20 Word Counts',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Words"
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
    # not passing any args to let flask automatically select host
    app.run()
    #app.run(host='0.0.0.0', port='33', debug=True)


if __name__ == '__main__':
    main()
