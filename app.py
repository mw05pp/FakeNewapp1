import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.externals import joblib

import pickle

import flask
from flask import Flask, request, render_template, url_for
from flask_cors import CORS

import os

import newspaper
from newspaper import Article

import urllib

news = pd.read_csv('data/news.csv')
X = news['text']
y = news['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(stop_words='english')),
                    ('nbmodel', MultinomialNB())
                    ])

pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)

print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

with open('model.pickle', 'wb') as handle:
    pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)


# initialize app
app = Flask(__name__)
CORS(app)

with open('model.pickle', 'rb') as handle:
    model = pickle.load(handle)

@app.route('/')
def index():
    return render_template('index.html')


#Receiving the input url from the user and using Web Scrapping to extract the news content
@app.route('/predict', methods=['GET','POST'])
def predict():

    article_data = {}

    url = request.get_data(as_text=True)[5:]
    url = urllib.parse.unquote(url)
    article = Article(str(url))

    if article.is_valid_url():
        article.download()
        article.parse()
        article.nlp()

        article_data = {
            'title': article.title,
            'authors': article.authors,
            'summary': article.summary,
            'publish_date': article.publish_date,
            'images': article.images,
            'videos': article.movies,
            'url': article.url
        }

    #Passing the news article to the model and returing whether it is Fake or Real
        article_data['pred'] = model.predict([ article_data['summary'] ])

    else:
        article_data['error'] = 'Something seems wrong with the link provided.'

    return render_template('index.html', article_data=article_data)


if __name__ == '__main__':
    app.run(debug=True)
