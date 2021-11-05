import json
import plotly
import pandas as pd
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
#import sklearn.external.joblib as extjoblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
	'''
	INPUT: 
	text - text that should be separated into words and lemmaticed (organiced by root words). 

	OUTPUT:
	clean_tokens - a list of base words contained and hidden in the input text
	'''
	tokens = word_tokenize(text)
	lemmatizer = WordNetLemmatizer()

	clean_tokens = []
	for tok in tokens:
		clean_tok = lemmatizer.lemmatize(tok).lower().strip()
		clean_tokens.append(clean_tok)

	return clean_tokens
	

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('message_categories', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index(): 
	'''
	OUTPUT:
	render_template('master.html', ids=ids, graphJSON=graphJSON) - 
		a rendered webpage based on "master.html", displaying js-Plotly graphs as defined 
		in the list of graphs. Ids is a count of graphs, which gets iterated within the master.html
	'''   
	# extract data needed for visuals
	genre_counts = df.groupby('genre').count()['message']
	genre_names = list(genre_counts.index)
	#count of categories
	cats_count_sorted = (df[df.columns[9:]].sum()).sort_values(ascending = True)          
	# list of sorted categories
	categories = list(cats_count_sorted.index)      
	# create visuals
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
					x=cats_count_sorted,
					y=categories,
					orientation='h'
				)
			],

			'layout': {
				'title': 'Distribution of Categories',
				'xaxis': {
					'title': "Count",
					'automargin':'true'
				},
				'yaxis': {
					'automargin':'true'
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
	'''
	Gets the user input off of the master.html - query form.
	It classifies that input using the model and displays the results.
	
	OUTPUT:
	render_template('go.html', query=query, classification_result=classification_results)) - 
		a rendered webpage based on "go.html", passing the user input and the classification_results.
	'''   
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
