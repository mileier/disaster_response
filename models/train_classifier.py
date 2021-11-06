import pandas as pd
import numpy as np
import nltk
import sys
import re
import pickle
from sqlalchemy import create_engine
from datetime import datetime

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score, fbeta_score, precision_score, recall_score

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):	
	'''
	INPUT: 
	database_filepath - the location of a sqlite database file containing 
	                    merged and cleaned disaster recovery messages

	OUTPUT:
	X - the translated disaster recovery messages
	y - the given/correct output for these recovery messages in 36 categories
	y.columns - the category names
	'''
	engine = create_engine('sqlite:///'+database_filepath)
	df = pd.read_sql_table("message_categories", con = engine)
	df['related'] = df.related.map({0:0, 1:1, 2:1}) 
	X = df['message']
	y = df.loc[:, 'related':'direct_report']
	return X, y, y.columns

def tokenize(text):	
	'''
	INPUT: 
	text - text that should be stripped of punctuation marks and separated into words.
	       the words are then lemmaticed (organiced by root words) for all nouns (default), 
		   as well as verbs and adjectives. This step helps to improve the ML result

	OUTPUT:
	clean_tokens - a list of base words contained and hidden in the input text
	'''
	stop_words = stopwords.words("english")
	lemmatizer = WordNetLemmatizer()
	# normalize case and remove punctuation
	text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
	# tokenize text
	words = word_tokenize(text)
    
	# lemmatize andremove stop words
	clean_tokens = []
	for word in words:
		if word not in stop_words:
			clean_tok = lemmatizer.lemmatize(word)
			clean_tok = lemmatizer.lemmatize(clean_tok, pos='v')  
			clean_tok = lemmatizer.lemmatize(clean_tok, pos='a')        
			clean_tokens.append(clean_tok)
	return clean_tokens

def build_model():
	'''
	OUTPUT:
	model - machine learning pipeline model. 
			The model will expect a text column as input.
			This text is first tokenized and then transformed into a 
			term-frequency times inversed document-frequency.
			The Classifier used here is a AdaBoost implementation.
	'''
	pipeline = Pipeline([
		('vect', CountVectorizer(tokenizer=tokenize)),
		('tfidf', TfidfTransformer()),
		('clf', MultiOutputClassifier(AdaBoostClassifier())) 
	])
	
	parameters = {
		'vect__max_df':  [0.8, 0.9],
		'clf__estimator__learning_rate': [0.6, 0.7]
	}
	model = GridSearchCV(pipeline, param_grid = parameters, verbose=3)
	
	return model


def evaluate_model(model, X_test, y_test, category_names):
	'''
	Will print the aggregated average of F1-Score, F0.5-Score, Precision, 
	Recall, and Accuracy on the 36 predicted categories

	INPUT: 
	model  - machine learning model
	X_test - messages within the test-portion of the dataset
	y_test - the given/correct output for these messages in 36 categories
	category_names - the category names, or y.columns

	'''
	y_pred    = model.predict(X_test)
	y_pred_df = pd.DataFrame(data=y_pred, columns=category_names, index=y_test.index)

	f1 = 0.0
	fb = 0.0
	prec = 0.0
	recall = 0.0
	accuracy = 0.0

	for col in category_names:
		f1       += f1_score(y_test[col], y_pred_df[col], average='weighted')
		fb       += fbeta_score(y_test[col], y_pred_df[col], beta=0.5, average='weighted')
		prec     += precision_score(y_test[col], y_pred_df[col], average='weighted')
		recall   += recall_score(y_test[col], y_pred_df[col], average='weighted')
		accuracy += accuracy_score(y_test[col], y_pred_df[col])
		
	f1_avg = f1/y_test.shape[1]
	fb_avg = fb/y_test.shape[1]
	prec_avg = prec/y_test.shape[1]
	recall_avg = recall/y_test.shape[1]
	accuracy_avg = accuracy/y_test.shape[1]
	print(f"F1-Score:   {f1_avg}")
	print(f"F0.5-Score: {fb_avg}")
	print(f"Precision:  {prec_avg}")
	print(f"Recall:     {recall_avg}")
	print(f"Accuracy:   {accuracy_avg}")


def save_model(model, model_filepath):	
	'''
	Will serialize the ML model object and save it in pickle-format to the given model_filepath
	
	INPUT: 
	model  - machine learning model
	model_filepath - filepath to save the model to
	'''

	pickle.dump(model, open(model_filepath, 'wb'))


def main():
	'''
	This main function starts to read in user arguments for the different 
	filepaths for the disaster messages database and the filepath 
	to determine were the resulting model is to be stored.
	
	Using these filepaths, first data will be retrieved from the database_filepath.
	This data will then be split into training and testing, as well as input and result datasets.
	Consequently, a ML model will be build, trained and evaluated.
	The resulting model will then be saved in pickle format to the given filepath.
	'''
	if len(sys.argv) == 3:
		database_filepath, model_filepath = sys.argv[1:]
		print('Loading data...\n    DATABASE: {}'.format(database_filepath))
		
		X, y, category_names = load_data(database_filepath)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
		print('Building model...')
		model = build_model()
        
		print('Training model...')
		model.fit(X_train, y_train)
		print(f"Best Grid Search Parameters: {model.best_params_}")
        
		print('Evaluating model...')
		evaluate_model(model, X_test, y_test, category_names)

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
