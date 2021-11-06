# Disaster Response Pipeline Project

### Installations
Python, make sure to have all the imports installed (e.g. via conda or pip) prior to trying out and executing this project:
- sklearn, numpy, nltk, json, Flask, pickle, plotly, joblib, sqlalchemy
		
### Project Motivation
This is the second project as part of the Udacity Data Scientist Nanodegree.
It contains real world data provided by Figure Eight containing messages send during disaster recovery phases and their respective categories.
The goal is to train a model to auto-detect these categories.
This could be used during the next disaster recovery to forward the right subsets of the messages to the different help organizations.
I wonder how good the provided categorization actually is - I think this would actually be worth challenging.

### File Descriptions
	- data:
		- disaster_messages.csv		The messages send during past disaster recovery phases...
		- disaster_categories.csv	and their respective classification for 36 categories
		- process_data.py		Python to read in both files, merge them, and write the resulting pandas dataframe in a sqllite databank
	- models:
		- train_classifier.py		Python to read the sqllite database table containing the disaster messages and categories.
						After reading in the data, the messages get cleaned, tokenized and used as input for a machine learning pipeline.
	- app:
		- run.py			Python to provide the graphs that are to be displayed in master.html and to predict the categories of messages that were input on go.html.
		- templates: 		
			- master.html		HTML structure to provide some overview graphs on the combined messages and categories dataset
			- go.html		HTML structure to allow the input of individual messages that will then be classified on the 36 categories

Feel free to explore more of the data for yourself, by creating a fork and experimenting with the existing Jupyter notebook and the earthquake data set.

### Acknowledgements

Thanks to figure eight (https://www.figure-eight.com/) for providing the data set.
			
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to the address that after * Running on. For example http://0.0.0.0:3001/
