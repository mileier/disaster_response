# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
	'''
	INPUT: 
	messages_filepath   - to find the correct current location of the messages .csv file 
	categories_filepath - to find the correct current location of the categories .csv file 

	OUTPUT:
	dataframe, merging the two input .csv-files containing messages and categories 
	'''
	messages   = pd.read_csv(messages_filepath)
	categories = pd.read_csv(categories_filepath) 
	return messages.merge(categories, on="id")


def clean_data(df):
	'''
	INPUT: 
	df - pandas dataframe

	OUTPUT:
	df - cleaned pandas dataframe
	'''
	# create a dataframe of the 36 individual category columns
	categories =  df["categories"].str.split(";", expand=True)
	# select the first row of the categories dataframe
	row = categories[:1]
	# list of new column names for the categories, rename the columns of `categories`
	category_colnames = [list(row[name])[0][:-2] for name in row]
	categories.columns = category_colnames
	# only keep the last char of the string per cell as an integer (0 or 1)
	for column in categories:
		# set each value to be the last character of the string
		categories[column] = categories[column].str[-1]
		# convert column from string to numeric
		categories[column] = categories[column].astype(int)
	
	# drop the old categories column
	df = df.drop(['categories'], axis=1).copy()
	# concatenate the original dataframe with the new `categories` dataframe
	df = pd.concat([df, categories], axis=1)
	# drop duplicated rows from the df dataframe
	df = df.drop_duplicates()
	
	return df

def save_data(df, database_filename):
	'''
	INPUT: 
	df - pandas dataframe to be saved into a sqlite db file

	OUTPUT:
	-
	'''
	# new sqllite dbbase named database_filename
	engine = create_engine('sqlite:///'+database_filename)
	# write merged and cleaned dataframe df into a new table
	df.to_sql('message_categories', engine, index=False, if_exists='replace')  

def main():	
	'''
	This main function starts to read in user arguments for the different 
	filepaths for the messages.csv and categories.csv, as well as one to 
	determine were the resulting sqlite database is to be stored.
	
	Using these filepaths, it first loads the messages and categories data.
	Then, the data is cleaned and processed to be ready as input for ML algorithms.
	As a last step, this cleaned data gets stored into a sqlite database file.
	'''
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
