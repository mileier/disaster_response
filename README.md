# Disaster Response Pipeline Project

Make sure to have all the imports installed prior to trying out and executing this project:
- nltk
- json
- Flask
- pickle
- plotly
- sklearn
- joblib
- sqlalchemy


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to the address that after * Running on. For example http://10.203.16.91:3001/
