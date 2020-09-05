# Disaster_Response_Pipeline
In this project, I apply disaster data from Figure Eight to build a model for an API that classifies disaster messages.
Through the resulting Web App, an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

### File Description:
- Jupyternote book: 
1. ETL pipeline: clean and preprocessing the data
2. ML pipeline: build NLP model to classify the data
- Files:
Set up the database and model.
1.data: To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
2. models:To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
3. app: Run web app through  `python run.py`

*Note:Go to http://0.0.0.0:3001/


