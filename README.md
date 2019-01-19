# Disaster Response Pipeline Project

### Overview
In this project, disaster data from Figure Eight is analyzed, and a model is built for an API that classifies disaster messages.

In the Project Workspace, there is a data set containing real messages that were sent during disaster events. A machine learning pipeline is created to categorize these events so that the messages can be sent to an appropriate disaster relief agency.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also display visualizations of the data.

### Instructions:
1. Run the following commands in the workspace's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Folder structure (explanations):
.
├── app
│   ├── run.py  # Flask file that runs app
│   └── templates
│       ├── master.html  # main page of web app
│       └── go.html  # classification result page of web app
├── data
│   ├── disaster_categories.csv  # data source
│   ├── disaster_messages.csv  # data source
│   ├── process_data.py  # ETL pipeline script
│   └── DisasterResponse.db  # database to save clean data to
├── models
│   ├── train_classifier.py  # ML pipeline script
│   ├── text_preprocess.py  # custom function and class
│   ├── classifier.pkl  # saved model
│   └── __init__.py  # python package init file
└── README.md
