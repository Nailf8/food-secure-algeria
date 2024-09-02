# Introduction 
Machine learning project to provide a document issue date prediction model.
This project contains scripts to deploy the model into an AzureML endpoint.

# Project structure

```
├── README.md               <- The top-level README for developers using this project
├── data                    <- All data including pdf, images and parquet files 
├── src                     <- Contains all the codes necessary to deploy the app
    ├── notebooks           <- Notebooks for evaluation, index creation and experiments
    ├── scripts             <- Contains classes needed for the conversational agent, SQL agent and router
    └── app.py              <- Srcipt to run the app
├── tests                   <- Contains unit tests for some components
    ├── test_router.py      <- Test the router 
    └── test_sql_agent.py   <- Test the sql agent queries
├── .env                    <- File to add environment variables
├── demo.db                 <- SQLite database for SQL agent demo
├── Makefile                <- Contains commands to facilitate the creation of environments, the app deployment...
├── requirements.txt        <- Contains the python packages needed to run the app
└── setup.py                <- To set up local package
```

# Prerequisites

Make sure you have installed make

# Quick Start
- Create the local environment:

  For a Conda env use
  > make initconda
  
  For a Venv env use
  > make initvenv

- activate the environment (specified after env creation)
- run the fastAPI locally
  > make run-app

