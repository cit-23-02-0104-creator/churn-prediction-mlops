 Customer Churn Prediction — MLOps Pipeline

<p align="center">
  <b>An end-to-end Machine Learning Operations (MLOps) project for customer churn prediction.</b><br>
  Data Pipeline • Model Training • MLflow • DVC • Airflow • FastAPI • Docker
</p>

<p align="center">
  <a href="https://github.com/cit-23-02-0104-creator/churn-prediction-mlops">🏠 Repository</a> •
  <a href="https://github.com/cit-23-02-0104-creator/churn-prediction-mlops/tree/main/src">💻 Source Code</a> •
  <a href="https://github.com/cit-23-02-0104-creator/churn-prediction-mlops/issues">🐛 Issues</a>
</p>

📌 Overview

Customer churn is a major challenge for telecom and subscription-based businesses. This project demonstrates an end-to-end MLOps workflow for preparing customer data, training machine learning models, evaluating performance, tracking experiments, orchestrating workflows, and serving predictions through a REST API.

What this project does

Predicts whether a customer is likely to churn

Calculates churn probability

Compares machine learning models

Tracks experiments with MLflow

Reproduces the ML pipeline with DVC

Orchestrates workflow tasks with Apache Airflow

Serves predictions using FastAPI

Supports Docker-based execution

Generates model evaluation artifacts

Includes an optional LLM-based retention-offer demonstration

✨ Features

Feature

Description

📥 Data Ingestion

Prepares the customer churn dataset

🧹 Preprocessing

Cleans and transforms customer data

🤖 Model Training

Trains and compares classification models

📊 Evaluation

Calculates classification metrics

🧪 MLflow

Tracks experiments and model metrics

🔄 DVC

Reproduces the ML workflow

⚙️ Airflow

Automates pipeline execution

🌐 FastAPI

Provides REST API prediction endpoints

🐳 Docker

Supports containerized deployment

💡 LLM Bonus

Generates retention-offer ideas

🛠️ Tech Stack

Category

Technologies

Programming

Python

Data Processing

pandas

Machine Learning

scikit-learn, XGBoost

Experiment Tracking

MLflow

Pipeline Versioning

DVC

Workflow Orchestration

Apache Airflow / Astronomer

API

FastAPI, Uvicorn

Containerization

Docker

Visualization

Matplotlib, Seaborn

🏗️ Architecture

                    ┌─────────────────────┐
                    │   Customer Dataset  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Data Ingestion    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Preprocessing     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Model Training    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Model Evaluation  │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
              ┌───────────┐        ┌─────────────┐
              │  MLflow   │        │ DVC / DAG   │
              └───────────┘        └─────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │   REST Prediction   │
                    └─────────────────────┘

🔄 MLOps Pipeline

The project uses a reproducible pipeline for the machine learning lifecycle:

Data Ingestion
      ↓
Data Preprocessing
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Artifact
      ↓
FastAPI Prediction

Pipeline stages

Data Ingestion — prepares the raw customer data.

Preprocessing — cleans and transforms the dataset.

Model Training — trains the configured classification models.

Evaluation — calculates performance metrics and creates artifacts.

Experiment Tracking — logs experiments and metrics using MLflow.

Pipeline Reproduction — manages stages using DVC.

Workflow Orchestration — automates the process using Airflow.

Inference — exposes the trained model through FastAPI.

📂 Project Structure

churn-prediction-mlops/
├── src/
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── dags/
│   ├── churn_dag.py
│   └── exampledag.py
├── models/
│   ├── model.pkl
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── tests/
├── app.py
├── bonus_llm.py
├── dvc.yaml
├── dvc.lock
├── requirements.txt
├── Dockerfile
├── metrics.txt
└── README.md

🔗 Browse the project files

 Source Code

 Airflow DAGs

 Tests
 
 Models & Artifacts

 FastAPI Application

 DVC Pipeline

 Requirements

Dockerfile

 Machine Learning

The project evaluates classification models and records their performance.

Models

Logistic Regression — interpretable classification baseline

Random Forest — ensemble of decision trees

XGBoost — gradient boosting algorithm for tabular data

Model performance can be inspected through the generated evaluation metrics and MLflow experiment runs.

📊 Model Evaluation

The evaluation workflow includes commonly used classification metrics:

Accuracy

Precision

Recall

F1 Score

ROC-AUC

Visual artifacts include:

Confusion Matrix

ROC Curve

📈 Evaluation Artifacts

Confusion Matrix

ROC Curve

Metrics

🧪 MLflow Experiment Tracking

MLflow is used to record machine learning experiments, parameters, metrics and artifacts.

Start MLflow locally:

mlflow ui

Then open:

http://localhost:5000

Official documentation:

https://mlflow.org/docs/latest/

🔄 DVC

DVC is used to define and reproduce the machine learning pipeline.

Run the pipeline

dvc repro

Run individual stages

python src/data_ingestion.py
python src/preprocessing.py
python src/train.py
python src/evaluate.py

Useful files:

dvc.yaml

dvc.lock

Official documentation:

https://dvc.org/doc

⚙️ Apache Airflow

The project contains Airflow DAG definitions for workflow orchestration.

Start Astronomer

astro dev start

Airflow UI:

http://localhost:8080

Then open the project DAG and trigger the workflow.

Open Airflow DAG folder

Official documentation:

https://airflow.apache.org/docs/

🌐 FastAPI REST API

The trained model is exposed through a FastAPI service.

Start the API

python app.py

API:

http://localhost:8000

Swagger UI:

http://localhost:8000/docs

⚠️ The localhost URLs work only when the API is running on your own computer. They are not permanent GitHub links.

🔗 API Source Code

Open app.py on GitHub

🔌 API Example

POST /predict

Example request:

{
  "tenure": 12,
  "MonthlyCharges": 70.5,
  "TotalCharges": 840.0
}

Example response:

{
  "churn_probability": 0.42,
  "prediction": "No"
}

Use the Swagger interface at:

http://localhost:8000/docs

to test the endpoint interactively.

🧠 Bonus — LLM Retention Offer Generator

The optional LLM demonstration is available in:

bonus_llm.py

Run:

python bonus_llm.py

This demonstrates how an LLM can be used to generate personalized retention-offer ideas for customers identified as being at risk.

🐳 Docker

The project includes a Dockerfile for containerized execution.

Build

docker build -t churn-prediction-mlops .

Run

docker run -p 8000:8000 churn-prediction-mlops

Then open:

http://localhost:8000/docs

Open Dockerfile

Official documentation:

https://docs.docker.com/

🚀 Installation

Prerequisites

Python 3.9+

Git

Docker — optional

DVC — optional

Astronomer CLI — required for local Astro Airflow workflow

Clone the repository

git clone https://github.com/cit-23-02-0104-creator/churn-prediction-mlops.git
cd churn-prediction-mlops

Create virtual environment

Windows

python -m venv venv
venv\Scripts\activate

Linux / macOS

python3 -m venv venv
source venv/bin/activate

Install dependencies

pip install -r requirements.txt

⚡ Quick Start

git clone https://github.com/cit-23-02-0104-creator/churn-prediction-mlops.git
cd churn-prediction-mlops

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

dvc repro

python app.py

Then open:

http://localhost:8000/docs

🔐 Security

Never commit sensitive information such as:

API keys

Passwords

Access tokens

Private credentials

.env files containing secrets

Use environment variables for sensitive configuration.

📚 Useful Project Links

Resource

Link

🏠 Repository

GitHub Repository

💻 Source Code

src

⚙️ Airflow

dags

🧪 Tests

tests

🤖 Models

models

🌐 API

app.py

🔄 DVC

dvc.yaml

📦 Dependencies

requirements.txt

🐳 Docker

Dockerfile

👩‍💻 Author

Jayani Samarakoon

GitHub:
https://github.com/cit-23-02-0104-creator

Project Repository:
https://github.com/cit-23-02-0104-creator/churn-prediction-mlops

⭐ Project Goal

This project demonstrates practical MLOps engineering practices by connecting machine learning development with reproducibility, experiment tracking, workflow automation, model evaluation and API deployment.

If you find this project useful, consider giving the repository a ⭐.

📄 License

This project is intended for educational and portfolio p
