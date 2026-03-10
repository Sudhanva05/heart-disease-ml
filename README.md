# Heart Disease Prediction System

An end-to-end Machine Learning project that predicts the likelihood of heart disease using clinical patient data.

## Project Overview

This project demonstrates a complete ML workflow including:

* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Model training and comparison
* Model deployment using FastAPI
* REST API for real-time prediction

The best performing model achieved **91.8% accuracy**.

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* FastAPI
* Uvicorn
* Joblib

---

## Project Structure

```
heart-disease-ml
│
├── data/               # Dataset
├── notebooks/          # EDA notebook
├── src/                # Training scripts
│   └── train_model.py
│
├── api/                # FastAPI service
│   └── main.py
│
├── models/             # Saved model
│
├── requirements.txt
└── README.md
```

---

## Model Performance

| Model               | Accuracy  |
| ------------------- | --------- |
| KNN                 | **91.8%** |
| SVM                 | 90.2%     |
| Logistic Regression | 88.5%     |
| Random Forest       | 88.5%     |

---

## Running the Project

### Install dependencies

```
pip install -r requirements.txt
```

### Train the model

```
python src/train_model.py
```

### Start API

```
uvicorn api.main:app --reload
```

---

## API Documentation

FastAPI automatically generates interactive docs:

```
http://127.0.0.1:8000/docs
```

---

## Example Prediction Request

```
POST /predict
```

Example input:

```
{
 "age": 63,
 "sex": 1,
 "cp": 3,
 "trestbps": 145,
 "chol": 233,
 "fbs": 1,
 "restecg": 0,
 "thalach": 150,
 "exang": 0,
 "oldpeak": 2.3,
 "slope": 0,
 "ca": 0,
 "thal": 1
}
```

Response:

```
{
 "prediction": 0,
 "result": "No Heart Disease"
}
```

---

## Future Improvements

* Model hyperparameter tuning
* Docker containerization
* Web UI for predictions
* Cloud deployment

---

## Author

Sudhanva J Rao
