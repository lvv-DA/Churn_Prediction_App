# 📉 Customer Churn Prediction and Retention System

## 🎯 Objective

The primary goal of this system is to **proactively identify customers at risk of churning** and provide **AI-powered, actionable retention strategies**. It utilizes machine learning to:

- Predict customer churn likelihood.
- Enable customer search and manual churn prediction via a web interface.
- Generate personalized recommendations to mitigate churn.
- Automate the model training and deployment pipeline.

---

## 🚀 Key Features

- 🔍 **Churn Prediction:** Uses behavioral and demographic data to forecast churn.
- 🧠 **Multiple Model Support:** Employs XGBoost and various Artificial Neural Networks (ANNs) with different imbalance handling strategies.
- 🧼 **Preprocessing Pipeline:** One-hot encoding, standard scaling, and SMOTE for training.
- 🧪 **Synthetic Data Generator:** Generates realistic but anonymized customer identifiers.
- 🌐 **Streamlit Interface:** Allows customer search, data input, and visualized results.
- 🤖 **AI Recommendations:** Integrates Google Gemini API to suggest churn prevention actions.
- 💾 **Model Management:** Automatically saves and loads models and preprocessing artifacts.

---

## 🧱 Application Structure

| Script               | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `data_loader.py`     | Loads raw datasets with robust error handling for file encoding issues.     |
| `datagenrator.py`    | Generates synthetic identifiers (CustomerID, Name, etc.) for anonymized data. |
| `preprocessor.py`    | Handles data encoding, scaling, and SMOTE application.                      |
| `model_trainer.py`   | Trains XGBoost and ANN models; saves models and preprocessing artifacts.    |
| `model_predictor.py` | Loads trained models and scalers; handles churn prediction logic.           |
| `app.py`             | Streamlit frontend with search, manual input, predictions, and recommendations. |

---

## 🧰 Tools & Technologies

### 📝 Programming
- **Python**

### 📦 Libraries
- **Data Handling:** `pandas`, `numpy`
- **ML & DL:** `scikit-learn`, `xgboost`, `imbalanced-learn`, `tensorflow/keras`
- **Web App:** `streamlit`, `plotly-express`
- **Model Storage:** `joblib`
- **AI API Integration:** `google-generativeai`
- **Utilities:** `os`, `sys`, `io`, `time`, `gc`

---

## 🔄 Data Flow

### 1. **Data Ingestion**
- `CFM KTP_Stage 1 task_churn dataset.xlsx` (original)
- `customer_churn.csv` (cleaned dataset)
- `customer_data_with_identifiers.csv` (augmented for app)

### 2. **Synthetic Data Creation**
- Generates CustomerID, Name, Phone, Address
- Saves output to `customer_data_with_identifiers.csv`

### 3. **Preprocessing**
- One-hot encoding of categorical variables
- Standard scaling of numerical features
- SMOTE for imbalance correction during training

### 4. **Model Training**
- Splits data (80/20, stratified)
- Trains:
  - XGBoost model
  - ANN models (Class Weights, SMOTE, Focal Loss)
- Saves models and preprocessing assets to `/models/`

### 5. **Prediction**
- Loads trained models and scaler
- Makes predictions on new input via `make_predictions()`

### 6. **Streamlit Web App**
- Search customer data
- Manual churn prediction
- Generate Google Gemini-based retention strategies
- Model selection for predictions

---


