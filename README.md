# Customer Churn Prediction Dashboard - ABC Telecom

## Overview

This project develops an interactive Customer Churn Prediction Dashboard designed for **ABC Telecom**. It leverages advanced machine learning models to predict customer churn probability and integrates cutting-edge Generative AI (LLM) capabilities to provide actionable, personalized retention strategies and contextual insights for customer service representatives.

The dashboard aims to empower customer service teams by providing real-time churn risk assessments and AI-driven recommendations, enabling proactive engagement and improved customer retention.

## Key Features

* **Customer Search & Selection:** Easily find customers by ID, Name, or Phone Number to load their profiles.
* **Manual Data Entry:** Option to manually input customer features for ad-hoc churn predictions.
* **Ensemble Churn Prediction:** Utilizes an ensemble of robust Machine Learning models (XGBoost, Artificial Neural Networks with various techniques like SMOTE and Class Weights) to predict churn likelihood.
* **Visualized Predictions:** Presents churn probabilities and ensemble model votes through intuitive charts.
* **AI-Powered Retention Recommendations:** Integrates **Google Gemini 1.5 Flash (a Large Language Model - LLM)** to generate tailored retention strategies, potential churn drivers, and key talking points based on the customer's profile and predicted churn risk.
* **Contextual Customer Q&A (AI Agent Functionality):** Allows customer representatives to ask natural language questions about a specific customer's profile, with the LLM providing direct, data-driven answers. This acts as an intelligent assistant for information retrieval.
* **Data Overview:** Provides insights into the underlying customer data distributions used for model training.
* **Model Training & Management Guidance:** Instructions on how to re-train the ML models to keep them updated with fresh data.

## Technical Stack

* **Frontend & Application Framework:** Streamlit (Python)
* **Machine Learning:** XGBoost, TensorFlow/Keras (for ANNs), scikit-learn
* **Data Manipulation:** Pandas, NumPy
* **Generative AI / LLM:** Google Gemini 1.5 Flash API
* **Plotting:** Plotly Express
* **Serialization:** Joblib

## Setup and Installation

Follow these steps to get the project up and running on your local machine.

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)

### 1. Clone the Repository

```bash
git clone <repository_url> # Replace with your actual repository URL
cd churn_app_01            # Navigate into your project directory

### 2. Set Up Python Environment

```bash
* It's highly recommended to use a virtual environment to manage dependencies.

python -m venv churn_env
source churn_env/bin/activate  # On macOS/Linux
# churn_env\Scripts\activate   # On Windows