import pandas as pd
import joblib
import os
from tf_keras.models import load_model
from tf_keras import backend as K
import tensorflow as tf
import numpy as np
# Removed 'import google.generativeai as genai' as it's not directly used here for prediction

# Import preprocess_data from preprocessor.py
from preprocessor import preprocess_data

# Re-define focal_loss for loading model if it was used in training
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        eps = K.epsilon()
        y_pred = K.clip(y_pred, eps, 1. - eps)
        pt_1 = tf.where(K.equal(y_true, 1), y_pred, K.ones_like(y_pred))
        pt_0 = tf.where(K.equal(y_true, 0), y_pred, K.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

def load_all_models():
    """
    Loads all trained models, the scaler, and the training column names.
    It determines the models directory relative to the current file.
    Includes verbose logging for debugging deployment issues.
    Returns:
        dict: A dictionary containing loaded models, the scaler, and X_train_columns.
    """
    loaded_assets = {}
    
    # Get the directory of the current script (model_predictor.py)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Assume 'models' directory is one level up from 'src' (at the project root)
    project_root = os.path.dirname(current_script_dir)
    models_dir = os.path.join(project_root, 'models')

    print(f"DEBUG: Attempting to load models from calculated path: {models_dir}")
    print(f"DEBUG: Current working directory (from os.getcwd()): {os.getcwd()}")
    print(f"DEBUG: Contents of current_script_dir ({current_script_dir}): {os.listdir(current_script_dir) if os.path.exists(current_script_dir) else 'Path does not exist'}")
    print(f"DEBUG: Contents of project_root ({project_root}): {os.listdir(project_root) if os.path.exists(project_root) else 'Path does not exist'}")
    print(f"DEBUG: Contents of models_dir ({models_dir}): {os.listdir(models_dir) if os.path.exists(models_dir) else 'Path does not exist'}")


    # Define paths to models, scaler, and X_train_columns
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    xgb_path = os.path.join(models_dir, 'xgb_model.joblib')
    # Paths now correctly look for .keras files
    ann_class_weights_path = os.path.join(models_dir, 'ann_class_weights_model.keras')
    ann_smote_path = os.path.join(models_dir, 'ann_smote_model.keras')
    ann_focal_loss_path = os.path.join(models_dir, 'ann_focal_loss_model.keras')
    x_train_cols_path = os.path.join(models_dir, 'X_train_columns.pkl') # Path for training columns

    custom_objects = {'focal_loss_fixed': focal_loss()} # Required for loading ANN models with custom loss

    # Load scaler
    try:
        if os.path.exists(scaler_path):
            loaded_assets['scaler'] = joblib.load(scaler_path)
            print(f"DEBUG: Successfully loaded scaler from {scaler_path}")
        else:
            print(f"ERROR: Scaler file not found at expected path: {scaler_path}")
    except Exception as e:
        print(f"ERROR: Exception loading scaler: {e}")

    # Load XGBoost model
    try:
        if os.path.exists(xgb_path):
            loaded_assets['xgb_smote'] = joblib.load(xgb_path)
            print(f"DEBUG: Successfully loaded XGBoost model from {xgb_path}")
        else:
            print(f"ERROR: XGBoost model file not found at expected path: {xgb_path}")
    except Exception as e:
        print(f"ERROR: Exception loading XGBoost model: {e}")

    # Load ANN models
    ann_models_to_load = {
        'ann_class_weights': ann_class_weights_path,
        'ann_smote': ann_smote_path,
        'ann_focal_loss': ann_focal_loss_path
    }

    for key, path in ann_models_to_load.items():
        try:
            if os.path.exists(path):
                K.clear_session()
                loaded_assets[key] = load_model(path, custom_objects=custom_objects)
                print(f"DEBUG: Successfully loaded {key} model from {path}")
            else:
                print(f"ERROR: {key} model file not found at expected path: {path}")
        except Exception as e:
            print(f"ERROR: Exception loading {key} model from {path}: {e}")
            K.clear_session()
            
    # Ensure models are loaded and compiled (if ANN)
    for model_key in ['ann_class_weights', 'ann_smote', 'ann_focal_loss']:
        if model_key in loaded_assets and loaded_assets[model_key] is not None:
            try:
                if not hasattr(loaded_assets[model_key], 'optimizer') or loaded_assets[model_key].optimizer is None:
                    loaded_assets[model_key].compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    print(f"DEBUG: Re-compiled {model_key} for deployment stability.")
            except Exception as e:
                print(f"WARNING: Could not re-compile {model_key}: {e}")

    # Load X_train_columns
    try:
        if os.path.exists(x_train_cols_path):
            loaded_assets['X_train_columns'] = joblib.load(x_train_cols_path)
            print(f"DEBUG: Successfully loaded X_train_columns from {x_train_cols_path}")
        else:
            print(f"ERROR: X_train_columns file not found at expected path: {x_train_cols_path}")
    except Exception as e:
        print(f"ERROR: Exception loading X_train_columns: {e}")

    return loaded_assets


# Removed get_gemini_recommendations as it's now handled directly in app.py for streaming.

def predict_churn(model, customer_df, scaler, X_train_columns, model_type='xgb'): # Removed gemini_model argument
    """
    Makes a churn prediction for a single customer using the given model.
    Args:
        model: The trained churn prediction model (XGBoost or Keras ANN).
        customer_df (pd.DataFrame): DataFrame with a single customer's data.
        scaler: The fitted StandardScaler.
        X_train_columns (list): List of columns from the training data.
        model_type (str): Type of model ('xgb' or 'ann').
    Returns:
        tuple: (prediction (0 or 1), probability, None) - Recommendations are generated separately.
    """
    if customer_df.empty:
        return 0, 0.0, "Customer data is empty."

    # Preprocess the single customer data
    customer_processed, _, _, _, _, _ = preprocess_data(
        customer_df, is_training=False, scaler=scaler, X_train_columns=X_train_columns
    )

    if customer_processed is None or customer_processed.size == 0:
        return 0, 0.0, "Preprocessing failed or resulted in empty data."

    prediction = 0
    probability = 0.0

    # Ensure customer_processed is 2D for prediction
    if isinstance(customer_processed, np.ndarray) and customer_processed.ndim == 1:
        customer_processed = customer_processed.reshape(1, -1)
    elif isinstance(customer_processed, pd.Series):
        customer_processed = pd.DataFrame([customer_processed])


    try:
        if model_type == 'xgb':
            probability = model.predict_proba(customer_processed)[:, 1][0]
            prediction = (probability > 0.5).astype(int)
        elif model_type == 'ann':
            probability = model.predict(customer_processed)[0][0]
            prediction = (probability > 0.5).astype(int)
        else:
            return 0, 0.0, "Invalid model type specified."

    except Exception as e:
        print(f"Error during prediction with {model_type} model: {e}")
        return 0, 0.0, f"Error during prediction: {e}"

    return prediction, probability, None # Recommendations are generated in app.py, so return None here.