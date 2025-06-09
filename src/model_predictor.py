import pandas as pd
import joblib
import os
from tf_keras.models import load_model
from tf_keras import backend as K
import tensorflow as tf
import numpy as np

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
    loaded_assets = {}
    
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    models_dir = os.path.join(project_root, 'models')

    print(f"DEBUG_DEPLOY: Attempting to load models from calculated path: {models_dir}")
    print(f"DEBUG_DEPLOY: Current working directory (from os.getcwd()): {os.getcwd()}")
    print(f"DEBUG_DEPLOY: Contents of current_script_dir ({current_script_dir}): {os.listdir(current_script_dir) if os.path.exists(current_script_dir) else 'Path does not exist'}")
    print(f"DEBUG_DEPLOY: Contents of project_root ({project_root}): {os.listdir(project_root) if os.path.exists(project_root) else 'Path does not exist'}")
    
    # --- ADD THIS LINE FOR SUPER VERBOSE DEBUGGING ---
    print(f"DEBUG_DEPLOY: Contents of models_dir ({models_dir}): {os.listdir(models_dir) if os.path.exists(models_dir) else 'Path does not exist'} (Path existence: {os.path.exists(models_dir)})")
    # --- END OF ADDED LINE ---

    # Define paths to models, scaler, and X_train_columns
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    xgb_path = os.path.join(models_dir, 'xgb_model.joblib')
    ann_class_weights_path = os.path.join(models_dir, 'ann_class_weights_model.keras')
    ann_smote_path = os.path.join(models_dir, 'ann_smote_model.keras')
    ann_focal_loss_path = os.path.join(models_dir, 'ann_focal_loss_model.keras')
    x_train_cols_path = os.path.join(models_dir, 'X_train_columns.pkl')

    custom_objects = {'focal_loss_fixed': focal_loss()}

    # Load scaler
    try:
        if os.path.exists(scaler_path):
            loaded_assets['scaler'] = joblib.load(scaler_path)
            print(f"DEBUG_DEPLOY: Successfully loaded scaler from {scaler_path}")
        else:
            print(f"ERROR_DEPLOY: Scaler file not found at expected path: {scaler_path}")
    except Exception as e:
        print(f"ERROR_DEPLOY: Exception loading scaler: {e}")

    # Load XGBoost model
    try:
        if os.path.exists(xgb_path):
            loaded_assets['xgb_smote'] = joblib.load(xgb_path)
            print(f"DEBUG_DEPLOY: Successfully loaded XGBoost model from {xgb_path}")
        else:
            print(f"ERROR_DEPLOY: XGBoost model file not found at expected path: {xgb_path}")
    except Exception as e:
        print(f"ERROR_DEPLOY: Exception loading XGBoost model: {e}")

    # Load ANN models
    ann_models_to_load = {
        'ann_class_weights': ann_class_weights_path,
        'ann_smote': ann_smote_path,
        'ann_focal_loss': ann_focal_loss_path
    }

    for key, path in ann_models_to_load.items():
        try:
            print(f"DEBUG_DEPLOY: Checking for {key} at {path}. Exists: {os.path.exists(path)}") # ADD THIS LINE
            if os.path.exists(path):
                K.clear_session()
                loaded_assets[key] = load_model(path, custom_objects=custom_objects)
                print(f"DEBUG_DEPLOY: Successfully loaded {key} model from {path}")
            else:
                print(f"ERROR_DEPLOY: {key} model file not found at expected path: {path}")
        except Exception as e:
            print(f"ERROR_DEPLOY: Exception loading {key} model from {path}: {e}")
            K.clear_session()
            
    # Ensure models are loaded and compiled (if ANN)
    for model_key in ['ann_class_weights', 'ann_smote', 'ann_focal_loss']:
        if model_key in loaded_assets and loaded_assets[model_key] is not None:
            try:
                if not hasattr(loaded_assets[model_key], 'optimizer') or loaded_assets[model_key].optimizer is None:
                    loaded_assets[model_key].compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    print(f"DEBUG_DEPLOY: Re-compiled {model_key} for deployment stability.")
            except Exception as e:
                print(f"WARNING_DEPLOY: Could not re-compile {model_key}: {e}")

    # Load X_train_columns
    try:
        if os.path.exists(x_train_cols_path):
            loaded_assets['X_train_columns'] = joblib.load(x_train_cols_path)
            print(f"DEBUG_DEPLOY: Successfully loaded X_train_columns from {x_train_cols_path}")
        else:
            print(f"ERROR_DEPLOY: X_train_columns file not found at expected path: {x_train_cols_path}")
    except Exception as e:
        print(f"ERROR_DEPLOY: Exception loading X_train_columns: {e}")

    return loaded_assets