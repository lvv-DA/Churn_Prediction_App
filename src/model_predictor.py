import joblib
import os
import pandas as pd
from tf_keras.models import load_model # Correct import for loading Keras models
from tf_keras import backend as K
import tensorflow as tf

# Re-define focal_loss here because Keras needs access to the custom loss function
# when loading a model that was compiled with it.
# This must match the focal_loss definition in model_trainer.py EXACTLY.
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        eps = K.epsilon()
        y_pred = K.clip(y_pred, eps, 1. - eps)
        pt_1 = tf.where(K.equal(y_true, 1), y_pred, K.ones_like(y_pred))
        pt_0 = tf.where(K.equal(y_true, 0), y_pred, K.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

def load_all_models(models_dir='models'):
    """
    Loads all trained models and the scaler.
    Assumes 'models' directory is relative to the project root.
    """
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir) # Go up one level from 'src'
    actual_models_dir = os.path.join(project_root, models_dir)

    # print(f"DEBUG_DEPLOY: Attempting to load models from calculated path: {actual_models_dir}")
    # print(f"DEBUG_DEPLOY: Current working directory (from os.getcwd()): {os.getcwd()}")
    # print(f"DEBUG_DEPLOY: Contents of models_dir ({actual_models_dir}): {os.listdir(actual_models_dir)}")

    assets = {}

    try:
        scaler_path = os.path.join(actual_models_dir, 'scaler.pkl')
        assets['scaler'] = joblib.load(scaler_path)
        # print(f"DEBUG_DEPLOY: Successfully loaded scaler from {scaler_path}")
    except Exception as e:
        print(f"ERROR: Could not load scaler: {e}")

    try:
        xgb_path = os.path.join(actual_models_dir, 'xgb_model.joblib')
        assets['xgb_smote'] = joblib.load(xgb_path) # Changed key to xgb_smote for consistency in app.py
        # print(f"DEBUG_DEPLOY: Successfully loaded XGBoost model from {xgb_path}")
    except Exception as e:
        print(f"ERROR: Could not load XGBoost model: {e}")

    # Load Keras/ANN models
    ann_models = {
        'ann_class_weights': 'ann_class_weights_model.h5',
        'ann_smote': 'ann_smote_model.h5',
        'ann_focal_loss': 'ann_focal_loss_model.h5'
    }

    for key, filename in ann_models.items():
        model_path = os.path.join(actual_models_dir, filename)
        # print(f"DEBUG_DEPLOY: Checking for {key} at {model_path}. Exists: {os.path.exists(model_path)}")
        if os.path.exists(model_path):
            try:
                # Provide custom_objects if the model was compiled with a custom loss function
                custom_objects = {'focal_loss_fixed': focal_loss()} if 'focal_loss' in key else None
                assets[key] = load_model(model_path, custom_objects=custom_objects)
                # print(f"DEBUG_DEPLOY: Successfully loaded {key} model from {model_path}")
            except Exception as e:
                print(f"ERROR: Could not load {key} model from {model_path}: {e}")
        else:
            print(f"WARNING: ANN model file not found for {key}: {model_path}")


    try:
        X_train_columns_path = os.path.join(actual_models_dir, 'X_train_columns.pkl')
        assets['X_train_columns'] = joblib.load(X_train_columns_path)
        # print(f"DEBUG_DEPLOY: Successfully loaded X_train_columns from {X_train_columns_path}")
    except Exception as e:
        print(f"ERROR: Could not load X_train_columns: {e}")

    return assets


def make_predictions(model, data_df, scaler, X_train_columns, model_type='xgb'):
    """
    Makes predictions using a given model.

    Args:
        model: The trained machine learning model (XGBoost or Keras).
        data_df (pd.DataFrame): The input customer data for prediction. This DataFrame
                                 should already be one-hot encoded and aligned with X_train_columns.
        scaler (StandardScaler): The fitted StandardScaler object.
        X_train_columns (list): List of column names the model was trained on.
        model_type (str): 'xgb' for XGBoost, 'ann' for Keras/ANN.

    Returns:
        tuple: (prediction (int), probability (float), feature_names (list))
    """
    # Ensure data_df has the same columns and order as X_train_columns
    # This step was already performed in app.py before calling make_predictions,
    # but it's a good safety check if this function were called independently.
    # For now, assuming data_df is already aligned.
    
    # Scale the input data using the loaded scaler
    try:
        # Check if data_df is empty before scaling
        if data_df.empty:
            print("ERROR: Empty DataFrame provided for prediction.")
            return -1, 0.0, []

        # Ensure the columns in data_df match X_train_columns for scaling
        # The reindex in app.py should have handled this, but a last check.
        if list(data_df.columns) != list(X_train_columns):
            print(f"WARNING: Column mismatch before scaling in make_predictions. Reindexing to {len(X_train_columns)} columns.")
            data_df = data_df.reindex(columns=X_train_columns, fill_value=0)

        data_scaled = scaler.transform(data_df)
        # Convert scaled data back to DataFrame to retain column names for potential XGBoost internal checks
        data_scaled_df = pd.DataFrame(data_scaled, columns=X_train_columns, index=data_df.index)

    except Exception as e:
        print(f"ERROR: Error during scaling data in make_predictions: {e}")
        return -1, 0.0, []

    prediction = -1
    probability = 0.0

    try:
        if model_type == 'xgb':
            # XGBoost can handle DataFrames, but we've already ensured column alignment
            prediction = model.predict(data_scaled_df)[0]
            probability = model.predict_proba(data_scaled_df)[:, 1][0]
        elif model_type == 'ann':
            # Keras models usually prefer numpy arrays
            ann_probs = model.predict(data_scaled_df.values)[0][0] # Get the first element (probability of class 1)
            prediction = (ann_probs > 0.5).astype("int32")
            probability = ann_probs
        else:
            print(f"ERROR: Unknown model type: {model_type}")
    except Exception as e:
        print(f"ERROR: Prediction error for {model_type}: {e}")
        prediction = -1
        probability = 0.0 # Or np.nan

    return prediction, probability, X_train_columns # Return X_train_columns as feature_names for consistency