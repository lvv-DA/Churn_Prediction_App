import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

def preprocess_data(df, is_training, scaler=None, X_train_columns=None):
    """
    Preprocesses the data by:
    1. Separating features (X) and target (y).
    2. Handling one-hot encoding for categorical features.
    3. Scaling numerical features.
    4. Applying SMOTE for oversampling if in training mode.

    Args:
        df (pd.DataFrame): The input DataFrame containing features and 'Churn' target.
        is_training (bool): True if preprocessing training data, False for testing/prediction.
        scaler (StandardScaler, optional): Fitted scaler for consistency in testing/prediction.
        X_train_columns (list, optional): List of column names from the training data after preprocessing.
                                          Used to align columns in testing/prediction.

    Returns:
        tuple: (X_scaled, y, scaler, X_smote, y_smote, X_processed_columns)
            X_scaled (pd.DataFrame): Scaled features.
            y (pd.Series): Target variable.
            scaler (StandardScaler): The fitted scaler (if is_training=True) or the provided scaler.
            X_smote (pd.DataFrame): Features after SMOTE (if is_training=True, else None).
            y_smote (pd.Series): Target after SMOTE (if is_training=True, else None).
            X_processed_columns (list): List of column names after one-hot encoding.
    """
    if 'Churn' not in df.columns:
        raise ValueError("DataFrame must contain a 'Churn' column.")

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Identify categorical columns (numeric columns that represent categories, e.g., 'AgeGroup', 'TariffPlan', 'Status')
    # Assuming these are the columns you want to one-hot encode
    # These should be consistent with what you've identified in app.py for manual entry.
    categorical_features_to_encode = ['AgeGroup', 'TariffPlan', 'Status'] 

    # Convert specified columns to 'category' dtype if they exist in X
    for col in categorical_features_to_encode:
        if col in X.columns:
            X[col] = X[col].astype(str).astype('category') # Convert to string first to handle numeric categories properly

    # One-hot encode categorical features. drop_first=True to avoid multicollinearity.
    X_processed = pd.get_dummies(X, drop_first=True)
    
    # --- CRITICAL FIX: Align columns based on X_train_columns ---
    # This ensures that the columns match exactly between training and testing data,
    # and fills missing columns with 0 (e.g., if a category doesn't appear in test set).
    if X_train_columns is not None:
        # Reindex to ensure all columns present in training data are here, fill missing with 0
        X_processed = X_processed.reindex(columns=X_train_columns, fill_value=0)
    elif is_training: # If it's training data AND X_train_columns is None, then this IS the source
        X_train_columns = X_processed.columns.tolist() # Capture the columns for future use

    # Feature Scaling
    if is_training:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_processed)
        X_scaled = pd.DataFrame(X_scaled, columns=X_processed.columns, index=X_processed.index)
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided for non-training data preprocessing.")
        if X_train_columns is None:
            raise ValueError("X_train_columns must be provided for non-training data preprocessing.")
        
        # Ensure the order of columns matches the one used for fitting the scaler
        # This is implicitly handled by reindex above, but good to be explicit for clarity
        X_scaled = scaler.transform(X_processed)
        X_scaled = pd.DataFrame(X_scaled, columns=X_processed.columns, index=X_processed.index)


    # Apply SMOTE only on training data (after scaling)
    X_smote, y_smote = None, None
    if is_training:
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X_scaled, y)
        # Convert back to DataFrame to retain column names
        X_smote = pd.DataFrame(X_smote, columns=X_scaled.columns, index=y_smote.index)

    return X_scaled, y, scaler, X_smote, y_smote, X_train_columns

def save_scaler(scaler_obj, path):
    """Saves the scaler object to a file."""
    joblib.dump(scaler_obj, path)

# This function is not used in preprocess_data but is kept for potential future use or external calls
def load_scaler(path):
    """Loads a scaler object from a file."""
    if os.path.exists(path):
        return joblib.load(path)
    return None