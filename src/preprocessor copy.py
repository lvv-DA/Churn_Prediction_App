import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os
# IMPORTANT: There should be NO 'from preprocessor import ...' line here!

def preprocess_data(df, target_column='Churn', is_training=True, scaler=None, X_train_columns=None):
    """
    Preprocesses the data by separating features and target, encoding categoricals,
    and optionally scaling features and applying SMOTE.
    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        is_training (bool): If True, fits and transforms scaler/SMOTE. If False,
                            only transforms using provided scaler.
        scaler (StandardScaler, optional): Fitted scaler object for transforming new data.
        X_train_columns (list, optional): List of columns from training data to ensure
                                           consistent columns for new data.
    Returns:
        tuple: (X, y, scaler, smote_X, smote_y) if is_training=True,
               (X_processed, scaler) if is_training=False.
               smote_X and smote_y are None if is_training=False.
    """
    X = df.drop(columns=[target_column], errors='ignore')
    y = df[target_column] if target_column in df.columns else None

    # Encode categoricals (assuming get_dummies handles non-existent columns gracefully for prediction)
    X = pd.get_dummies(X, drop_first=True)

    if is_training:
        # Fit and transform scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_sm, y_sm = smote.fit_resample(X_scaled, y)

        X_cols = X.columns.tolist() # Capture column names after one-hot encoding
        return X_scaled, y, scaler, X_sm, y_sm, X_cols
    else:
        # For prediction, ensure columns match training data
        if X_train_columns is not None:
            # Add missing columns with 0 and remove extra columns
            missing_cols = set(X_train_columns) - set(X.columns)
            for c in missing_cols:
                X[c] = 0
            X = X[X_train_columns] # Ensure column order is the same

        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X # If no scaler provided, return unscaled

        return X_scaled, y, scaler, None, None, X_train_columns # Return X_train_columns as last item

def save_scaler(scaler, path):
    """Saves the trained scaler."""
    joblib.dump(scaler, path)
    print(f"Scaler saved to {path}")

# This block is for local testing of preprocessor.py only
if __name__ == '__main__':
    print("--- Running preprocessor.py for local testing ---")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Ensure data_loader is imported correctly for the local testing block
    from data_loader import load_data 

    # --- THIS LINE HAS BEEN MODIFIED ---
    data_path = os.path.join(project_root, 'data', 'customer_churn.csv')

    models_dir = os.path.join(project_root, 'models') # Define models_dir for scaler saving

    df = load_data(data_path) # Use the correct data_path

    if df is not None:
        import numpy as np # Import numpy for dummy data creation if needed
        if 'Churn' not in df.columns:
            print("Warning: 'Churn' column not found in the dataset for preprocessor testing. Adding dummy churn.")
            df['Churn'] = np.random.randint(0, 2, df.shape[0]) # Add dummy for testing

        X_scaled, y, scaler, X_sm, y_sm, X_cols = preprocess_data(df, is_training=True)
        print("\n--- Preprocessing for Training (Preprocessor Test) ---")
        print("X_scaled shape:", X_scaled.shape)
        print("y shape:", y.shape)
        print("X_sm shape:", X_sm.shape)
        print("y_sm shape:", y_sm.shape)

        # Save the scaler for later use
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        if not os.path.exists(models_dir): # Check if models_dir exists
            os.makedirs(models_dir)
        save_scaler(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")

        # Example usage for prediction (using a new dummy df)
        print("\n--- Preprocessing for Prediction (Preprocessor Test) ---")
        df_new = pd.DataFrame({
            'CallFailure': [5], 'Complains': [0], 'SubscriptionLength': [30],
            'ChargeAmount': [1], 'SecondsUse': [1000], 'FrequencyUse': [20],
            'FrequencySMS': [10], 'DistinctCalls': [5], 'AgeGroup': [2],
            'TariffPlan': [1], 'Status': [1], 'Age': [25],
            'CustomerValue': [500.0]
        })
        # Load the saved scaler and X_train_columns for prediction preprocessing
        loaded_scaler = joblib.load(scaler_path)

        # Assume X_train_columns.pkl exists and is loaded from models_dir for consistency
        # For this test, let's derive it from the dummy data if it doesn't exist.
        # In a real scenario, this would come from the training phase.
        X_dummy = df.drop(columns=['Churn'], errors='ignore')
        dummy_X_train_columns = pd.get_dummies(X_dummy, drop_first=True).columns.tolist()

        X_new_scaled, _, _, _, _, _ = preprocess_data(
            df_new, is_training=False, scaler=loaded_scaler, X_train_columns=dummy_X_train_columns
        )
        print("X_new_scaled shape:", X_new_scaled.shape)
        print("Preprocessor testing complete.")
    else:
        print("Data loading failed in preprocessor.py test block. Skipping preprocessing tests.")