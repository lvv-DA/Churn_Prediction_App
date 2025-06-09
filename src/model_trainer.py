import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from tf_keras.models import Sequential
from tf_keras.layers import Dense
from tf_keras import backend as K
import tensorflow as tf

# Ensure data_loader and preprocessor are imported correctly
from data_loader import load_data
from preprocessor import preprocess_data, save_scaler

def focal_loss(gamma=2., alpha=.25):
    """Focal loss for binary classification."""
    def focal_loss_fixed(y_true, y_pred):
        eps = K.epsilon()
        y_pred = K.clip(y_pred, eps, 1. - eps)
        pt_1 = tf.where(K.equal(y_true, 1), y_pred, K.ones_like(y_pred))
        pt_0 = tf.where(K.equal(y_true, 0), y_pred, K.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

def train_and_evaluate_models(X_train_scaled, y_train, X_test_scaled, y_test, X_sm, y_sm, X_train_cols, models_dir):
    """
    Trains and evaluates multiple churn prediction models.
    Saves the trained models and training column names.
    """
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created models directory at: {models_dir}")

    # --- XGBoost Model ---
    print("\n--- Training XGBoost Model ---")
    xgb_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        random_state=42
    )
    xgb_model.fit(X_sm, y_sm) # Train on SMOTE'd data
    xgb_preds = xgb_model.predict(X_test_scaled)
    xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]
    print("XGBoost Classification Report:\n", classification_report(y_test, xgb_preds))
    print("XGBoost AUC-ROC Score:", roc_auc_score(y_test, xgb_probs))
    joblib.dump(xgb_model, os.path.join(models_dir, 'xgb_model.joblib'))
    print("XGBoost model saved.")

    # --- ANN Model with Class Weights ---
    print("\n--- Training ANN Model with Class Weights ---")
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    ann_model_cw = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    ann_model_cw.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    ann_model_cw.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0, class_weight=class_weight_dict)
    ann_cw_preds = (ann_model_cw.predict(X_test_scaled) > 0.5).astype("int32")
    ann_cw_probs = ann_model_cw.predict(X_test_scaled)
    print("ANN (Class Weights) Classification Report:\n", classification_report(y_test, ann_cw_preds))
    print("ANN (Class Weights) AUC-ROC Score:", roc_auc_score(y_test, ann_cw_probs))
    ann_model_cw.save(os.path.join(models_dir, 'ann_class_weights_model.h5'))
    print("ANN (Class Weights) model saved.")

    # --- ANN Model with SMOTE ---
    print("\n--- Training ANN Model with SMOTE ---")
    ann_model_sm = Sequential([
        Dense(128, activation='relu', input_shape=(X_sm.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    ann_model_sm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    ann_model_sm.fit(X_sm, y_sm, epochs=50, batch_size=32, verbose=0) # Train on SMOTE'd data
    ann_sm_preds = (ann_model_sm.predict(X_test_scaled) > 0.5).astype("int32")
    ann_sm_probs = ann_model_sm.predict(X_test_scaled)
    print("ANN (SMOTE) Classification Report:\n", classification_report(y_test, ann_sm_preds))
    print("ANN (SMOTE) AUC-ROC Score:", roc_auc_score(y_test, ann_sm_probs))
    ann_model_sm.save(os.path.join(models_dir, 'ann_smote_model.h5'))
    print("ANN (SMOTE) model saved.")

    # --- ANN Model with Focal Loss ---
    print("\n--- Training ANN Model with Focal Loss ---")
    ann_model_fl = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    ann_model_fl.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy'])
    ann_model_fl.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
    ann_fl_preds = (ann_model_fl.predict(X_test_scaled) > 0.5).astype("int32")
    ann_fl_probs = ann_model_fl.predict(X_test_scaled)
    print("ANN (Focal Loss) Classification Report:\n", classification_report(y_test, ann_fl_preds))
    print("ANN (Focal Loss) AUC-ROC Score:", roc_auc_score(y_test, ann_fl_probs))
    ann_model_fl.save(os.path.join(models_dir, 'ann_focal_loss_model.h5'))
    print("ANN (Focal Loss) model saved.")

    # Save training columns for consistent preprocessing during prediction
    joblib.dump(X_train_cols, os.path.join(models_dir, 'X_train_columns.pkl'))
    print(f"X_train_columns saved to {os.path.join(models_dir, 'X_train_columns.pkl')}")

if __name__ == '__main__':
    print("Starting model training process...")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # --- THIS LINE HAS BEEN MODIFIED ---
    data_path = os.path.join(project_root, 'data', 'customer_churn.csv')
    
    models_dir = os.path.join(project_root, 'models')

    df = load_data(data_path)
    
    if df is not None:
        if 'Churn' not in df.columns:
            print("Error: 'Churn' column not found in the dataset.")
            # Create a dummy 'Churn' column if not present for local testing, or exit
            # For actual training, 'Churn' must be present.
            # You might want to raise an error or exit here for a real application.
            df['Churn'] = np.random.randint(0, 2, df.shape[0]) # Dummy Churn for testing
            print("A dummy 'Churn' column has been added for demonstration purposes.")
        
        X = df.drop('Churn', axis=1)
        y = df['Churn']

        # Ensure consistent column names between training and prediction
        # Get dummified column names from the full dataset before splitting
        # This creates X_full_processed, which is then used to identify X_train_cols
        X_full_processed = pd.get_dummies(X, drop_first=True)
        # Ensure that X_train_cols captures all possible columns after one-hot encoding
        # This will be used to align columns during prediction.
        X_train_cols = X_full_processed.columns.tolist()


        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        # Preprocess training data (scales features and applies SMOTE)
        # The preprocess_data function now also returns X_train_cols derived from the dummying process.
        X_train_scaled, y_train_processed, scaler, X_sm, y_sm, _ = preprocess_data(
            pd.concat([X_train, y_train], axis=1), is_training=True
        )
        
        # Save the scaler for later use in prediction
        save_scaler(scaler, os.path.join(models_dir, 'scaler.pkl'))
        print(f"Scaler saved to {os.path.join(models_dir, 'scaler.pkl')}")

        # Preprocess testing data (only scales, does not apply SMOTE)
        X_test_scaled, _, _, _, _, _ = preprocess_data(
            pd.concat([X_test, y_test], axis=1), is_training=False, scaler=scaler, X_train_columns=X_train_cols
        )
        
        train_and_evaluate_models(X_train_scaled, y_train, X_test_scaled, y_test, X_sm, y_sm, X_train_cols, models_dir)
        print("\nModel training and evaluation complete.")
    else:
        print("Data loading failed. Skipping model training.")