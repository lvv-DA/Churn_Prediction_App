import streamlit as st
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import time
import joblib
import gc # Import garbage collection module

import google.generativeai as genai
# Import K from tf_keras.backend to use K.clear_session() and tensorflow for session management
from tf_keras import backend as K
import tensorflow as tf
import numpy as np # Ensure numpy is imported for numerical operations

# Suppress TensorFlow warnings related to GPU and oneDNN
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # This hides all GPUs from TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # This disables oneDNN optimizations as suggested by a warning

# Assuming st_copy_to_clipboard is a custom component you have installed
# If you don't have this, you might need to install it or remove this line and its usage.
try:
    from st_copy_to_clipboard import st_copy_to_clipboard
except ImportError:
    st.warning("`st_copy_to_clipboard` component not found. Copy functionality will be disabled.")
    st_copy_to_clipboard = None # Set to None if not available

# --- Company Branding ---
COMPANY_NAME = "ABC Telecom"

# --- Streamlit Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND AND ONLY ONCE) ---
st.set_page_config(
    page_title=f"Customer Churn Prediction - {COMPANY_NAME}",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.success("Application started and dependencies checked.")

# --- Configure Google Gemini API ---
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY") # Attempt to get from environment variable first
    if not gemini_api_key:
        # If running in Streamlit Community Cloud, st.secrets is the way
        gemini_api_key = st.secrets.get("GEMINI_API_KEY")
        if not gemini_api_key:
            st.sidebar.error("GEMINI_API_KEY not found in environment variables or Streamlit secrets.")
            raise ValueError("GEMINI_API_KEY is not set.")

    genai.configure(api_key=gemini_api_key)
    # Using 'gemini-1.5-flash' is generally faster and cheaper than 'gemini-pro'
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    st.sidebar.success("Gemini API configured successfully!")
except Exception as e:
    st.sidebar.warning(f"Gemini API not configured. Please set GEMINI_API_KEY in environment variables or Streamlit secrets: {e}")
    gemini_model = None

# --- Paths Configuration ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming 'app.py' is in the root directory, and 'data', 'src', 'models' are subfolders
project_root = current_script_dir

src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# Import necessary functions from src
from model_predictor import make_predictions, load_all_models


# --- PRE-LISTED OFFERS for Gemini Recommendations ---
PRE_LISTED_OFFERS = {
    "Very High Risk": """
    - **Urgent Retention Call**: Immediate outreach with a dedicated retention specialist.
    - **Aggressive Discount Offer**: Provide a 30-40% discount on the next 6-12 months' bills.
    - **Premium Service Package**: Offer an upgrade to the highest-tier plan (e.g., unlimited data, fastest internet) for a significant period at no extra cost.
    - **Personalized Problem Resolution**: Assign a high-level support agent to address all past and present issues proactively.
    """,
    "High Risk": """
    - **Proactive Retention Call**: Offer a dedicated account manager and a personalized service review.
    - **Exclusive Discount on Next Bill**: Provide a 20% discount on the next 3 months' bills.
    - **Free Service Upgrade**: Upgrade to a higher-tier plan (e.g., more data, faster internet) for 3 months at no extra cost.
    - **Loyalty Program Boost**: Double loyalty points for the next 6 months.
    """,
    "Medium Risk": """
    - **Personalized Bundle Offer**: Create a custom bundle based on their usage (e.g., more talk time, international calls pack).
    - **Device Upgrade Incentive**: Offer a significant discount on a new device with contract renewal.
    - **Value-Added Service Trial**: Free trial of a premium service (e.g., streaming, cloud storage).
    """,
    "Low Risk": """
    - **Customer Appreciation Gift**: Small token of appreciation (e.g., bonus data, movie rental credit).
    - **Referral Program Incentive**: Encourage referrals with an attractive bonus for both parties.
    - **Service Optimization Tips**: Provide advice on how to get more value from their current plan.
    """,
    "Complaints": """
    - **Dedicated Support Resolution**: Assign a senior support agent to resolve their issues promptly.
    - **Service Recovery Credit**: Offer a credit on their bill as an apology for inconvenience.
    - **Feedback Integration**: Assure them their feedback is being used to improve services.
    """
}

@st.cache_resource(hash_funcs={tf.keras.Model: lambda _: None}) # Add hash_funcs to prevent hashing issues with Keras models
def get_model_assets():
    """
    Loads all trained models and the scaler.
    Clears the TensorFlow session before loading models to ensure a clean graph,
    which is a common practice when using Keras models with st.cache_resource
    to prevent memory issues over time.
    """
    # These are crucial for preventing memory leaks with Keras in Streamlit
    K.clear_session()
    tf.compat.v1.reset_default_graph() # Reset default graph for TF2 compatibility
    gc.collect() # Explicitly run garbage collection

    try:
        loaded_assets = load_all_models()
        return loaded_assets
    except Exception as e:
        st.error(f"Error loading model assets: {e}. Please ensure models are trained and saved correctly by running `python src/model_trainer.py`.")
        st.stop() # Stop the app if models cannot be loaded

# Load models and scaler
assets = get_model_assets()

scaler = assets.get('scaler')
xgb_model = assets.get('xgb_smote') # Key should match the one used in load_all_models
ann_class_weights_model = assets.get('ann_class_weights')
ann_smote_model = assets.get('ann_smote')
ann_focal_loss_model = assets.get('ann_focal_loss')
X_train_columns = assets.get('X_train_columns')

# Check if all critical assets are loaded
if not all([scaler, xgb_model, ann_class_weights_model, ann_smote_model, ann_focal_loss_model, X_train_columns]):
    st.error("One or more models/scaler/training columns could not be loaded. Please ensure models are trained and saved correctly by running `python src/model_trainer.py`.")
    st.stop()

# Define all models for the ensemble
ENSEMBLE_MODELS = {
    'XGBoost + SMOTE': {'model': xgb_model, 'type': 'xgb'},
    'ANN + Class Weights': {'model': ann_class_weights_model, 'type': 'ann'},
    'ANN + SMOTE': {'model': ann_smote_model, 'type': 'ann'},
    'ANN + Focal Loss': {'model': ann_focal_loss_model, 'type': 'ann'}
}


# --- Data Loading for App Display and Search ---
@st.cache_data
def load_customer_identifiers_data():
    customer_data_path = os.path.join(project_root, 'data', 'customer_data_with_identifiers.csv')

    if not os.path.exists(customer_data_path):
        st.error(f"Customer identifier data file not found: {customer_data_path}. Please ensure it's in the 'data' directory.")
        return pd.DataFrame() # Return empty DataFrame on failure

    try:
        df_customers = pd.read_csv(customer_data_path)

        required_id_cols = ['CustomerID', 'CustomerName', 'PhoneNumber']

        # Check if required columns exist and log any missing ones
        missing_required_cols = [col for col in required_id_cols if col not in df_customers.columns]
        if missing_required_cols:
            st.warning(f"Missing crucial identifier columns in '{os.path.basename(customer_data_path)}': {', '.join(missing_required_cols)}. Search may be impaired.")

        # Convert CustomerID to string early to avoid issues with mixed types
        if 'CustomerID' in df_customers.columns:
            df_customers['CustomerID'] = df_customers['CustomerID'].astype(str)

        # Drop rows where any of the critical identifier columns are NaN/empty *after* ensuring they are strings
        initial_rows = df_customers.shape[0]

        for col in required_id_cols:
            if col in df_customers.columns:
                # Replace empty strings with NaN for object columns
                if pd.api.types.is_object_dtype(df_customers[col]):
                    df_customers[col] = df_customers[col].astype(str).replace(r'^\s*$', np.nan, regex=True)

        # Drop rows where any of the IDENTIFIERS are truly NaN/None/empty string
        df_customers.dropna(subset=[col for col in required_id_cols if col in df_customers.columns], inplace=True)

        rows_after_dropna = df_customers.shape[0]
        if initial_rows != rows_after_dropna:
            pass # Removed print, keeping silent for now

        if df_customers.empty:
            st.error("No valid customer records found after filtering for complete identifiers. Please check your `customer_data_with_identifiers.csv` file.")
            return pd.DataFrame() # Return empty DataFrame if no valid customers

        return df_customers

    except pd.errors.EmptyDataError:
        st.error(f"'{os.path.basename(customer_data_path)}' is empty or contains no data.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading customer identifier data from '{os.path.basename(customer_data_path)}': {e}. Search functionality will be disabled.")
        return pd.DataFrame()

all_customers_df = load_customer_identifiers_data()

# CRITICAL CHECK: If all_customers_df is still empty, stop the app or handle gracefully.
if all_customers_df.empty:
    st.error("Cannot proceed as no valid customer identifier data could be loaded. Please ensure `data/customer_data_with_identifiers.csv` is correctly formatted and contains data.")
    st.stop()


@st.cache_data
def load_customer_churn_data():
    customer_churn_path = os.path.join(project_root, 'data', 'customer_churn.csv')

    if not os.path.exists(customer_churn_path):
        st.warning(f"Customer churn data file not found: {customer_churn_path}.")
        return pd.DataFrame() # Return empty DataFrame on failure

    try:
        df_churn = pd.read_csv(customer_churn_path)

        if 'Churn' in df_churn.columns:
            df_churn['Churn'] = pd.to_numeric(df_churn['Churn'], errors='coerce')
            initial_rows = df_churn.shape[0]
            df_churn.dropna(subset=['Churn'], inplace=True) # Drop rows where Churn is NaN after conversion
            rows_after_dropna = df_churn.shape[0]
            if initial_rows != rows_after_dropna:
                pass # Removed print, keeping silent for now
        else:
            st.warning("No 'Churn' column found in `customer_churn.csv` for distribution analysis. Some features might be missing for prediction.")

        return df_churn
    except pd.errors.EmptyDataError:
        st.error(f"'{os.path.basename(customer_churn_path)}' is empty or contains no data.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading customer churn data from '{os.path.basename(customer_churn_path)}': {e}.")
        return pd.DataFrame()

customers_churn_df = load_customer_churn_data()

# CRITICAL CHECK: Ensure customers_churn_df is not empty for dynamic form generation
if customers_churn_df.empty:
    st.error("Cannot proceed as `customer_churn.csv` could not be loaded or is empty. This data is needed to define features for manual entry.")
    st.stop()


# --- Session State Initialization ---
# Initialize session state keys for predictability
if 'selected_customer_id' not in st.session_state:
    st.session_state['selected_customer_id'] = None
if 'selected_customer_data' not in st.session_state:
    st.session_state['selected_customer_data'] = None
if 'search_query' not in st.session_state:
    st.session_state['search_query'] = ""
if 'search_results_display_df' not in st.session_state:
    st.session_state['search_results_display_df'] = pd.DataFrame()
if 'show_prediction_results' not in st.session_state:
    st.session_state['show_prediction_results'] = False
if 'selected_customer_from_search_option' not in st.session_state:
    st.session_state['selected_customer_from_search_option'] = None
if 'manual_entry_mode' not in st.session_state:
    st.session_state['manual_entry_mode'] = False


# --- Callback for customer selection ---
def on_customer_select():
    selected_option_label = st.session_state.selected_customer_from_search_option
    if selected_option_label:
        selected_customer_id_from_label = selected_option_label.split(' | ')[0]

        # Fetch the row from the *original* all_customers_df, not a session state copy
        filtered_df = all_customers_df[
            all_customers_df['CustomerID'].astype(str) == selected_customer_id_from_label
        ]

        if not filtered_df.empty:
            selected_customer_row = filtered_df.iloc[0]
            st.session_state['selected_customer_id'] = selected_customer_row['CustomerID']
            st.session_state['selected_customer_data'] = selected_customer_row.to_dict()
            st.session_state['show_prediction_results'] = False # Reset prediction view on new selection
            st.session_state['manual_entry_mode'] = False # Exit manual entry mode if a customer is selected

            if any(pd.isna(v) for v in st.session_state['selected_customer_data'].values()):
                st.warning("Some feature values are missing (NaN) for this customer after selection. Please review the loaded data.")

        else:
            st.warning(f"Selected customer (ID: {selected_customer_id_from_label}) not found. Please re-search or select again.")
            st.session_state['selected_customer_id'] = None
            st.session_state['selected_customer_data'] = None
            st.session_state['show_prediction_results'] = False
            st.session_state['manual_entry_mode'] = False
    else:
        st.session_state['selected_customer_id'] = None
        st.session_state['selected_customer_data'] = None
        st.session_state['show_prediction_results'] = False
        st.session_state['manual_entry_mode'] = False


# --- Function to generate recommendations using Gemini (kept in app.py for streaming behavior) ---
def get_gemini_recommendations(model, churn_risk_level, customer_details, offers, company_name, customer_complains=0):
    if not model:
        yield "Gemini model is not initialized. Cannot generate recommendations."
        return # Use return instead of break for generator

    relevant_offers_list = []
    if churn_risk_level in offers:
        relevant_offers_list.append(offers[churn_risk_level])
    if customer_complains > 0 and "Complaints" in offers:
        relevant_offers_list.append(offers["Complaints"])

    relevant_offers_str = "\n\n".join(relevant_offers_list) if relevant_offers_list else "No specific pre-listed offers to suggest at this time."


    prompt = f"""
    You are an AI assistant for a telecom company named {company_name}.
    A customer is at '{churn_risk_level}' risk of churning.
    Here are the customer's details:
    {customer_details}

    Based on these details, and the following pre-listed offers, suggest personalized recommendations to retain this customer.
    Focus on suggesting 1-3 highly relevant offers. If no offer is directly relevant, suggest a general retention strategy.
    The tone should be empathetic and persuasive, suitable for an internal report.

    Pre-listed offers tailored to risk level and complaints:
    {relevant_offers_str}

    Generate the recommendations in a concise markdown format, suitable for an internal report.
    Start with "AI-Powered Retention Strategy:" followed by a bulleted list of recommendations.
    """
    try:
        response_stream = model.generate_content(prompt, stream=True)
        full_response = ""
        for chunk in response_stream:
            if chunk.text: # Ensure chunk.text is not None
                full_response += chunk.text
                time.sleep(0.01) # Small sleep for better UX for streaming
                yield full_response

        # Ensure the final full response is yielded after all chunks are processed
        if full_response:
            yield full_response

    except Exception as e:
        yield f"An error occurred while generating recommendations: {e}"


# --- Streamlit UI ---
st.title(f"Customer Churn Prediction Dashboard - {COMPANY_NAME}")

st.markdown("""
    This dashboard predicts customer churn based on various service usage patterns and demographic information.
    It utilizes multiple machine learning models (XGBoost, ANN with Class Weights, ANN with SMOTE, ANN with Focal Loss)
    and provides AI-powered retention strategies using Google Gemini.
""")

# Custom CSS for transient flash alerts (optional, but good for UX)
st.markdown("""
<style>
.stAlert {
    position: fixed;
    top: 10px;
    right: 10px;
    width: auto;
    max-width: 300px;
    z-index: 9999;
    animation: fadeout 5s forwards;
}

@keyframes fadeout {
    0% { opacity: 1; }
    80% { opacity: 1; }
    100% { opacity: 0; display: none; }
}
</style>
""", unsafe_allow_html=True)

# Add a simple mechanism for flash messages
def show_flash_message(message, type="info"):
    if type == "success":
        st.success(message, icon="✅")
    elif type == "error":
        st.error(message, icon="❌")
    elif type == "warning":
        st.warning(message, icon="⚠️")
    else:
        st.info(message, icon="ℹ️")


# --- Tabbed Interface ---
tab1, tab2 = st.tabs(["Customer Churn Prediction", "Data Overview & Training"])

with tab1:
    st.header("1. Customer Search & Prediction")

    # Layout for search and customer details
    col_search, col_details = st.columns([1, 2])

    with col_search:
        search_query_input = st.text_input("Search Customer by ID, Name, or Phone:",
                                            value=st.session_state['search_query'],
                                            key="search_query_input_text")

        if st.button("Search", key="search_button"):
            st.session_state['search_query'] = search_query_input.strip() # Strip whitespace
            if st.session_state['search_query']:
                search_lower = st.session_state['search_query'].lower()

                searchable_cols = []
                if 'CustomerID' in all_customers_df.columns: searchable_cols.append('CustomerID')
                if 'PhoneNumber' in all_customers_df.columns: searchable_cols.append('PhoneNumber')
                if 'CustomerName' in all_customers_df.columns: searchable_cols.append('CustomerName')

                if searchable_cols:
                    conditions = []
                    for col in searchable_cols:
                        col_series = all_customers_df[col].astype(str)
                        conditions.append(col_series.str.lower().str.contains(search_lower, na=False))

                    if conditions:
                        combined_condition = conditions[0]
                        for cond in conditions[1:]:
                            combined_condition = combined_condition | cond
                        results = all_customers_df[combined_condition].copy()
                    else:
                        results = pd.DataFrame()

                    st.session_state['search_results_display_df'] = results

                else:
                    st.warning("No searchable identifier columns found in the customer data.")
                    st.session_state['search_results_display_df'] = pd.DataFrame()

                st.session_state['selected_customer_id'] = None # Reset selected customer on new search
                st.session_state['selected_customer_data'] = None
                st.session_state['show_prediction_results'] = False
                st.session_state['manual_entry_mode'] = False
            else:
                st.session_state['search_results_display_df'] = pd.DataFrame() # Clear results if search query is empty or df is None
                st.info("Please enter a search query.")
                st.session_state['selected_customer_id'] = None
                st.session_state['selected_customer_data'] = None
                st.session_state['show_prediction_results'] = False
                st.session_state['manual_entry_mode'] = False

        if not st.session_state['search_results_display_df'].empty:
            st.subheader("Search Results:")
            st.write(f"{len(st.session_state['search_results_display_df'])} matches found.")
            display_label = st.session_state['search_results_display_df'].apply(
                lambda row: f"{row.get('CustomerID', 'N/A')} | {row.get('CustomerName', 'N/A')} | {row.get('PhoneNumber', 'N/A')}",
                axis=1
            ).tolist()

            # Determine default index for selectbox
            default_index = 0
            if st.session_state['selected_customer_id'] and not st.session_state['manual_entry_mode']:
                try:
                    current_selected_label = next(
                        label for label in display_label
                        if label.startswith(str(st.session_state['selected_customer_id']) + ' | ')
                    )
                    default_index = display_label.index(current_selected_label)
                except StopIteration:
                    default_index = 0

            st.selectbox(
                "Select a Customer:",
                options=display_label,
                key="selected_customer_from_search_option",
                index=default_index,
                on_change=on_customer_select
            )
        elif st.session_state['search_query'] and not all_customers_df.empty: # Only show this if search was attempted and no results
            st.info("No customers found matching your search query.")

        # Add a button to switch to manual entry
        if st.button("Manual Entry", key="manual_entry_toggle_button"):
            st.session_state['manual_entry_mode'] = True
            st.session_state['selected_customer_id'] = "Manual Entry"
            # Initialize with default/placeholder values for manual entry
            # Get default values from the churn data's first row or mode for numerical/categorical
            # Ensure 'Churn' column is excluded if it exists
            default_numeric = {
                col: customers_churn_df[col].mean()
                for col in customers_churn_df.select_dtypes(include=np.number).columns if col != 'Churn'
            }
            default_categorical = {
                col: customers_churn_df[col].mode()[0]
                for col in customers_churn_df.select_dtypes(exclude=np.number).columns if col != 'Churn'
            }

            st.session_state['selected_customer_data'] = {
                'CustomerID': 'Manual Entry',
                'CustomerName': 'N/A',
                'PhoneNumber': 'N/A',
                **default_numeric,
                **default_categorical
            }
            st.session_state['show_prediction_results'] = False
            # Clear search query and results when entering manual mode
            st.session_state['search_query'] = ""
            st.session_state['search_results_display_df'] = pd.DataFrame()


    with col_details:
        st.subheader("Customer Details for Prediction")
        if st.session_state['selected_customer_data'] is not None or st.session_state['manual_entry_mode']:
            customer_data = st.session_state['selected_customer_data']

            if any(pd.isna(v) for v in customer_data.values()):
                st.warning("Some feature values are missing (NaN) for this customer. Please review the loaded data or manually adjust inputs.")

            # st.json({k: (v if pd.notna(v) else 'NaN/Missing') for k,v in customer_data.items()}) # Show NaN explicitly

            # Create an input form for all features, using session state for consistency
            with st.form("customer_details_form"):
                form_columns = st.columns(2)
                form_index = 0

                # Filter out 'Churn' column if it exists, as it's the target
                all_features_from_churn_df = [col for col in customers_churn_df.columns if col != 'Churn']

                # IMPORTANT: Explicitly define which columns should be treated as categorical for one-hot encoding
                # These are usually columns that are integers but represent categories (e.g., AgeGroup 1, 2, 3)
                explicit_categorical_features = ['AgeGroup', 'TariffPlan', 'Status']

                # Filter out explicitly categorical features from numerical
                numerical_features = customers_churn_df.select_dtypes(include=np.number).columns.drop(
                    ['Churn'] + [col for col in explicit_categorical_features if col in customers_churn_df.columns],
                    errors='ignore'
                ).tolist()

                # Add any other categorical features that are strings/objects
                inferred_categorical_features = customers_churn_df.select_dtypes(exclude=np.number).columns.drop(['Churn'], errors='ignore').tolist()

                # Combine and remove duplicates while maintaining order (explicit first, then inferred)
                categorical_features = list(set(explicit_categorical_features + inferred_categorical_features))
                categorical_features.sort() # Sort for consistent display

                # Sort numerical features
                numerical_features.sort()


                # Populate form with current customer data or defaults
                for feature in numerical_features:
                    with form_columns[form_index % 2]:
                        # Ensure value is float and handle potential NaNs from customer_data
                        current_value = customer_data.get(feature)
                        if pd.isna(current_value):
                            current_value = customers_churn_df[feature].mean() if customers_churn_df is not None and feature in customers_churn_df.columns else 0.0

                        st.session_state[f'input_{feature}'] = st.number_input(feature, value=float(current_value), key=f"input_{feature}_num")
                    form_index += 1

                for feature in categorical_features:
                    with form_columns[form_index % 2]:
                        unique_values = customers_churn_df[feature].unique()
                        options = sorted([str(x) for x in unique_values if pd.notna(x)])

                        if not options: # Fallback if no options are found (e.g., column is all NaN or not in data)
                            st.warning(f"No valid options found for categorical feature: {feature}. Please check your data.")
                            st.session_state[f'input_{feature}'] = None # Cannot set a selectbox without options
                            continue

                        current_value = customer_data.get(feature)
                        # Ensure current_value is also a string for comparison with options
                        current_value_str = str(current_value) if pd.notna(current_value) else options[0] # Default to first option if NaN/None

                        # Handle case where current_value from customer_data might be NaN or not in options
                        if current_value_str not in options:
                            # Try to get the mode from original df, convert to string
                            default_option = str(customers_churn_df[feature].mode()[0]) if customers_churn_df is not None and feature in customers_churn_df.columns and not customers_churn_df[feature].mode().empty else options[0]
                            current_value_str = default_option

                        st.session_state[f'input_{feature}'] = st.selectbox(feature, options=options, index=options.index(current_value_str) if current_value_str in options else 0, key=f"input_{feature}_cat")
                    form_index += 1

                predict_button = st.form_submit_button("Predict Churn for this Customer")

                if predict_button:
                    # Collect all input values into a dictionary
                    input_data = {
                        feature: st.session_state[f'input_{feature}']
                        for feature in numerical_features + categorical_features
                    }

                    # Convert to DataFrame
                    customer_input_df = pd.DataFrame([input_data])

                    # This block must EXACTLY mirror the preprocessing in preprocessor.py
                    # which first converts to category then calls get_dummies without explicit columns,
                    # relying on pandas to identify category dtypes and apply drop_first=True.

                    # Ensure these columns are of object/category dtype before one-hot encoding
                    # Otherwise, pd.get_dummies won't process them if they are numerical.
                    # This is the same logic as in preprocessor.py
                    for col in explicit_categorical_features:
                        if col in customer_input_df.columns:
                            customer_input_df[col] = customer_input_df[col].astype(str).astype('category')

                    # Apply one-hot encoding
                    # This should now match exactly what preprocessor.py does:
                    # It infers categorical columns from dtypes (which we just set)
                    # and uses drop_first=True as specified in preprocessor.py.
                    customer_processed_df = pd.get_dummies(customer_input_df, drop_first=True)

                    # Align columns with X_train_columns, filling missing with 0
                    # This step is critical to ensure the input to the model has the correct columns and order.
                    # It will add any missing columns from X_train_columns and fill them with 0.
                    # It will also drop any extra columns that are not in X_train_columns.
                    final_customer_features_df = customer_processed_df.reindex(columns=X_train_columns, fill_value=0)

                    # --- IMPORTANT: Ensure final_customer_features_df is not empty or malformed here ---
                    if final_customer_features_df.empty:
                        st.error("Error: Prepared features DataFrame is empty. Cannot make predictions.")
                        st.stop() # Or handle more gracefully
                    if final_customer_features_df.shape[1] != len(X_train_columns):
                        st.error(f"Error: Number of features ({final_customer_features_df.shape[1]}) does not match expected ({len(X_train_columns)}).")
                        st.stop() # Or handle more gracefully

                    predictions_info = []

                    with st.spinner("Predicting churn and generating recommendations..."):
                        for model_name, model_info in ENSEMBLE_MODELS.items():
                            model = model_info['model']
                            model_type = model_info['type']
                            try:
                                # CALLING make_predictions with correct arguments
                                # make_predictions takes: model, data_df, scaler, X_train_columns, model_type
                                pred, prob, _ = make_predictions(model, final_customer_features_df, scaler, X_train_columns, model_type=model_type)
                                predictions_info.append({'Model': model_name, 'Prediction': pred, 'Probability': prob})
                            except Exception as e:
                                st.error(f"Error predicting with {model_name}: {e}")
                                predictions_info.append({'Model': model_name, 'Prediction': -1, 'Probability': 'Error'}) # Indicate error

                        # Create prediction_df for internal use and plotting
                        prediction_df = pd.DataFrame(predictions_info)
                        prediction_df['Prediction Label'] = prediction_df['Prediction'].apply(lambda x: "Churn" if x == 1 else ("No Churn" if x == 0 else "Error"))

                        # Store the original numerical probability for plotting before formatting as string for display
                        prediction_df['Numerical Probability'] = pd.to_numeric(
                            prediction_df['Probability'], errors='coerce'
                        ) # Keep as float 0-1 for plotting


                        # --- PLOTLY GRAPH FOR INDIVIDUAL MODEL PREDICTIONS ---
                        st.subheader("Individual Model Churn Probabilities")

                        fig = px.bar(
                            prediction_df, # Use the DataFrame directly
                            x='Model',
                            y='Numerical Probability', # Use the numerical column for the y-axis
                            color='Prediction Label', # Color bars based on the prediction outcome
                            color_discrete_map={'Churn': '#D62728', 'No Churn': '#1F77B4', 'Error': '#AAAAAA'}, # Define colors
                            title='Individual Model Churn Probabilities',
                            labels={
                                'Numerical Probability': 'Churn Probability (%)',
                                'Model': 'Machine Learning Model'
                            },
                            text_auto=True, # Automatically display values on top of bars
                            height=450 # Adjust height as needed
                        )

                        # Further customize the text display on bars and Y-axis format
                        fig.update_traces(texttemplate='%{y:.2%}', textposition='outside') # Format text on bars as percentage
                        fig.update_layout(yaxis_tickformat=".0%") # Format Y-axis labels as percentage

                        st.plotly_chart(fig, use_container_width=True)
                        # --- END OF PLOTLY CODE BLOCK ---


                        # Ensemble Voting (consider only successful predictions for voting)
                        successful_predictions = [p['Prediction'] for p in predictions_info if p['Prediction'] in [0, 1]]
                        churn_votes = sum(successful_predictions)
                        no_churn_votes = len(successful_predictions) - churn_votes

                        ensemble_prediction_label = "No Churn"
                        churn_risk_level = "Low Risk" # Default
                        if churn_votes > 0: # If any model predicts churn
                            if churn_votes == len(ENSEMBLE_MODELS): # All models predict churn
                                ensemble_prediction_label = "Churn"
                                churn_risk_level = "Very High Risk"
                            elif churn_votes >= (len(ENSEMBLE_MODELS) / 2): # Majority vote for churn
                                ensemble_prediction_label = "Churn"
                                churn_risk_level = "High Risk"
                            else: # Minority vote for churn, but still significant
                                ensemble_prediction_label = "Potentially Churn" # Changed to make it clearer
                                churn_risk_level = "Medium Risk"

                        st.markdown(f"### Ensemble Prediction: **{ensemble_prediction_label}**")
                        st.info(f"**Churn Risk Level:** {churn_risk_level}")

                        # Visualize ensemble votes
                        if successful_predictions: # Only plot if there are valid predictions
                            vote_data = pd.DataFrame({
                                'Vote': ['Models Predicting Churn', 'Models Predicting No Churn'],
                                'Count': [churn_votes, no_churn_votes]
                            })
                            fig_votes = px.bar(vote_data, x='Vote', y='Count',
                                               title='Ensemble Model Votes for Churn',
                                               labels={'Count': 'Number of Models'},
                                               color='Vote',
                                               color_discrete_map={'Models Predicting Churn': '#D62728', 'Models Predicting No Churn': '#1F77B4'},
                                               range_y=[0, len(ENSEMBLE_MODELS)],
                                               text_auto=True,
                                               height=400
                                               )
                            fig_votes.update_layout(xaxis_title="Prediction Outcome", yaxis_title="Number of Models")
                            st.plotly_chart(fig_votes, use_container_width=True)
                        else:
                            st.warning("No successful model predictions to visualize ensemble votes.")


                        st.markdown("---")
                        st.subheader("AI-Powered Recommendations from Google Gemini")

                        # Pass only relevant customer details to Gemini
                        customer_details_for_llm = {k: v for k, v in input_data.items()} # Use input_data directly as it has the raw customer features
                        customer_details_for_llm_str = "\n".join([f"{k}: {v}" for k, v in customer_details_for_llm.items()])

                        if gemini_model:
                            recommendation_placeholder = st.empty()
                            full_recommendation_text = ""
                            for text_chunk in get_gemini_recommendations(
                                gemini_model,
                                churn_risk_level,
                                customer_details_for_llm_str,
                                PRE_LISTED_OFFERS,
                                COMPANY_NAME,
                                customer_complains=input_data.get('Complains', 0)
                            ):
                                full_recommendation_text = text_chunk
                                recommendation_placeholder.markdown(full_recommendation_text)

                            if full_recommendation_text and st_copy_to_clipboard:
                                st_copy_to_clipboard(full_recommendation_text)
                            elif not st_copy_to_clipboard:
                                st.info("Copy to clipboard functionality is disabled. Install `st-copy-to-clipboard` for this feature.")
                        else:
                            st.info("Gemini recommendations are disabled because the API is not configured.")

                    # Attempt to force garbage collection after prediction
                    gc.collect()

                    st.session_state['show_prediction_results'] = True # Set flag to show results
        else:
            st.info("Please search and select a customer, or use 'Manual Entry' to input details.")


with tab2:
    st.header("Data Overview and Model Training")
    st.subheader("Customer Data Overview (from `customer_data_with_identifiers.csv`)")
    if not all_customers_df.empty: # Check if DataFrame is not empty
        st.write("Displaying sample of data used for search and general overview:")
        st.dataframe(all_customers_df.head())
    else:
        st.info("No customer identifier data loaded for search overview. Please check `data/customer_data_with_identifiers.csv`.")

    st.subheader("Data Statistics (from `customer_churn.csv`)")
    if not customers_churn_df.empty: # Check if DataFrame is not empty
        st.write(customers_churn_df.describe())

        st.subheader("Churn Distribution (from `customer_churn.csv`)")
        if 'Churn' in customers_churn_df.columns:
            churn_counts = customers_churn_df['Churn'].value_counts().rename(index={0: 'No Churn', 1: 'Churn'})

            # --- MODIFIED: Use a bar chart for churn distribution ---
            fig_churn = px.bar(
                x=churn_counts.index,
                y=churn_counts.values,
                title='Distribution of Churn Status',
                labels={'x': 'Churn Status', 'y': 'Number of Customers'},
                color=churn_counts.index, # Color bars by status
                color_discrete_map={'No Churn': '#1F77B4', 'Churn': '#D62728'}, # Red for Churn, Blue for No Churn
                text_auto=True # Display values on bars
            )
            fig_churn.update_layout(xaxis_title="Churn Status", yaxis_title="Number of Customers")
            st.plotly_chart(fig_churn, use_container_width=True)
            # --- END MODIFIED ---
        else:
            st.info("No 'Churn' column found in `customer_churn.csv` for distribution analysis.")
    else:
        st.info("No customer churn data loaded for analysis. Please check `data/customer_churn.csv`.")


    st.subheader("Model Training")
    st.markdown("""
        To train or re-train the models, run the `model_trainer.py` script directly.
        Ensure your `customer_churn.csv` (the larger dataset) is in the `data/` directory.
        ```bash
        python src/model_trainer.py
        ```
        This will train XGBoost and ANN models, save them to the `models/` directory,
        along with the `scaler.pkl` and `X_train_columns.pkl`.
    """)

st.markdown("---")
st.markdown(f"Developed by Vinu & UK for {COMPANY_NAME} | Version 1.0")