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

# Suppress TensorFlow warnings related to GPU and oneDNN
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # This hides all GPUs from TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # This disables oneDNN optimizations as suggested by a warning

# Assuming st_copy_to_clipboard is a custom component you have installed
# If you don't have this, you might need to install it or remove this line and its usage.
try:
    from st_copy_to_clipboard import st_copy_to_clipboard
except ImportError:
    st_copy_to_clipboard = None
    st.warning("`st_copy_to_clipboard` component not found. Copy functionality will be disabled.")


# --- Company Branding (DEFINED BEFORE set_page_config) ---
COMPANY_NAME = "ABC Telecom"

# --- Streamlit Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title=f"Customer Churn Prediction - {COMPANY_NAME}",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.success("Application started and dependencies checked.")
st.write("Ready to predict customer churn!")

# --- Configure Google Gemini API ---
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY") # Attempt to get from environment variable first
    if not gemini_api_key:
        gemini_api_key = st.secrets["GEMINI_API_KEY"] # Fallback to Streamlit secrets

    genai.configure(api_key=gemini_api_key)
    # Using 'gemini-1.5-flash' is generally faster and cheaper than 'gemini-pro'
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    st.sidebar.success("Gemini API configured successfully!")
except Exception as e:
    st.sidebar.warning(f"Gemini API not configured. Please set GEMINI_API_KEY in environment variables or Streamlit secrets: {e}")
    gemini_model = None

# --- Paths Configuration ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_script_dir

src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# Ensure the src directory exists and contains the required modules
if not os.path.exists(os.path.join(src_path, 'data_loader.py')):
    st.error("Could not find 'data_loader.py' in the 'src' directory. Please ensure the file exists.")
    st.stop()

# Import necessary functions from src
from data_loader import load_data
from preprocessor import preprocess_data
from model_predictor import predict_churn, load_all_models # Changed from load_all_models to load_model_assets

# --- PRE-LISTED OFFERS for Gemini Recommendations ---
PRE_LISTED_OFFERS = {
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

# --- Model Loading with Caching ---
@st.cache_resource(hash_funcs={tf.keras.Model: lambda _: None}) # Add hash_funcs to prevent hashing issues with Keras models
def get_model_assets():
    """
    Loads all trained models and the scaler.
    Clears the TensorFlow session before loading models to ensure a clean graph,
    which is a common practice when using Keras models with st.cache_resource
    to prevent memory issues over time.
    """
    # Clear TF session before loading models to ensure a clean graph
    # This is critical for preventing memory leaks with Keras/TensorFlow in Streamlit.
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    gc.collect() # Explicitly run garbage collection

    try:
        loaded_assets = load_all_models()
        return loaded_assets
    except Exception as e:
        st.error(f"Error loading model assets: {e}. Please ensure models are trained and saved correctly.")
        st.stop() # Stop the app if models cannot be loaded

# Load models and scaler
assets = get_model_assets()
scaler = assets.get('scaler')
xgb_model = assets.get('xgb_smote_model')
# Re-enabling ANN models. Ensure these models are actually available and correctly loaded
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
    try:
        df_customers = pd.read_csv(customer_data_path)
        if 'CustomerID' in df_customers.columns:
            df_customers['CustomerID'] = df_customers['CustomerID'].astype(str)
        df_customers.dropna(subset=['CustomerID', 'CustomerName', 'PhoneNumber'], inplace=True)
        return df_customers
    except FileNotFoundError:
        st.warning(f"Customer identifier data not found at {customer_data_path}. Search functionality will be disabled.")
        return None
    except Exception as e:
        st.error(f"Error loading customer identifier data: {e}. Search functionality will be disabled.")
        return None

all_customers_df = load_customer_identifiers_data()

if all_customers_df is None:
    st.stop() # Stop the app if data for search/display cannot be loaded

# --- Session State Initialization ---
# Initialize session state keys for predictability
if 'selected_customer_id' not in st.session_state:
    st.session_state['selected_customer_id'] = None
if 'selected_customer_data' not in st.session_state:
    st.session_state['selected_customer_data'] = None
if 'search_query' not in st.session_state:
    st.session_state['search_query'] = ""
# Do not store full DataFrame in session state; store only identifiers if needed.
# 'search_results_data' will be a temporary DataFrame for display.
if 'search_results_display_df' not in st.session_state:
    st.session_state['search_results_display_df'] = pd.DataFrame()
if 'show_prediction_results' not in st.session_state:
    st.session_state['show_prediction_results'] = False
if 'selected_customer_from_search_option' not in st.session_state: # Renamed key for clarity
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
        else:
            st.warning("Selected customer not found. Please re-search or select again.")
            st.session_state['selected_customer_id'] = None
            st.session_state['selected_customer_data'] = None
            st.session_state['show_prediction_results'] = False
            st.session_state['manual_entry_mode'] = False
    else:
        st.session_state['selected_customer_id'] = None
        st.session_state['selected_customer_data'] = None
        st.session_state['show_prediction_results'] = False
        st.session_state['manual_entry_mode'] = False # Reset manual entry mode


# --- Function to generate recommendations using Gemini ---
def get_gemini_recommendations(model, churn_risk_level, customer_details, offers, company_name, customer_complains=0):
    if not model:
        return "Gemini model is not initialized. Cannot generate recommendations."

    relevant_offers = offers.get(churn_risk_level, "")
    if customer_complains > 0:
        relevant_offers += "\n\n" + offers.get("Complaints", "")

    prompt = f"""
    You are an AI assistant for a telecom company named {company_name}.
    A customer is at '{churn_risk_level}' risk of churning.
    Here are the customer's details:
    {customer_details}

    Based on these details, and the following pre-listed offers, suggest personalized recommendations to retain this customer.
    Focus on suggesting 1-3 highly relevant offers. If no offer is directly relevant, suggest a general retention strategy.
    The tone should be empathetic and persuasive, suitable for an internal report.

    Pre-listed offers tailored to risk level and complaints:
    {relevant_offers}

    Generate the recommendations in a concise markdown format, suitable for an internal report.
    Start with "AI-Powered Retention Strategy:" followed by a bulleted list of recommendations.
    """
    try:
        response_stream = model.generate_content(prompt, stream=True)
        full_response = ""
        for chunk in response_stream:
            full_response += chunk.text
            time.sleep(0.05) # Simulate typing for better UX
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

# --- Tabbed Interface ---
tab1, tab2 = st.tabs(["Customer Churn Prediction", "Data Overview & Training"])

with tab1:
    st.header("1. Customer Search & Prediction")

    # Layout for search and customer details
    col_search, col_details = st.columns([1, 2])

    with col_search:
        search_query_input = st.text_input("Search Customer by ID, Name, or Phone (e.g., 1001, John Doe, 555-1234):",
                                           value=st.session_state['search_query'],
                                           key="search_query_input_text")

        if st.button("Search"):
            st.session_state['search_query'] = search_query_input
            if st.session_state['search_query']:
                search_lower = st.session_state['search_query'].lower()
                results = all_customers_df[
                    all_customers_df['CustomerID'].astype(str).str.lower().str.contains(search_lower) |
                    all_customers_df['PhoneNumber'].astype(str).str.lower().str.contains(search_lower) |
                    all_customers_df['CustomerName'].astype(str).str.lower().str.contains(search_lower)
                ].copy() # Use .copy() to avoid SettingWithCopyWarning
                st.session_state['search_results_display_df'] = results
                st.session_state['selected_customer_id'] = None # Reset selected customer on new search
                st.session_state['selected_customer_data'] = None
                st.session_state['show_prediction_results'] = False
                st.session_state['manual_entry_mode'] = False # Exit manual entry mode if search is performed
            else:
                st.session_state['search_results_display_df'] = pd.DataFrame() # Clear results if search query is empty
                st.session_state['selected_customer_id'] = None
                st.session_state['selected_customer_data'] = None
                st.session_state['show_prediction_results'] = False
                st.session_state['manual_entry_mode'] = False

        if not st.session_state['search_results_display_df'].empty:
            st.subheader("Search Results:")
            st.write(f"{len(st.session_state['search_results_display_df'])} matches found.")
            display_label = st.session_state['search_results_display_df'].apply(
                lambda row: f"{row['CustomerID']} | {row['CustomerName']} | {row['PhoneNumber']}",
                axis=1
            ).tolist() # Convert to list for selectbox options

            # Determine default index for selectbox
            default_index = 0
            if st.session_state['selected_customer_id'] and not st.session_state['manual_entry_mode']:
                # Try to find the currently selected customer in the new search results
                try:
                    current_selected_label = next(
                        label for label in display_label
                        if label.startswith(str(st.session_state['selected_customer_id']) + ' | ')
                    )
                    default_index = display_label.index(current_selected_label)
                except StopIteration:
                    default_index = 0 # Fallback if not found

            st.selectbox(
                "Select a Customer:",
                options=display_label,
                key="selected_customer_from_search_option",
                index=default_index,
                on_change=on_customer_select
            )
        elif st.session_state['search_query'] and not st.session_state['search_results_display_df'].empty:
             st.info("No customers found matching your search query.")
        
        # Add a button to switch to manual entry if no search results or if user wants to manually input
        if st.button("Manual Entry", key="manual_entry_toggle_button"):
            st.session_state['manual_entry_mode'] = True
            st.session_state['selected_customer_id'] = "Manual Entry"
            # Initialize with default/placeholder values for manual entry
            st.session_state['selected_customer_data'] = {
                'CustomerID': 'Manual Entry',
                'CustomerName': 'N/A',
                'PhoneNumber': 'N/A',
                **{col: 0.0 for col in ['CallFailure', 'Complains', 'SubscriptionLength', 'ChargeAmount', 'SecondsUse', 'FrequencyUse', 'FrequencySMS', 'DistinctCalls', 'Age', 'CustomerValue']},
                **{col: (all_customers_df[col].mode()[0] if col in all_customers_df.columns else 1) for col in ['AgeGroup', 'TariffPlan', 'Status']}
            }
            st.session_state['show_prediction_results'] = False
            # Clear search query and results when entering manual mode
            st.session_state['search_query'] = ""
            st.session_state['search_results_display_df'] = pd.DataFrame()


    with col_details:
        st.subheader("Customer Details for Prediction")
        # Check if either a customer is selected or manual entry mode is active
        if st.session_state['selected_customer_data'] is not None or st.session_state['manual_entry_mode']:
            customer_data = st.session_state['selected_customer_data']

            # Create an input form for all features, using session state for consistency
            with st.form("customer_details_form"):
                form_columns = st.columns(2)
                form_index = 0
                numerical_features = [
                    'CallFailure', 'Complains', 'SubscriptionLength', 'ChargeAmount',
                    'SecondsUse', 'FrequencyUse', 'FrequencySMS', 'DistinctCalls', 'Age'
                ]
                categorical_features = [
                    'AgeGroup', 'TariffPlan', 'Status'
                ]
                customer_value_feature = 'CustomerValue'

                # Populate form with current customer data or defaults
                for feature in numerical_features:
                    with form_columns[form_index % 2]:
                        current_value = customer_data.get(feature, 0.0)
                        # Ensure value is float for number_input
                        st.session_state[f'input_{feature}'] = st.number_input(feature, value=float(current_value), key=f"input_{feature}_num")
                    form_index += 1

                for feature in categorical_features:
                    with form_columns[form_index % 2]:
                        # Get options from the full dataset for categorical features
                        options = sorted(all_customers_df[feature].unique().tolist()) if feature in all_customers_df.columns else [1, 2, 3]
                        current_value = customer_data.get(feature, (all_customers_df[feature].mode()[0] if feature in all_customers_df.columns else options[0]) if options else None)

                        # Ensure current_value is a valid option, otherwise default to the first option
                        if current_value not in options and options:
                            current_value = options[0]

                        # Handle the case where there are no options (shouldn't happen if data is loaded)
                        if not options:
                            st.warning(f"No options available for categorical feature: {feature}. Please check data.")
                            st.session_state[f'input_{feature}'] = None # Or a suitable default
                        else:
                            st.session_state[f'input_{feature}'] = st.selectbox(feature, options=options, index=options.index(current_value) if current_value in options else 0, key=f"input_{feature}_cat")
                    form_index += 1

                with form_columns[form_index % 2]:
                    current_value_cv = customer_data.get(customer_value_feature, 0.0)
                    st.session_state[f'input_{customer_value_feature}'] = st.number_input(customer_value_feature, value=float(current_value_cv), key=f"input_{customer_value_feature}_num")

                predict_button = st.form_submit_button("Predict Churn for this Customer")

                if predict_button:
                    # Collect all input values into a dictionary
                    input_data = {
                        feature: st.session_state[f'input_{feature}']
                        for feature in numerical_features + categorical_features + [customer_value_feature]
                    }
                    customer_features_df = pd.DataFrame([input_data])

                    # Ensure the DataFrame has the correct columns for prediction using X_train_columns
                    # Create a DataFrame with all expected training columns, initialized to 0
                    final_customer_features_df = pd.DataFrame(columns=X_train_columns)
                    final_customer_features_df.loc[0] = 0 # Initialize with zeros to handle missing dummified columns

                    # Fill in the values from the input form for numerical features
                    for col in numerical_features + [customer_value_feature]:
                        if col in final_customer_features_df.columns:
                            final_customer_features_df[col].iloc[0] = customer_features_df[col].iloc[0]

                    # Handle one-hot encoding for categorical features
                    for cat_col in ['AgeGroup', 'TariffPlan', 'Status']:
                        if cat_col in customer_features_df.columns:
                            unique_val = customer_features_df[cat_col].iloc[0]
                            # Construct the dummified column name
                            dummy_col_name = f"{cat_col}_{unique_val}"
                            if dummy_col_name in X_train_columns:
                                final_customer_features_df[dummy_col_name].iloc[0] = 1
                            # Special handling for TariffPlan_2 if it's explicitly a separate column for dummification
                            if cat_col == 'TariffPlan' and f'TariffPlan_2' in X_train_columns:
                                if unique_val == 2:
                                    final_customer_features_df['TariffPlan_2'].iloc[0] = 1
                                else:
                                    final_customer_features_df['TariffPlan_2'].iloc[0] = 0 # Ensure it's 0 if not 2

                    # Drop original categorical columns if they exist in X_train_columns (which they shouldn't after dummification)
                    # This line is primarily for robustness if `X_train_columns` still contains original categorical names.
                    final_customer_features_df = final_customer_features_df.drop(columns=['AgeGroup', 'TariffPlan', 'Status'], errors='ignore')

                    # Debugging: Print final DataFrame to ensure correct structure
                    # st.write("Final DataFrame for prediction:")
                    # st.dataframe(final_customer_features_df)

                    with st.spinner("Predicting churn and generating recommendations..."):
                        predictions_info = []

                        for model_name, model_info in ENSEMBLE_MODELS.items():
                            model = model_info['model']
                            model_type = model_info['type']
                            try:
                                pred, prob, _ = predict_churn(model, final_customer_features_df, scaler, X_train_columns, model_type=model_type)
                                predictions_info.append({'Model': model_name, 'Prediction': pred, 'Probability': prob})
                            except Exception as e:
                                st.error(f"Error predicting with {model_name}: {e}")
                                predictions_info.append({'Model': model_name, 'Prediction': -1, 'Probability': 'Error'}) # Indicate error

                        # Display individual model predictions
                        st.subheader("Individual Model Predictions:")
                        prediction_df = pd.DataFrame(predictions_info)
                        prediction_df['Prediction Label'] = prediction_df['Prediction'].apply(lambda x: "Churn" if x == 1 else ("No Churn" if x == 0 else "Error"))
                        prediction_df['Probability'] = prediction_df['Probability'].apply(lambda x: f"{x:.2f}" if isinstance(x, (float, int)) else x) # Format probability
                        st.dataframe(prediction_df[['Model', 'Prediction Label', 'Probability']])

                        # Ensemble Voting (consider only successful predictions for voting)
                        successful_predictions = [p['Prediction'] for p in predictions_info if p['Prediction'] in [0, 1]]
                        churn_votes = sum(successful_predictions)
                        no_churn_votes = len(successful_predictions) - churn_votes

                        ensemble_prediction_label = "No Churn"
                        churn_risk_level = "Low Risk"
                        if churn_votes >= 1: # If even one model predicts churn, consider it high risk
                            ensemble_prediction_label = "Churn"
                            if churn_votes == len(ENSEMBLE_MODELS):
                                churn_risk_level = "Very High Risk"
                            else:
                                churn_risk_level = "High Risk"

                        st.markdown(f"### Ensemble Prediction: **{ensemble_prediction_label}**")
                        st.info(f"**Churn Risk Level:** {churn_risk_level}")

                        # Visualize ensemble votes
                        if successful_predictions: # Only plot if there are valid predictions
                            vote_data = pd.DataFrame({
                                'Vote': ['Models Predicting Churn', 'Models Predicting No Churn'],
                                'Count': [churn_votes, no_churn_votes]
                            })
                            fig_votes = px.bar(vote_data, x='Vote', y='Count',
                                               title='Ensemble Model Votes',
                                               labels={'Count': 'Number of Models'},
                                               color='Vote',
                                               color_discrete_map={'Models Predicting Churn': '#D62728', 'Models Predicting No Churn': '#1F77B4'},
                                               range_y=[0, len(ENSEMBLE_MODELS)],
                                               text_auto=True,
                                               height=400
                                               )
                            fig_votes.update_layout(xaxis_title="Ensemble Vote Outcome", yaxis_title="Number of Models")
                            st.plotly_chart(fig_votes, use_container_width=True)
                        else:
                            st.warning("No successful model predictions to visualize ensemble votes.")


                        st.markdown("---")
                        st.subheader("AI-Powered Recommendations from Google Gemini")

                        # Pass only relevant customer details to Gemini
                        customer_details_for_llm = {k: v for k, v in input_data.items() if k not in ['CustomerID', 'PhoneNumber', 'CustomerName']}
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

                    # Attempt to force garbage collection after prediction (this line should remain)
                    gc.collect()

                    st.session_state['show_prediction_results'] = True # Set flag to show results
        else:
            st.info("Please search and select a customer, or use 'Manual Entry' to input details.")


with tab2:
    st.header("Data Overview and Model Training")
    st.subheader("Customer Data Overview")
    if all_customers_df is not None:
        st.write("Displaying sample of data from `customer_data_with_identifiers.csv` (used for search):")
        st.dataframe(all_customers_df.head())

        st.subheader("Data Statistics (`customer_data_with_identifiers.csv`)")
        st.write(all_customers_df.describe())

        st.subheader("Churn Distribution (`customer_data_with_identifiers.csv`)")
        if 'Churn' in all_customers_df.columns:
            churn_counts = all_customers_df['Churn'].value_counts().rename(index={0: 'No Churn', 1: 'Churn'})
            fig_churn = px.pie(
                names=churn_counts.index,
                values=churn_counts.values,
                title='Distribution of Churn Status',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_churn, use_container_width=True)
        else:
            st.info("No 'Churn' column found in `customer_data_with_identifiers.csv` for distribution analysis.")


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