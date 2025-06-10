import streamlit as st
import pandas as pd
import os
import sys
import plotly.express as px
import time
import joblib
import gc

import google.generativeai as genai
from tf_keras import backend as K
import tensorflow as tf
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    from st_copy_to_clipboard import st_copy_to_clipboard
except ImportError:
    st_copy_to_clipboard = None

# --- Company Branding ---
COMPANY_NAME = "ABC Telecom"

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title=f"Customer Churn Prediction - {COMPANY_NAME}",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Theming and UI ---
st.markdown(f"""
<style>
/* Overall Page Background */
body {{
    background: linear-gradient(to right, #F0F2F6, #E0E7FF); /* Subtle light blue gradient */
    color: #212529; /* Main text color */
}}

/* Main Title */
.stApp > header {{
    background-color: transparent; /* Make header transparent for full-width title */
}}
h1 {{
    color: #004080; /* Darker blue for primary headings */
    text-align: center;
    font-size: 2.5em;
    padding-bottom: 10px;
    border-bottom: 2px solid #007BFF;
    margin-bottom: 30px;
}}

/* Subheaders */
h2, h3, h4 {{
    color: #0056B3; /* Slightly lighter blue for subheadings */
    border-bottom: 1px solid #C0D0E0; /* Subtle underline */
    padding-bottom: 5px;
    margin-top: 25px;
    margin-bottom: 15px;
}}

/* Buttons */
.stButton > button {{
    background-color: #007BFF;
    color: white;
    border-radius: 5px;
    padding: 10px 20px;
    font-size: 16px;
    border: none;
    transition: background-color 0.3s ease;
}}
.stButton > button:hover {{
    background-color: #0056B3;
    color: white;
}}

/* Text Inputs and Select Boxes */
.stTextInput > div > div > input,
.stSelectbox > div > div > div > div {{
    border: 1px solid #A0B0C0;
    border-radius: 5px;
    padding: 8px 10px;
    background-color: white;
}}

/* DataFrame and other data displays */
.stDataFrame {{
    border: 1px solid #D0D9E0;
    border-radius: 5px;
}}

/* Alerts */
.stAlert {{
    border-radius: 5px;
    background-color: #E6F7FF; /* Light blue background for info/success alerts */
    color: #004080; /* Dark blue text */
    border-left: 5px solid #007BFF; /* Primary color border */
}}
.stAlert.success {{
    background-color: #D4EDDA;
    color: #155724;
    border-left-color: #28A745;
}}
.stAlert.warning {{
    background-color: #FFF3CD;
    color: #856404;
    border-left-color: #FFC107;
}}
.stAlert.error {{
    background-color: #F8D7DA;
    color: #721C24;
    border-left-color: #DC3545;
}}


/* Custom flash messages (if used) */
.stAlert.flash-message {{
    position: fixed;
    top: 10px;
    right: 10px;
    width: auto;
    max-width: 300px;
    z-index: 9999;
    animation: fadeout 5s forwards;
}}

@keyframes fadeout {{
    0% {{ opacity: 1; }}
    80% {{ opacity: 1; }}
    100% {{ opacity: 0; display: none; }}
}}

/* Adjust sidebar elements */
.stSidebar > div:first-child {{
    background-color: #E0E7FF; /* Matches secondaryBackgroundColor from config.toml */
}}

/* --- Selectbox Display Area Enhancement --- */
/* This targets the main selectbox container */
div[data-baseweb="select"] {{
    width: 100% !important; /* Ensure it takes full width of its column */
    min-width: 300px; /* Significantly increase minimum width of the selectbox itself */
}}

/* This targets the input area of the selectbox where the selected value is shown */
div[data-baseweb="select"] > div:first-child {{
    min-height: 60px; /* Give the input area a bit more height */
    /* Ensure content is not cut off at the selected value display area */
    overflow: hidden; /* Hide overflow horizontally */
    text-overflow: ellipsis; /* Add ellipsis for overflowing text */
    white-space: nowrap; /* Keep text on a single line */
}}

/* This targets the actual dropdown list of options */
div[data-baseweb="select"] ul[role="listbox"] {{
    max-height: 300px; /* Increase visible height of dropdown list */
    overflow-y: auto; /* Enable scrolling if options exceed max-height */
    width: auto; /* Allow content to dictate width, or set a min-width */
    min-width: 600px; /* Ensure a minimum width for the dropdown list itself, matching the selectbox */
    padding-right: 10px; /* Add some padding if scrollbar is present */
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1); /* Add a subtle shadow for better visual separation */
    border-radius: 5px; /* Rounded corners for the dropdown */
}}

div[data-baseweb="select"] ul[role="listbox"] li {{
    white-space: normal; /* Allow text to wrap if it's too long in the options list */
    height: auto; /* Adjust height based on content */
    padding: 8px 10px; /* Adjust padding for better spacing */
    transition: background-color 0.2s ease; /* Smooth hover effect */
}}
div[data-baseweb="select"] ul[role="listbox"] li:hover {{
    background-color: #E0E7FF; /* Light blue on hover */
}}

</style>
""", unsafe_allow_html=True)

st.success("Application started and dependencies checked.")

# --- Configure Google Gemini API ---
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        gemini_api_key = st.secrets.get("GEMINI_API_KEY")
        if not gemini_api_key:
            st.sidebar.error("GEMINI_API_KEY not found in environment variables or Streamlit secrets.")
            raise ValueError("GEMINI_API_KEY is not set.")

    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    st.sidebar.success("Gemini API configured successfully!")
except Exception as e:
    st.sidebar.warning(f"Gemini API not configured. Please set GEMINI_API_KEY: {e}")
    gemini_model = None

# --- Paths Configuration ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_script_dir

src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

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

@st.cache_resource(hash_funcs={tf.keras.Model: lambda _: None})
def get_model_assets():
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    gc.collect()

    try:
        loaded_assets = load_all_models()
        return loaded_assets
    except Exception as e:
        st.error(f"Error loading model assets: {e}. Ensure models are trained and saved correctly by running `python src/model_trainer.py`.")
        st.stop()

assets = get_model_assets()

scaler = assets.get('scaler')
xgb_model = assets.get('xgb_smote')
ann_class_weights_model = assets.get('ann_class_weights')
ann_smote_model = assets.get('ann_smote')
ann_focal_loss_model = assets.get('ann_focal_loss')
X_train_columns = assets.get('X_train_columns')

if not all([scaler, xgb_model, ann_class_weights_model, ann_smote_model, ann_focal_loss_model, X_train_columns]):
    st.error("One or more models/scaler/training columns could not be loaded. Ensure models are trained and saved correctly by running `python src/model_trainer.py`.")
    st.stop()

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
        st.warning(f"Customer identifier data file not found: {customer_data_path}.")
        # Attempt to create a dummy if file doesn't exist to prevent immediate crash during dev
        try:
            pd.DataFrame({'CustomerID': ['C101'], 'CustomerName': ['John Doe'], 'PhoneNumber': ['555-1234']}).to_csv(customer_data_path, index=False)
            st.info(f"Created a dummy '{os.path.basename(customer_data_path)}' for initial run. Please replace with actual data.")
            return pd.DataFrame() # Return empty to indicate actual data wasn't loaded this time
        except Exception as e:
            st.error(f"Failed to create dummy file: {e}")
            return pd.DataFrame()


    try:
        df_customers = pd.read_csv(customer_data_path)
        required_id_cols = ['CustomerID', 'CustomerName', 'PhoneNumber']

        missing_required_cols = [col for col in required_id_cols if col not in df_customers.columns]
        if missing_required_cols:
            st.warning(f"Missing crucial identifier columns: {', '.join(missing_required_cols)}. Search may be impaired.")

        if 'CustomerID' in df_customers.columns:
            df_customers['CustomerID'] = df_customers['CustomerID'].astype(str)

        for col in required_id_cols:
            if col in df_customers.columns:
                if pd.api.types.is_object_dtype(df_customers[col]):
                    df_customers[col] = df_customers[col].astype(str).replace(r'^\s*$', np.nan, regex=True)

        df_customers.dropna(subset=[col for col in required_id_cols if col in df_customers.columns], inplace=True)

        if df_customers.empty:
            st.error("No valid customer records found after filtering for complete identifiers. Check `customer_data_with_identifiers.csv`.")
            return pd.DataFrame()

        return df_customers

    except pd.errors.EmptyDataError:
        st.error(f"'{os.path.basename(customer_data_path)}' is empty or contains no data.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading customer identifier data from '{os.path.basename(customer_data_path)}': {e}. Search functionality will be disabled.")
        return pd.DataFrame()

all_customers_df = load_customer_identifiers_data()

if all_customers_df.empty:
    st.error("Cannot proceed as no valid customer identifier data could be loaded. Ensure `data/customer_data_with_identifiers.csv` is correctly formatted and contains data.")
    st.stop()


@st.cache_data
def load_customer_churn_data():
    customer_churn_path = os.path.join(project_root, 'data', 'customer_churn.csv')

    if not os.path.exists(customer_churn_path):
        st.warning(f"Customer churn data file not found: {customer_churn_path}.")
        # Attempt to create a dummy if file doesn't exist to prevent immediate crash during dev
        try:
            # Create a dummy dataframe with some required columns
            dummy_data = {
                'CustomerID': ['C101', 'C102'],
                'AgeGroup': ['20-30', '30-40'],
                'MonthlyBill': [50.0, 75.0],
                'Complains': [0, 1],
                'TenureMonths': [12, 24],
                'TotalUsageGB': [100, 200],
                'TariffPlan': ['Basic', 'Premium'],
                'Status': ['Active', 'Active'],
                'Churn': [0, 1] # Example churn values
            }
            pd.DataFrame(dummy_data).to_csv(customer_churn_path, index=False)
            st.info(f"Created a dummy '{os.path.basename(customer_churn_path)}' for initial run. Please replace with actual data and run model_trainer.py.")
            return pd.DataFrame() # Return empty to indicate actual data wasn't loaded this time
        except Exception as e:
            st.error(f"Failed to create dummy file: {e}")
            return pd.DataFrame()


    try:
        df_churn = pd.read_csv(customer_churn_path)

        if 'Churn' in df_churn.columns:
            df_churn['Churn'] = pd.to_numeric(df_churn['Churn'], errors='coerce')
            df_churn.dropna(subset=['Churn'], inplace=True)
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

if customers_churn_df.empty:
    st.error("Cannot proceed as `customer_churn.csv` could not be loaded or is empty. This data is needed to define features for manual entry.")
    st.stop()


# --- Session State Initialization ---
if 'selected_customer_id' not in st.session_state: st.session_state['selected_customer_id'] = None
if 'selected_customer_data' not in st.session_state: st.session_state['selected_customer_data'] = None
if 'search_query' not in st.session_state: st.session_state['search_query'] = ""
if 'search_results_display_df' not in st.session_state: st.session_state['search_results_display_df'] = pd.DataFrame()
if 'show_prediction_results' not in st.session_state: st.session_state['show_prediction_results'] = False
if 'selected_customer_from_search_option' not in st.session_state: st.session_state['selected_customer_from_search_option'] = None
if 'manual_entry_mode' not in st.session_state: st.session_state['manual_entry_mode'] = False


# --- Callback for customer selection ---
def on_customer_select():
    selected_option_label = st.session_state.selected_customer_from_search_option
    if selected_option_label:
        selected_customer_id_from_label = selected_option_label.split(' | ')[0]
        filtered_df = all_customers_df[
            all_customers_df['CustomerID'].astype(str) == selected_customer_id_from_label
        ]

        if not filtered_df.empty:
            selected_customer_row = filtered_df.iloc[0]
            st.session_state['selected_customer_id'] = selected_customer_row['CustomerID']
            st.session_state['selected_customer_data'] = selected_customer_row.to_dict()
            st.session_state['show_prediction_results'] = False
            st.session_state['manual_entry_mode'] = False

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


# --- Function to generate recommendations using Gemini ---
def get_gemini_recommendations(model, churn_risk_level, customer_name, customer_details, offers, company_name, customer_complains=0):
    if not model:
        yield "Gemini model is not initialized. Cannot generate recommendations."
        return

    relevant_offers_list = []
    if churn_risk_level in offers:
        relevant_offers_list.append(offers[churn_risk_level])
    if customer_complains > 0 and "Complaints" in offers:
        relevant_offers_list.append(offers["Complaints"])

    relevant_offers_str = "\n\n".join(relevant_offers_list) if relevant_offers_list else "No specific pre-listed offers to suggest at this time."

    prompt = f"""
    ABC Telecom - Churn Retention Brief
    Customer: {customer_name} - {churn_risk_level} Churn Risk

    You are an AI assistant preparing a concise churn retention brief for a customer service representative at {company_name}.

    **Customer Profile Summary:**
    Based on the following key customer attributes, provide a concise, readable summary (1-2 sentences) highlighting factors relevant to churn risk (e.g., low usage, high complaints, long tenure, specific service issues):
    {customer_details}

    Based on this profile, and the available retention offers, generate an actionable brief for the customer rep.
    Focus on:
    1.  **Potential Churn Drivers:** Identify 1-2 most likely reasons for churn based on the customer's profile. Be specific about how the data points suggest the driver.
    2.  **Recommended Retention Strategy:** Suggest 1-2 highly relevant pre-listed offers. If the profile indicates service issues (e.g., high call failures, complaints, low charge amount possibly indicating billing issue), prioritize offers that address problem resolution and customer satisfaction first, followed by incentives.
    3.  **Key Talking Points for Rep:** Provide 2-3 empathetic and persuasive phrases or strategies the rep can use. For service issues, include phrases that acknowledge the problem, validate the customer's experience, and offer clear, concrete next steps for resolution. Ensure these points are practical and action-oriented for the rep.

    Pre-listed offers for guidance:
    {relevant_offers_str}

    Format your response in clear markdown using bullet points for each section.
    """
    try:
        response_stream = model.generate_content(prompt, stream=True)
        full_response = ""
        for chunk in response_stream:
            if chunk.text:
                full_response += chunk.text
                time.sleep(0.01)
                yield full_response
        if full_response:
            yield full_response
    except Exception as e:
        yield f"An error occurred while generating recommendations: {e}"


# --- Streamlit UI ---
st.title(f"Customer Churn Prediction Dashboard - {COMPANY_NAME}")

st.markdown("""
    This dashboard predicts customer churn based on various service usage patterns and demographic information.
    It leverages advanced machine learning models and provides AI-powered retention strategies using Google Gemini,
    designed to assist our customer representatives.
""")

# --- Tabbed Interface ---
tab1, tab2 = st.tabs(["Customer Churn Prediction", "Data Overview & Training"])

with tab1:
    st.header("Customer Churn Prediction")

    col_search, col_details = st.columns([1, 2])

    with col_search:
        st.subheader("1. Find Customer")
        search_query_input = st.text_input("Search by ID, Name, or Phone:",
                                            value=st.session_state['search_query'],
                                            key="search_query_input_text")

        search_button = st.button("Search Customer", key="search_button")
        manual_entry_button = st.button("Enter Manually", key="manual_entry_toggle_button")

        if search_button:
            st.session_state['search_query'] = search_query_input.strip()
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

                st.session_state['selected_customer_id'] = None
                st.session_state['selected_customer_data'] = None
                st.session_state['show_prediction_results'] = False
                st.session_state['manual_entry_mode'] = False
            else:
                st.session_state['search_results_display_df'] = pd.DataFrame()
                st.info("Please enter a search query.")
                st.session_state['selected_customer_id'] = None
                st.session_state['selected_customer_data'] = None
                st.session_state['show_prediction_results'] = False
                st.session_state['manual_entry_mode'] = False

        if manual_entry_button:
            st.session_state['manual_entry_mode'] = True
            st.session_state['selected_customer_id'] = "Manual Entry"
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
                'CustomerName': 'N/A', # Set to N/A for manual entry
                'PhoneNumber': 'N/A',
                **default_numeric,
                **default_categorical
            }
            st.session_state['show_prediction_results'] = False
            st.session_state['search_query'] = ""
            st.session_state['search_results_display_df'] = pd.DataFrame()


        if not st.session_state['search_results_display_df'].empty:
            st.markdown("---")
            st.subheader("Search Results")
            st.write(f"{len(st.session_state['search_results_display_df'])} matches found.")
            display_label = st.session_state['search_results_display_df'].apply(
                lambda row: f"{row.get('CustomerID', 'N/A')} | {row.get('CustomerName', 'N/A')} | {row.get('PhoneNumber', 'N/A')}",
                axis=1
            ).tolist()

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
        elif st.session_state['search_query'] and not all_customers_df.empty:
            st.info("No customers found matching your search query.")


    with col_details:
        st.subheader("2. Customer Profile & Input")
        if st.session_state['selected_customer_data'] is not None or st.session_state['manual_entry_mode']:
            customer_data = st.session_state['selected_customer_data']

            if any(pd.isna(v) for v in customer_data.values()):
                st.warning("Some feature values are missing (NaN) for this customer. Review and adjust inputs.")

            # st.json({k: (v if pd.notna(v) else 'NaN/Missing') for k,v in customer_data.items()})

            with st.form("customer_details_form"):
                form_columns = st.columns(2)
                form_index = 0

                all_features_from_churn_df = [col for col in customers_churn_df.columns if col != 'Churn']
                explicit_categorical_features = ['AgeGroup', 'TariffPlan', 'Status']

                numerical_features = customers_churn_df.select_dtypes(include=np.number).columns.drop(
                    ['Churn'] + [col for col in explicit_categorical_features if col in customers_churn_df.columns],
                    errors='ignore'
                ).tolist()

                inferred_categorical_features = customers_churn_df.select_dtypes(exclude=np.number).columns.drop(['Churn'], errors='ignore').tolist()
                categorical_features = list(set(explicit_categorical_features + inferred_categorical_features))
                categorical_features.sort()
                numerical_features.sort()


                for feature in numerical_features:
                    with form_columns[form_index % 2]:
                        current_value = customer_data.get(feature)
                        if pd.isna(current_value):
                            current_value = customers_churn_df[feature].mean() if customers_churn_df is not None and feature in customers_churn_df.columns else 0.0

                        st.session_state[f'input_{feature}'] = st.number_input(feature, value=float(current_value), key=f"input_{feature}_num")
                    form_index += 1

                for feature in categorical_features:
                    with form_columns[form_index % 2]:
                        unique_values = customers_churn_df[feature].unique()
                        options = sorted([str(x) for x in unique_values if pd.notna(x)])

                        if not options:
                            st.warning(f"No valid options found for categorical feature: {feature}. Check your data.")
                            st.session_state[f'input_{feature}'] = None
                            continue

                        current_value = customer_data.get(feature)
                        current_value_str = str(current_value) if pd.notna(current_value) else options[0]

                        if current_value_str not in options:
                            default_option = str(customers_churn_df[feature].mode()[0]) if customers_churn_df is not None and feature in customers_churn_df.columns and not customers_churn_df[feature].mode().empty else options[0]
                            current_value_str = default_option

                        st.session_state[f'input_{feature}'] = st.selectbox(feature, options=options, index=options.index(current_value_str) if current_value_str in options else 0, key=f"input_{feature}_cat")
                    form_index += 1

                predict_button = st.form_submit_button("Predict Churn for this Customer")

                if predict_button:
                    input_data = {
                        feature: st.session_state[f'input_{feature}']
                        for feature in numerical_features + categorical_features
                    }

                    customer_input_df = pd.DataFrame([input_data])

                    for col in explicit_categorical_features:
                        if col in customer_input_df.columns:
                            customer_input_df[col] = customer_input_df[col].astype(str).astype('category')

                    customer_processed_df = pd.get_dummies(customer_input_df, drop_first=True)

                    final_customer_features_df = customer_processed_df.reindex(columns=X_train_columns, fill_value=0)

                    if final_customer_features_df.empty:
                        st.error("Error: Prepared features DataFrame is empty. Cannot make predictions.")
                        st.stop()
                    if final_customer_features_df.shape[1] != len(X_train_columns):
                        st.error(f"Error: Number of features ({final_customer_features_df.shape[1]}) does not match expected ({len(X_train_columns)}).")
                        st.stop()

                    predictions_info = []

                    with st.spinner("Predicting churn and generating recommendations..."):
                        for model_name, model_info in ENSEMBLE_MODELS.items():
                            model = model_info['model']
                            model_type = model_info['type']
                            try:
                                pred, prob, _ = make_predictions(model, final_customer_features_df, scaler, X_train_columns, model_type=model_type)
                                predictions_info.append({'Model': model_name, 'Prediction': pred, 'Probability': prob})
                            except Exception as e:
                                st.error(f"Error predicting with {model_name}: {e}")
                                predictions_info.append({'Model': model_name, 'Prediction': -1, 'Probability': 'Error'})

                        prediction_df = pd.DataFrame(predictions_info)
                        prediction_df['Prediction Label'] = prediction_df['Prediction'].apply(lambda x: "Churn" if x == 1 else ("No Churn" if x == 0 else "Error"))
                        prediction_df['Numerical Probability'] = pd.to_numeric(
                            prediction_df['Probability'], errors='coerce'
                        )

                        st.markdown("---")
                        st.subheader("3. Prediction Results")

                        successful_predictions = [p['Prediction'] for p in predictions_info if p['Prediction'] in [0, 1]]
                        churn_votes = sum(successful_predictions)
                        no_churn_votes = len(successful_predictions) - churn_votes

                        ensemble_prediction_label = "No Churn"
                        churn_risk_level = "Low Risk"
                        if churn_votes > 0:
                            if churn_votes == len(ENSEMBLE_MODELS):
                                ensemble_prediction_label = "Churn"
                                churn_risk_level = "Very High Risk"
                            elif churn_votes >= (len(ENSEMBLE_MODELS) / 2):
                                ensemble_prediction_label = "Churn"
                                churn_risk_level = "High Risk"
                            else:
                                ensemble_prediction_label = "Potentially Churn"
                                churn_risk_level = "Medium Risk"

                        # Display Customer ID and Risk Level prominently
                        customer_display_id = st.session_state['selected_customer_id'] if st.session_state['selected_customer_id'] else "N/A"
                        st.markdown(f"### Customer ID: **{customer_display_id}** | Churn Risk: <span style='color: {'#D62728' if 'High' in churn_risk_level else ('#FFC107' if 'Medium' in churn_risk_level else '#1F77B4')}; font-weight: bold;'>{churn_risk_level}</span>", unsafe_allow_html=True)
                        st.markdown("<hr style='border:1px solid #C0D0E0;'>", unsafe_allow_html=True)


                        st.markdown("#### Individual Model Churn Probabilities")
                        fig = px.bar(
                            prediction_df,
                            x='Model',
                            y='Numerical Probability',
                            color='Prediction Label',
                            color_discrete_map={'Churn': '#D62728', 'No Churn': '#1F77B4', 'Error': '#AAAAAA'},
                            title='Churn Probability by Model',
                            labels={'Numerical Probability': 'Churn Probability (%)', 'Model': 'Machine Learning Model'},
                            text_auto=True,
                            height=450
                        )
                        fig.update_traces(texttemplate='%{y:.2%}', textposition='outside')
                        fig.update_layout(yaxis_tickformat=".0%")
                        st.plotly_chart(fig, use_container_width=True)

                        if successful_predictions:
                            vote_data = pd.DataFrame({
                                'Vote': ['Models Predicting Churn', 'Models Predicting No Churn'],
                                'Count': [churn_votes, no_churn_votes]
                            })
                            # fig_votes = px.bar(vote_data, x='Vote', y='Count',
                            #                    title='Ensemble Model Votes',
                            #                    labels={'Count': 'Number of Models'},
                            #                    color='Vote',
                            #                    color_discrete_map={'Models Predicting Churn': '#D62728', 'Models Predicting No Churn': '#1F77B4'},
                            #                    range_y=[0, len(ENSEMBLE_MODELS)],
                            #                    text_auto=True,
                            #                    height=400
                            #                    )
                            # fig_votes.update_layout(xaxis_title="Prediction Outcome", yaxis_title="Number of Models")
                            # st.plotly_chart(fig_votes, use_container_width=True)
                        else:
                            st.warning("No successful model predictions to visualize ensemble votes.")

                        st.markdown("---")
                        st.markdown(f"#### Overall Ensemble Prediction: <span style='color: {'#D62728' if ensemble_prediction_label == 'Churn' else '#1F77B4'}; font-weight: bold;'>{ensemble_prediction_label}</span>", unsafe_allow_html=True)
                        st.info(f"**Calculated Churn Risk Level:** {churn_risk_level}")
                        st.subheader("AI-Powered Retention Recommendations (for Customer Rep)")

                        # --- Retrieve Customer Name for brief ---
                        current_customer_name = st.session_state['selected_customer_data'].get('CustomerName', 'N/A')

                        customer_details_for_llm = {k: v for k, v in input_data.items()}
                        customer_details_for_llm_str = "\n".join([f"- {k}: {v}" for k, v in customer_details_for_llm.items()])

                        if gemini_model:
                            recommendation_placeholder = st.empty()
                            full_recommendation_text = ""
                            for text_chunk in get_gemini_recommendations(
                                gemini_model,
                                churn_risk_level,
                                current_customer_name, # Pass the customer name here
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

                    gc.collect()
                    st.session_state['show_prediction_results'] = True
        else:
            st.info("Please search and select a customer, or use 'Enter Manually' to input details for prediction.")


with tab2:
    st.header("Data Overview and Model Training")
    st.markdown("Explore the underlying data distributions and learn how to manage the predictive models.")

    st.subheader("Customer Data Overview (from `customer_data_with_identifiers.csv`)")
    if not all_customers_df.empty:
        st.write("First 5 rows of customer identifier data:")
        st.dataframe(all_customers_df.head())
    else:
        st.info("No customer identifier data loaded for overview. Check `data/customer_data_with_identifiers.csv`.")

    st.subheader("Data Statistics (from `customer_churn.csv`)")
    if not customers_churn_df.empty:
        st.write("Descriptive statistics of the features used for training:")
        st.dataframe(customers_churn_df.describe())

        st.subheader("Churn Distribution (from `customer_churn.csv`)")
        if 'Churn' in customers_churn_df.columns:
            churn_counts = customers_churn_df['Churn'].value_counts().rename(index={0: 'No Churn', 1: 'Churn'})

            fig_churn = px.bar(
                x=churn_counts.index,
                y=churn_counts.values,
                title='Distribution of Churn Status in Training Data',
                labels={'x': 'Churn Status', 'y': 'Number of Customers'},
                color=churn_counts.index,
                color_discrete_map={'No Churn': '#1F77B4', 'Churn': '#D62728'},
                text_auto=True
            )
            fig_churn.update_layout(xaxis_title="Churn Status", yaxis_title="Number of Customers")
            st.plotly_chart(fig_churn, use_container_width=True)
        else:
            st.info("No 'Churn' column found in `customer_churn.csv` for distribution analysis.")
    else:
        st.info("No customer churn data loaded for analysis. Check `data/customer_churn.csv`.")

    st.subheader("Model Training & Management")
    st.markdown("""
        To train or re-train the machine learning models (XGBoost, ANN models), run the `model_trainer.py` script directly from your terminal:

        ```bash
        python src/model_trainer.py
        ```
        This script will:
        - Load `customer_churn.csv` (the main dataset for training).
        - Preprocess the data (one-hot encoding, scaling).
        - Train the individual models.
        - Save the trained models (e.g., `xgb_model.joblib`, `ann_model.h5`), the `scaler.pkl`, and the `X_train_columns.pkl` (list of feature names) into the `models/` directory.

        **Important:** Always re-run the training script if you modify data or preprocessing logic to ensure the saved models are up-to-date with your application's expectations.
    """)

st.markdown("---")
st.markdown(f"<p style='text-align: center; color: #555;'>Developed by Vinu & UK for {COMPANY_NAME} | Version 1.0</p>", unsafe_allow_html=True)