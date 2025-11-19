import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings
import plotly.graph_objects as go
import plotly.express as px
warnings.filterwarnings('ignore')

# --- UI UPDATE: Sidebar Collapse Logic ---
# This must be the first Streamlit command
if 'model_loaded' not in st.session_state:
    st.session_state['model_loaded'] = False

sidebar_state = "collapsed" if st.session_state['model_loaded'] else "expanded"

# --- UI UPDATE: SVG for page_icon (as a Data URI) ---
page_icon_svg = "data:image/svg+xml,%3Csvg%20viewBox%3D%270%200%2068.918968%20109.39794%27%20fill%3D%27none%27%20xmlns%3D%27http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%27%3E%3Cpath%20d%3D%27M%2053.507868%2C107.78883%207.6318678%2C77.923829%20c%200%2C0%20-5.753%2C-3.308%20-7.07000001%2C-9.47%20-1.84999999%2C-8.652%204.24800001%2C-16.07%204.24800001%2C-16.07%20l%2059.5390002%2C38.775%20c%204.593%2C2.993%205.846%2C9.117001%202.894%2C13.736001%20-3.033%2C4.745%20-9.265%2C5.792%20-13.737%2C2.894%20z%27%20fill%3D%27%2339d201%27%2F%3E%3Cpath%20d%3D%27m%2010.054868%2C45.983829%20c%200%2C0%20-7.0270002%2C-4.243%20-7.0270002%2C-14.057%200%2C-9.78%207.0240002%2C-13.971%207.0240002%2C-13.971%20l%2025.08%2C-16.4140004%20a%209.927%2C9.927%200%200%201%2010.82%2C16.6450004%20l%20-21.138%2C13.764%2036.015%2C23.456%20c%200%2C0%206.483%2C3.714%207.76%2C9.984%201.783%2C8.747%20-4.238%2C15.954%20-4.238%2C15.954%20z%20M%209.9218678%2C109.37183%20c%202.3540002%2C0%204.0300002%2C-0.747%205.6650002%2C-1.807%20l%2012.243%2C-7.975001%20-12.243%2C-7.973%20c%20-1.65%2C-1.063%20-3.382%2C-1.807%20-5.6650002%2C-1.807%20a%209.782%2C9.782%200%201%200%200%2C19.562001%20z%27%20fill%3D%27%230473ea%27%2F%3E%3Cpath%20d%3D%27m%2057.260868%2C22.170829%20c%20-2.355%2C0%20-4.031%2C0.747%20-5.666%2C1.806%20l%20-12.244%2C7.974%2012.244%2C7.974%20c%201.65%2C1.064%203.383%2C1.808%205.666%2C1.808%20a%209.781%2C9.781%200%201%200%200%2C-19.562%27%20fill%3D%27%2339d201%27%2F%3E%3C%2Fsvg%3E"

# Set page configuration
st.set_page_config(
    page_title="Channel Preference Predictor",
    page_icon=page_icon_svg, 
    layout="wide",
    initial_sidebar_state=sidebar_state # <-- DYNAMICALLY SET
)

# Custom CSS for Standard Chartered professional theme
st.markdown("""
<style>
    /* --- Main App Font --- */
    body {
        font-family: 'Arial', sans-serif;
    }
    
    /* --- Sidebar Logo Sizing --- */
    [data-testid="stSidebar"] svg {
        width: 100%; /* Make the logo responsive to sidebar width */
        height: auto;
    }
    
    /* --- Radio Button Styling --- */
    [data-testid="stRadio"] [aria-checked="false"] div:first-child {
        border-color: #0473ea !important; 
    }
    [data-testid="stRadio"] [aria-checked="true"] div:first-child {
        border-color: #0473ea !important; /* Blue ring */
    }
    [data-testid="stRadio"] [aria-checked="true"] div:last-child {
        background-color: #39d201 !important; /* Green dot */
    }

    /* --- Headers --- */
    .main-header {
        font-size: 2.5rem;
        color: #0473ea; 
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: bold;
        font-family: 'Arial', sans-serif;
    }
    .sub-tagline {
        font-size: 1.2rem;
        color: #666666;
        text-align: center;
        margin-bottom: 3rem;
        font-style: italic;
        font-family: 'Arial', sans-serif;
    }
    .section-header {
        font-size: 1.8rem;
        color: #0473ea; 
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: bold;
        border-bottom: 2px solid #39d201; 
        padding-bottom: 0.5rem;
        display: flex; 
        align-items: center;
    }
    
    .subsection-header {
        font-size: 1.3rem;
        color: #0473ea; /* Changed to blue */
        margin-bottom: 0.8rem;
        font-weight: bold;
        display: flex; 
        align-items: center;
    }
    
    /* --- SVG Icon Styling --- */
    .section-header svg, .subsection-header svg {
        width: 1em; 
        height: 1em;
        margin-right: 0.6rem;
        vertical-align: -0.15em;
        stroke-width: 2.5px;
        flex-shrink: 0;
        stroke: #39d201 !important; /* Force green stroke */
    }

    /* --- Tooltip Icon Styling --- */
    .info-icon {
        font-family: 'Arial', sans-serif;
        font-weight: bold;
        font-size: 0.7em;
        cursor: help;
        border: 1.5px solid #666;
        border-radius: 50%;
        padding: 0px 5px;
        color: #666;
        vertical-align: super; /* Aligns it nicely */
        margin-left: 4px;
    }
    
    /* --- Cards --- */
    .profile-card {
        background-color: #f8f9fa; /* Light gray */
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #0473ea; 
        margin-bottom: 1.5rem;
        height: 100%; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .spend-card {
        background-color: #f8f9fa; /* Light gray */
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #39d201; 
        margin-bottom: 1.5rem;
        height: 100%; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .recommendation-card {
        background-color: #e8f4f8; /* Light blue */
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #39d201; 
        margin-bottom: 1.5rem;
    }
    .action-plan-card {
        background-color: #ffffff; /* White */
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #0473ea; 
        margin-top: 1rem;
    }
    
    .channel-box {
        background-color: #ffffff; /* White */
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #0473ea; 
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        height: 100%; 
        display: flex; 
        flex-direction: column; 
        justify-content: center; 
    }
    
    .channel-box-negative {
        background-color: #ffffff; 
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #E50000 !important; /* Red border */
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        height: 100%; 
        display: flex; 
        flex-direction: column; 
        justify-content: center; 
    }


    /* --- Channel Ranking --- */
    .star-rating {
        font-size: 1.2rem; 
        color: #FFD700;
        margin-top: 0.5rem;
    }
    .channel-name {
        font-size: 1.2rem;
        font-weight: bold;
        color: #000000; 
        margin-top: 0.5rem;
    }
    .channel-score {
        font-size: 0.9rem;
        color: #666666; 
        margin-top: 0.3rem;
    }

    /* --- Button Styles --- */
    /* Primary Button (default) */
    .stButton>button {
        width: 100%;
        height: 45px;
        font-size: 1rem;
        font-weight: bold;
        background-color: #0473ea; 
        color: white;
        border: none;
        border-radius: 5px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #035fbc; 
    }
    .stButton>button:active {
        background-color: #024b94; 
    }
    .stButton>button:disabled {
        background-color: #cccccc;
        color: #666666;
    }

    /* Secondary Button (for 'Back' etc.) */
    .stButton>button[kind="secondary"] {
        background-color: #f0f2f5; 
        color: #0473ea; 
        border: 1px solid #0473ea; 
    }
    .stButton>button[kind="secondary"]:hover {
        background-color: #e0e4e8;
        color: #035fbc; 
        border: 1px solid #035fbc; 
    }
    .stButton>button[kind="secondary"]:active {
        background-color: #d0d4d8;
    }
    
    /* Download Button (styled as SC Green) */
    .stDownloadButton>button {
        width: 100%;
        height: 45px;
        font-size: 1rem;
        font-weight: bold;
        background-color: #39d201; 
        color: white;
        border: none;
        border-radius: 5px;
        transition: background-color 0.3s ease;
    }
    .stDownloadButton>button:hover {
        background-color: #2db601; 
    }
    
</style>
""", unsafe_allow_html=True)

class LightGBMPredictor:
    """LightGBM model predictor for channel preference ranking"""

    def __init__(self, model_path='lightgbm_channel_ranker.pkl'):
        """Initialize the LightGBM predictor"""
        try:
            self.model = joblib.load(model_path)
            self.channels = ['Call', 'SMS', 'WhatsApp', 'Email', 'IVR', 'Field_Agent']
        except FileNotFoundError:
            st.error("Model file not found. Please ensure the model file exists.")
            raise
        except Exception as e:
            st.error(f"Error loading model: {e}")
            raise

    def load_customer_data(self, data_path, customer_id=None):
        """Load customer data for prediction"""
        try:
            self.df = pd.read_csv(data_path)
            if customer_id:
                self.customer_data = self.df[self.df['Customer_id'] == customer_id]
                if len(self.customer_data) == 0:
                    available_ids = self.df['Customer_id'].head(5).tolist()
                    raise ValueError(f"Customer ID '{customer_id}' not found! Available: {available_ids}")
            else:
                self.customer_data = self.df
        except Exception as e:
            st.error(f"Error loading customer data: {e}")
            raise

    def load_processed_data(self, data_path):
        """Load processed data"""
        try:
            self.processed_df = pd.read_csv(data_path)
        except Exception as e:
            st.error(f"Error loading processed data: {e}")
            raise

    def load_singapore_data(self, data_path):
        """Load Singapore loan data"""
        try:
            self.singapore_df = pd.read_csv(data_path)
        except Exception as e:
            st.error(f"Error loading Singapore data: {e}")
            raise

    def prepare_features(self, customer_data):
        """Prepare features for prediction"""
        if isinstance(customer_data, pd.Series):
            customer_data = customer_data.to_frame().T

        exclude_cols = ['Customer_id', 'Channel_Preference_Order', 'Preference_Label', 'Top_Channel']
        exclude_cols.extend([col for col in customer_data.columns if 'Prefers_' in col])
        exclude_cols.extend(['Last_Successful_Agent_ID', 'Best_Contact_Agent_IDs'])
        
        feature_cols = [col for col in customer_data.columns if col not in exclude_cols]
        X_customer = customer_data[feature_cols].copy()
        categorical_cols = X_customer.select_dtypes(include=['object']).columns.tolist()

        if categorical_cols:
            for col in categorical_cols:
                try:
                    le = LabelEncoder()
                    X_customer[col] = le.fit_transform(X_customer[col].astype(str))
                except Exception as e:
                    X_customer[col] = 0

        return X_customer.values.flatten(), feature_cols

    def predict_single_customer(self, customer_data):
        """Predict channel preferences for a single customer with 0-1 normalized scores"""
        if isinstance(customer_data, pd.Series):
            customer_data = customer_data.to_frame().T

        customer_id = customer_data['Customer_id'].iloc[0]
        customer_features, feature_cols = self.prepare_features(customer_data)
        channel_predictions = []

        for channel in self.channels:
            channel_features = np.zeros(len(self.channels))
            channel_idx = self.channels.index(channel)
            channel_features[channel_idx] = 1
            combined_features = np.concatenate([customer_features, channel_features])

            try:
                prediction_score = self.model.predict(combined_features.reshape(1, -1))[0]
                channel_predictions.append({
                    'channel': channel,
                    'score': prediction_score,
                    'raw_score': prediction_score,
                    'customer_id': customer_id
                })
            except Exception as e:
                channel_predictions.append({
                    'channel': channel,
                    'score': 0.0,
                    'raw_score': 0.0,
                    'customer_id': customer_id
                })

        # Sort by raw score first
        channel_predictions.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply softmax for 0-1 normalized probabilities
        scores = np.array([pred['score'] for pred in channel_predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probabilities = exp_scores / np.sum(exp_scores)
        
        for i, pred in enumerate(channel_predictions):
            pred['score'] = probabilities[i] # Update score to be the probability

        return channel_predictions

    def predict_multiple_customers(self, start_idx=0, end_idx=None):
        """Predict channel preferences for multiple customers"""
        if end_idx is None:
            end_idx = len(self.df)
        
        customers_to_predict = self.df.iloc[start_idx:end_idx]
        all_predictions = {}
        progress_bar = st.progress(0)

        for idx, (_, customer_row) in enumerate(customers_to_predict.iterrows()):
            customer_df = pd.DataFrame([customer_row])
            predictions = self.predict_single_customer(customer_df)
            all_predictions[customer_row['Customer_id']] = predictions
            progress = (idx + 1) / len(customers_to_predict)
            progress_bar.progress(progress)

        progress_bar.empty()
        return all_predictions

    def get_customer_profile(self, customer_id):
        """Get complete customer profile from all data sources"""
        profile = {'customer_id': customer_id}
        
        # Get from features data
        feature_row = self.df[self.df['Customer_id'] == customer_id]
        if not feature_row.empty:
            if 'Behavioral_Segment' in feature_row.columns:
                profile['customer_segment'] = feature_row['Behavioral_Segment'].iloc[0]
        
        # Get from processed data
        if hasattr(self, 'processed_df'):
            processed_row = self.processed_df[self.processed_df['Customer_id'] == customer_id]
            if not processed_row.empty:
                # Map Income Band
                try:
                    income_val = int(processed_row['Income_Band_SGD'].iloc[0])
                    income_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'} # Add more if needed
                    profile['income_band'] = income_map.get(income_val, str(income_val)) # Use str() as fallback
                except (ValueError, TypeError):
                     profile['income_band'] = processed_row['Income_Band_SGD'].iloc[0]

        # Get from Singapore data
        if hasattr(self, 'singapore_df'):
            sg_row = self.singapore_df[self.singapore_df['Customer_id'] == customer_id]
            if not sg_row.empty:
                # Basic Information
                profile['age'] = sg_row['Age'].iloc[0] if 'Age' in sg_row.columns else 'N/A'
                profile['region'] = sg_row['Region'].iloc[0] if 'Region' in sg_row.columns else 'N/A'
                profile['preferred_language'] = sg_row['Language_Preference'].iloc[0] if 'Language_Preference' in sg_row.columns else 'N/A'
                
                # Financial Analysis
                profile['financial_health'] = sg_row['Financial_Health_Status'].iloc[0] if 'Financial_Health_Status' in sg_row.columns else 'N/A'
                profile['financial_stress'] = sg_row['Finance_Stress_Status'].iloc[0] if 'Finance_Stress_Status' in sg_row.columns else 'N/A'
                profile['aar_risk'] = sg_row['AAR_Risk_Level'].iloc[0] if 'AAR_Risk_Level' in sg_row.columns else 'N/A'
                
                # Convert Flight Risk to Percentage String
                try:
                    risk_score = float(sg_row['Flight_Risk_Score'].iloc[0])
                    profile['flight_risk'] = f"{risk_score * 100:.2f}%"
                except (ValueError, TypeError):
                    profile['flight_risk'] = sg_row['Flight_Risk_Score'].iloc[0] # Keep as 'N/A' or original text

                
                # Agent Interaction
                profile['last_agent'] = sg_row['Last_Successful_Agent_ID'].iloc[0] if 'Last_Successful_Agent_ID' in sg_row.columns else 'N/A'
                profile['best_agents'] = sg_row['Best_Contact_Agent_IDs'].iloc[0] if 'Best_Contact_Agent_IDs' in sg_row.columns else 'N/A'
                profile['avg_time'] = sg_row['Avg_Time_With_Best_Agents_Min'].iloc[0] if 'Avg_Time_With_Best_Agents_Min' in sg_row.columns else 'N/A'
                profile['interaction_count'] = sg_row['Customer_Best_Agent_Interaction_Count'].iloc[0] if 'Customer_Best_Agent_Interaction_Count' in sg_row.columns else 'N/A'
                
                # Spend Analysis
                profile['utility_spend'] = sg_row['Utility_Spend_SGD'].iloc[0] if 'Utility_Spend_SGD' in sg_row.columns else 0
                profile['shopping_spend'] = sg_row['Shopping_Spend_SGD'].iloc[0] if 'Shopping_Spend_SGD' in sg_row.columns else 0
                profile['entertainment_spend'] = sg_row['Entertainment_Spend_SGD'].iloc[0] if 'Entertainment_Spend_SGD' in sg_row.columns else 0
                profile['health_spend'] = sg_row['Health_Spend_SGD'].iloc[0] if 'Health_Spend_SGD' in sg_row.columns else 0
                profile['education_spend'] = sg_row['Education_Spend_SGD'].iloc[0] if 'Education_Spend_SGD' in sg_row.columns else 0
                profile['travel_spend'] = sg_row['Travel_Spend_SGD'].iloc[0] if 'Travel_Spend_SGD' in sg_row.columns else 0
                profile['monthly_income'] = sg_row['Monthly_Income_SGD'].iloc[0] if 'Monthly_Income_SGD' in sg_row.columns else 1
                
                # Transaction Analysis
                profile['upi_count'] = sg_row['UPI_Transaction_Count'].iloc[0] if 'UPI_Transaction_Count' in sg_row.columns else 0
                profile['debit_count'] = sg_row['Debit_Card_Transaction_Count'].iloc[0] if 'Debit_Card_Transaction_Count' in sg_row.columns else 0
                profile['credit_count'] = sg_row['Credit_Card_Transaction_Count'].iloc[0] if 'Credit_Card_Transaction_Count' in sg_row.columns else 0
                profile['cash_count'] = sg_row['Cash_Withdrawal_Count'].iloc[0] if 'Cash_Withdrawal_Count' in sg_row.columns else 0
                profile['recurring_count'] = sg_row['Recurring_Transaction_Count'].iloc[0] if 'Recurring_Transaction_Count' in sg_row.columns else 0
                profile['preferred_payment'] = sg_row['Preferred_Payment_Channel'].iloc[0] if 'Preferred_Payment_Channel' in sg_row.columns else 'N/A'
        
        return profile

def get_star_rating(rank):
    """Get star emoji rating based on rank (1-6)"""
    stars = {
        1: "⭐⭐⭐⭐⭐",
        2: "⭐⭐⭐⭐",
        3: "⭐⭐⭐",
        4: "⭐⭐",
        5: "⭐",
        6: "⭐"
    }
    return stars.get(rank, "")

# Removed get_channel_effectiveness function
# Removed format_business_action_plan function

def main():
    # --- UI UPDATE: SVG for Sidebar Logo ---
    sidebar_logo_svg = """
    <svg width="560" height="220" viewBox="0 0 280 110" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M53.68 107.884 7.804 78.019s-5.753-3.308-7.07-9.47c-1.85-8.652 4.248-16.07 4.248-16.07l59.539 38.775c4.593 2.993 5.846 9.117 2.894 13.736-3.033 4.745-9.265 5.792-13.737 2.894Z" fill="#39d201"/>
    <path d="M10.227 46.079S3.2 41.836 3.2 32.022c0-9.78 7.024-13.971 7.024-13.971l25.08-16.414a9.927 9.927 0 0 1 10.82 16.645L24.986 32.046l36.015 23.456s6.483 3.714 7.76 9.984c1.783 8.747-4.238 15.954-4.238 15.954L10.227 46.079Zm-.133 63.388c2.354 0 4.03-.747 5.665-1.807l12.243-7.975-12.243-7.973c-1.65-1.063-3.382-1.807-5.665-1.807a9.782 9.782 0 1 0 0 19.562Z" fill="#0473EA"/>
    <path d="M57.433 22.266c-2.355 0-4.031.747-5.666 1.806l-12.244 7.974 12.244 7.974c1.65 1.064 3.383 1.808 5.666 1.808a9.781 9.781 0 1 0 0-19.562" fill="#39d201"/>
    <path d="M108.368 78.42c2.589 0 5.003-.702 6.889-2.326v3.642c-2.107 1.448-4.608 2.062-7.241 2.062-6.63 0-11.234-4.608-11.234-11.234 0-6.583 4.739-11.19 11.322-11.19 2.589 0 5.002.747 6.933 2.194v3.993c-1.711-1.842-4.08-2.808-6.714-2.808-4.608 0-7.855 3.204-7.855 7.81 0 4.608 3.292 7.857 7.9 7.857Zm14.393 2.984h-3.555V52.912l3.555 2.197v8.347c1.624-2.501 4.213-4.08 7.504-4.08 5.442 0 8.338 3.992 8.338 9.302v12.726h-3.598V69.292c0-4.213-1.843-6.45-5.618-6.45-3.993 0-6.629 3.422-6.629 7.723l.003 10.84Zm37.915-3.689c-1.798 2.545-4.563 4.081-7.942 4.081-6.319 0-10.839-4.695-10.839-11.102 0-6.536 4.52-11.322 10.839-11.322 3.379 0 6.144 1.58 7.942 4.17v-3.775h3.599v21.634h-3.599v-3.686Zm-15.095-7.065c0 4.476 3.116 7.767 7.504 7.767 4.432 0 7.591-3.291 7.591-7.767 0-4.564-3.159-7.899-7.591-7.899-4.388.003-7.504 3.337-7.504 7.902v-.003Zm27.778 10.754h-3.555V59.77h3.555v4.607c1.711-3.204 4.476-5.002 7.899-5.002v4.081c-4.301-.614-7.899 2.282-7.899 7.635v10.313Zm44.322-9.435H200.61c.527 3.906 3.599 6.495 7.899 6.495 2.765 0 5.442-.79 7.724-2.414v3.423c-2.326 1.58-4.959 2.326-7.724 2.326-6.802 0-11.671-4.388-11.671-11.018 0-6.495 4.125-11.409 10.664-11.409 6.231 0 10.313 4.607 10.313 10.795.001.603-.044 1.206-.134 1.802Zm-17.027-2.894h13.385c-.351-3.687-2.677-6.407-6.495-6.407-3.95-.002-6.32 2.763-6.89 6.405v.002Zm24.619 12.329h-3.555V59.77h3.555v4.607c1.711-3.204 4.475-5.002 7.898-5.002v4.081c-4.299-.614-7.898 2.282-7.898 7.635v10.313Zm29.007-9.435h-17.07c.525 3.906 3.598 6.495 7.898 6.495 2.765 0 5.442-.79 7.724-2.414v3.423c-2.326 1.58-4.959 2.326-7.724 2.326-6.802 0-11.671-4.388-11.671-11.018 0-6.495 4.124-11.409 10.663-11.409 6.231 0 10.312 4.607 10.312 10.795a11.94 11.94 0 0 1-.132 1.802Zm-17.026-2.894h13.383c-.351-3.687-2.677-6.407-6.494-6.407-3.949-.002-6.317 2.763-6.889 6.405v.002Zm38.364 8.614c-1.754 2.59-4.563 4.169-7.942 4.169-6.32 0-10.84-4.696-10.84-11.103 0-6.536 4.52-11.322 10.84-11.322 3.379 0 6.188 1.625 7.942 4.257V52.917l3.599 2.198v26.288h-3.599V77.69Zm-15.095-6.977c0 4.476 3.115 7.768 7.504 7.768 4.432 0 7.591-3.292 7.591-7.768 0-4.564-3.159-7.899-7.591-7.899-4.389 0-7.504 3.334-7.504 7.9Zm-73.054 3.501c0 3.028 1.843 4.213 4.169 4.213a8.584 8.584 0 0 0 5.047-1.58v3.467c-1.494 1.097-3.555 1.536-5.354 1.536-4.608 0-7.417-2.063-7.417-7.9V52.907l3.555 2.197v19.109Zm5.746-14.726a2.745 2.745 0 0 0-2.988 4.606l.076.048a2.742 2.742 0 0 0 2.958 0l3.535-2.267-3.581-2.387Zm-89.016-12.265c2.544 0 4.388-1.097 4.388-3.072 0-1.185-.614-2.106-2.241-2.765l-5.135-2.062c-2.633-1.141-4.081-2.633-4.081-5.442 0-3.774 3.028-6.144 7.416-6.144 2.544 0 4.914.702 6.582 1.975v3.735c-1.974-1.756-4.389-2.765-6.536-2.765-2.283 0-3.907 1.053-3.907 2.809 0 1.316.615 2.106 2.327 2.764l5.135 2.063c2.764 1.185 4.038 2.852 4.038 5.178 0 4.125-3.336 6.67-8.03 6.67-2.854 0-5.443-.79-7.329-2.193v-3.867c2.281 2.15 4.827 3.116 7.373 3.116Zm14.745-4.696c0 3.028 1.842 4.213 4.169 4.213a8.59 8.59 0 0 0 5.042-1.58v3.467c-1.494 1.097-3.556 1.536-5.355 1.536-4.608 0-7.415-2.062-7.415-7.899V23.402l3.554-2.2.005 21.324Zm29.094 3.554c-1.799 2.546-4.564 4.082-7.943 4.082-6.319 0-10.839-4.695-10.839-11.103 0-6.536 4.52-11.322 10.839-11.322 3.379 0 6.144 1.58 7.943 4.17v-3.772h3.599v21.632h-3.599v-3.686Zm-15.096-7.064c0 4.476 3.116 7.767 7.504 7.767 4.433 0 7.592-3.291 7.592-7.767 0-4.564-3.159-7.9-7.592-7.9-4.388 0-7.504 3.336-7.504 7.9Zm27.778 10.751h-3.555V28.135h3.555v3.686c1.624-2.501 4.213-4.081 7.504-4.081 5.442 0 8.338 3.994 8.338 9.304V49.77h-3.598V37.659c0-4.213-1.843-6.451-5.617-6.451-3.994 0-6.63 3.423-6.63 7.724l.003 10.835Zm37.916-3.774c-1.755 2.59-4.564 4.17-7.943 4.17-6.319 0-10.839-4.696-10.839-11.104 0-6.536 4.52-11.322 10.839-11.322 3.379 0 6.188 1.624 7.943 4.257v-8.595l3.599-2.197v28.565h-3.599v-3.774Zm-15.096-6.977c0 4.476 3.116 7.767 7.504 7.767 4.433 0 7.592-3.291 7.592-7.767 0-4.564-3.159-7.9-7.592-7.9-4.385 0-7.504 3.336-7.504 7.9Zm80.923 6.977c-1.755 2.59-4.564 4.17-7.943 4.17-6.32 0-10.84-4.696-10.84-11.104 0-6.536 4.52-11.322 10.84-11.322 3.379 0 6.188 1.624 7.943 4.257v-8.595l3.598-2.197v28.565h-3.598v-3.774Zm-15.096-6.977c0 4.476 3.115 7.767 7.504 7.767 4.432 0 7.592-3.291 7.592-7.767 0-4.564-3.16-7.9-7.592-7.9-4.386 0-7.504 3.336-7.504 7.9Zm-24.488 7.064c-1.799 2.546-4.564 4.082-7.943 4.082-6.319 0-10.839-4.695-10.839-11.103 0-6.536 4.52-11.322 10.839-11.322 3.379 0 6.144 1.58 7.943 4.17v-3.772h3.599v21.632h-3.599v-3.686Zm-15.096-7.064c0 4.476 3.116 7.767 7.504 7.767 4.432 0 7.592-3.291 7.592-7.767 0-4.564-3.16-7.9-7.592-7.9-4.391 0-7.504 3.336-7.504 7.9Zm27.778 10.751h-3.555V28.135h3.555v4.608c1.711-3.204 4.476-5.003 7.899-5.003v4.081c-4.301-.614-7.899 2.282-7.899 7.636v10.31ZM123.954 32.743a2.747 2.747 0 0 0 4.203-1.809 2.738 2.738 0 0 0-.45-2.06 2.744 2.744 0 0 0-.775-.744l-.076-.047a2.743 2.743 0 0 0-2.958.01l-3.533 2.275 3.589 2.375Z" fill="#525355"/>
    </svg>
    """
    
    # Define Lucide SVG strings here to keep code clean
    icon_user = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>'
    icon_clipboard = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><rect width="8" height="4" x="8" y="2" rx="1" ry="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><path d="M12 11h4"/><path d="M12 16h4"/><path d="M8 11h.01"/><path d="M8 16h.01"/></svg>'
    icon_dollar = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>'
    icon_users = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>'
    icon_credit_card = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="14" x="2" y="5" rx="2"/><line x1="2" x2="22" y1="10" y2="10"/></svg>'
    icon_bar_chart_2 = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><line x1="18" x2="18" y1="20" y2="10"/><line x1="12" x2="12" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="14"/></svg>'
    icon_arrow_lr = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3 4 7l4 4"/><path d="M4 7h16"/><path d="M16 21l4-4-4-4"/><path d="M20 17H4"/></svg>'
    icon_target = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>'
    icon_bar_chart = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><line x1="12" x2="12" y1="20" y2="10"/><line x1="18" x2="18" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="16"/></svg>'
    icon_line_chart = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>'
    icon_list = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><line x1="8" x2="21" y1="6" y2="6"/><line x1="8" x2="21" y1="12" y2="12"/><line x1="8" x2="21" y1="18" y2="18"/><line x1="3" x2="3.01" y1="6" y2="6"/><line x1="3" x2="3.01" y1="12" y2="12"/><line x1="3" x2="3.01" y1="18" y2="18"/></svg>'
    icon_pie_chart = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/></svg>'
    icon_megaphone = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><path d="m3 11 18-5v12L3 14v-3z"/><path d="M11.6 16.8a3 3 0 1 1-5.8-1.6"/><path d="M15 8.9V15v0"/></svg>'
    icon_briefcase = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="14" x="2" y="7" rx="2" ry="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>'


    # Initialize session state
    if 'prediction_type' not in st.session_state:
        st.session_state['prediction_type'] = None
    if 'show_prediction_buttons' not in st.session_state:
        st.session_state['show_prediction_buttons'] = False
    if 'batch_predictions' not in st.session_state:
        st.session_state['batch_predictions'] = None

    # Sidebar
    with st.sidebar:
        # Use markdown for SVG logo
        st.markdown(sidebar_logo_svg, unsafe_allow_html=True)
        st.markdown("---")
        
        model_options = {
            "LightGBM": "lightgbm_channel_ranker.pkl",
            "XGBoost": "xgboost_channel_ranker.pkl",
            "RandomForest": "random_forest_channel_ranker.pkl"
        }
        
        # --- UI UPDATE: Replaced selectbox with st.radio ---
        selected_model = st.radio(
            "Select Model",
            list(model_options.keys()),
            index=0, # Default to LightGBM
        )

        if st.button("Load Model & Data", type="primary"):
            model_path = model_options[selected_model]
            try:
                predictor = LightGBMPredictor(model_path)
                predictor.load_customer_data('features_with_channel_labels.csv')
                predictor.load_processed_data('processed_data.csv')
                predictor.load_singapore_data('singapore_loan_data.csv')

                st.session_state['predictor'] = predictor
                st.session_state['model_loaded'] = True
                st.session_state['data_loaded'] = True
                st.session_state['sample_customers'] = predictor.df['Customer_id'].head(20).tolist()
                st.session_state['show_prediction_buttons'] = True
                st.session_state['prediction_type'] = None
                st.session_state['batch_predictions'] = None # Clear old results

                st.success("Model loaded!")
                st.success("Data loaded!")
                
                # --- UI UPDATE: Trigger rerun to collapse sidebar ---
                st.rerun() 
                
            except Exception as e:
                st.error(f"Failed: {e}")
                st.session_state['show_prediction_buttons'] = False

        st.markdown("---")
        # Removed status messages

    # Main Content Area
    predictor = st.session_state.get('predictor')
    
    # Always show header
    st.markdown('<div class="main-header">Channel Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-tagline">Intelligent Customer Contact Channel Engine</div>', unsafe_allow_html=True)
    
    # Show prediction type selection after loading model
    if st.session_state.get('show_prediction_buttons') and st.session_state.get('prediction_type') is None:
        st.markdown('<div class="section-header">Select Prediction Type</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Individual Prediction", use_container_width=True, type="primary"):
                st.session_state['prediction_type'] = 'individual'
                st.rerun()
        
        with col2:
            if st.button("Batch Prediction", use_container_width=True, type="primary"):
                st.session_state['prediction_type'] = 'batch'
                st.session_state['batch_predictions'] = None # Clear old results
                st.rerun()
    
    # Individual Prediction Flow
    elif st.session_state.get('prediction_type') == 'individual':
        if st.button("Back to Selection", type="secondary"):
            st.session_state['prediction_type'] = None
            st.rerun()
        
        st.markdown("---")
        st.markdown('<div class="section-header">Individual Customer Prediction</div>', unsafe_allow_html=True)
        
        sample_customers = st.session_state.get('sample_customers', [])
        
        col1, col2 = st.columns([3, 1])
        with col1:
            customer_id = st.text_input("Enter Customer ID", placeholder="e.g., SCB892183816", key="individual_customer_id")
        with col2:
            st.markdown("<div style='height: 1.8rem;'></div>", unsafe_allow_html=True)
            predict_button = st.button("Predict", type="primary", use_container_width=True)
        
        if sample_customers:
            st.caption(f"Sample IDs: {', '.join(sample_customers[:5])}")
        
        if predict_button and customer_id:
            customer_data = predictor.df[predictor.df['Customer_id'] == customer_id]
            
            if len(customer_data) == 0:
                st.error(f"Customer ID '{customer_id}' not found!")
            else:
                with st.spinner("Analyzing customer preferences..."):
                    customer_df = pd.DataFrame([customer_data.iloc[0]])
                    predictions = predictor.predict_single_customer(customer_df)
                    profile = predictor.get_customer_profile(customer_id)
                
                # 1. Customer Channel Profile Section
                st.markdown("---")
                # --- UI UPDATE: Renamed Header ---
                st.markdown(f'<div class="section-header">{icon_user} Customer Channel Profile</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                # Grid 1: Basic Information
                with col1:
                    st.markdown('<div class="profile-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="subsection-header">{icon_clipboard} Basic Information</div>', unsafe_allow_html=True)
                    
                    tooltip_text = (
                        "High Risk Delinquent: Overdue 30+ days, high financial stress.\n"
                        "Stable Payer: No overdue payments, excellent payment history.\n"
                        "High Value: With large loan size and high income.\n"
                        "Digital Savvy: Heavy app and online banking users.\n"
                        "Traditional: Minimal app and online banking users.\n"
                        "Standard: Mixed channel usage, no strong preferences."
                    )

                    basic_info_md = f"""
                    **Customer ID**: {customer_id}  
                    **Age**: {profile.get("age", "N/A")}  
                    **Region**: {profile.get("region", "N/A")}  
                    **Preferred Language**: {profile.get("preferred_language", "N/A")}  
                    **Customer Segment**: {profile.get("customer_segment", "N/A")} <span class="info-icon" title="{tooltip_text}">i</span>
                    """
                    
                    st.markdown(basic_info_md, unsafe_allow_html=True) 
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Grid 2: Financial Analysis
                with col2:
                    st.markdown('<div class="profile-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="subsection-header">{icon_dollar} Financial Analysis</div>', unsafe_allow_html=True)
                    
                    # --- UI UPDATE: Define tooltips ---
                    income_tooltip = (
                        "50,000 or Below : A\n"
                        "50,000 to 100,000 : B\n"
                        "100,000 to 200,000 : C\n"
                        "200,000 to 300,000 : D\n"
                        "300,000 to 500,000 : E\n"
                        "500,000 or Above : F"
                    )
                    health_tooltip = (
                        "Healthy: Customers with a financial Health score of 0-25.\n"
                        "Moderate: Customers with a financial Health score of 26-50.\n"
                        "Stressed: Customers with a financial Health score of 51-75.\n"
                        "Critical: Customers with a financial Health score of 76-100."
                    )
                    stress_tooltip = (
                        "Low stress: Customers with a financial stress score of 0-25.\n"
                        "Medium stress: Customers with a financial stress score of 26-50.\n"
                        "High stress: Customers with a financial stress score of 51-75.\n"
                        "Extreme High stress: Customers with a financial stress score of 76-100."
                    )
                    aar_tooltip = (
                        "Low: AAR score <= 0.5\n"
                        "Medium: 0.5 < AAR score <= 0.65\n"
                        "High: AAR score > 0.65"
                    )

                    financial_info_md = f"""
                    **Income Band**: {profile.get("income_band", "N/A")} <span class="info-icon" title="{income_tooltip}">i</span>  
                    **Financial Health Status**: {profile.get("financial_health", "N/A")} <span class="info-icon" title="{health_tooltip}">i</span>  
                    **Financial Stress Status**: {profile.get("financial_stress", "N/A")} <span class="info-icon" title="{stress_tooltip}">i</span>  
                    **Customer Risk Level(Rational)**: {profile.get("aar_risk", "N/A")} <span class="info-icon" title="{aar_tooltip}">i</span>  
                    **Flight Risk**: {profile.get("flight_risk", "N/A")}
                    """
                    st.markdown(financial_info_md, unsafe_allow_html=True) # Must be True for <span>
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Grid 3: Agent Interaction
                with col3:
                    st.markdown('<div class="profile-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="subsection-header">{icon_users} Agent Interaction</div>', unsafe_allow_html=True)
                    
                    agent_info_md = f"""
                    **Last Successful Agent ID**: {profile.get("last_agent", "N/A")}  
                    **Best Contact Agent IDs**: {profile.get("best_agents", "N/A")}  
                    **Avg Time With Best Agents (Min)**: {profile.get("avg_time", "N/A")}  
                    **Customer Best Agent Interaction Count**: {profile.get("interaction_count", "N/A")}
                    """
                    st.markdown(agent_info_md, unsafe_allow_html=False)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # 2. Spend Analysis Section (2 vertical grids)
                st.markdown("---")
                st.markdown(f'<div class="section-header">{icon_credit_card} Spend Analysis</div>', unsafe_allow_html=True)
                
                spend_col1, spend_col2 = st.columns(2)
                
                # Grid 1: Category-wise Spend Analysis (in percentages)
                with spend_col1:
                    st.markdown('<div class="spend-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="subsection-header">{icon_bar_chart_2} Category-wise Spend Analysis</div>', unsafe_allow_html=True)
                    
                    monthly_income = profile.get('monthly_income', 1)
                    if monthly_income == 0:
                        monthly_income = 1
                    
                    spend_data = [
                        ("Utility Spend", (profile.get('utility_spend', 0) / monthly_income) * 100),
                        ("Shopping Spend", (profile.get('shopping_spend', 0) / monthly_income) * 100),
                        ("Entertainment Spend", (profile.get('entertainment_spend', 0) / monthly_income) * 100),
                        ("Health Spend", (profile.get('health_spend', 0) / monthly_income) * 100),
                        ("Education Spend", (profile.get('education_spend', 0) / monthly_income) * 100),
                        ("Travel Spend", (profile.get('travel_spend', 0) / monthly_income) * 100)
                    ]
                    spend_data.sort(key=lambda x: x[1], reverse=True) # Sort by percentage
                    
                    category_spend_md = ""
                    for name, pct in spend_data:
                        category_spend_md += f"**{name}**: {pct:.2f}%  \n"

                    st.markdown(category_spend_md, unsafe_allow_html=False)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Grid 2: Transaction Type Analysis
                with spend_col2:
                    st.markdown('<div class="spend-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="subsection-header">{icon_arrow_lr} Transaction Type Analysis</div>', unsafe_allow_html=True)
                    
                    transaction_data = [
                        ("UPI Transaction Count", profile.get("upi_count", 0)),
                        ("Debit Card Transaction Count", profile.get("debit_count", 0)),
                        ("Credit Card Transaction Count", profile.get("credit_count", 0)),
                        ("Cash Withdrawal Count", profile.get("cash_count", 0)),
                        ("Recurring Transaction Count", profile.get("recurring_count", 0))
                    ]
                    transaction_data.sort(key=lambda x: x[1], reverse=True) # Sort by count
                    
                    transaction_info_md = ""
                    for name, count in transaction_data:
                        transaction_info_md += f"**{name}**: {count}  \n"
                    
                    # Add the non-count field at the end
                    transaction_info_md += f"\n**Preferred Payment Channel**: {profile.get('preferred_payment', 'N/A')}"

                    st.markdown(transaction_info_md, unsafe_allow_html=False)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # 3. Primary Recommendation
                st.markdown("---")
                st.markdown(f'<div class="section-header">{icon_target} Primary Recommendation</div>', unsafe_allow_html=True)
                
                top_channel = predictions[0]['channel']
                top_score = predictions[0]['score']
                
                st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
                
                st.markdown(f"### **Recommended Channel**: {top_channel}")
                
                # Calculate confidence
                confidence_gap = predictions[0]['score'] - predictions[1]['score']
                if confidence_gap > 0.3:
                    confidence_level = "HIGH"
                elif confidence_gap > 0.15:
                    confidence_level = "MEDIUM"
                else:
                    confidence_level = "LOW"
                
                top_score_pct = top_score * 100
                st.markdown(f"**Confidence Level:** {confidence_level} / {top_score_pct:.2f}%")
                
                # Removed Business Action Plan
                
                st.markdown('</div>', unsafe_allow_html=True) # Close recommendation-card

                # 4. Channel Preference Ranking (Horizontal)
                st.markdown("---")
                st.markdown(f'<div class="section-header">{icon_bar_chart} Channel Preference Ranking</div>', unsafe_allow_html=True)
                
                rank_cols = st.columns(6)
                for i, pred in enumerate(predictions, 1):
                    with rank_cols[i-1]:
                        stars = get_star_rating(i) # Use the star function
                        
                        box_class = "channel-box-negative" if pred['raw_score'] < 0 else "channel-box"
                        
                        # --- UI UPDATE: Score to percentage ---
                        score_pct = pred['score'] * 100
                        st.markdown(f'''
                        <div class="{box_class}">
                            <div class="star-rating">{stars}</div>
                            <div class="channel-name">{pred['channel']}</div>
                            <div class="channel-score">Score: {score_pct:.2f}%</div> 
                        </div>
                        ''', unsafe_allow_html=True)
                
                # 5. Channel Score Visualization (Horizontal Bar Chart)
                st.markdown("---")
                st.markdown(f'<div class="section-header">{icon_line_chart} Channel Score Visualization</div>', unsafe_allow_html=True)
                
                # Create horizontal bar chart using plotly
                channels = [pred['channel'] for pred in predictions]
                scores = [pred['score'] for pred in predictions]
                
                fig = go.Figure(go.Bar(
                    x=scores,
                    y=channels,
                    orientation='h',
                    marker=dict(
                        color=scores,
                        colorscale=[[0, '#e6f1ff'], [1, '#0473ea']],
                        showscale=True,
                        colorbar=dict(title="Probability")
                    ),
                    text=[f'{score:.4f}' for score in scores],
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title="Channel Preference Scores",
                    xaxis_title="Preference Score (Probability)",
                    yaxis_title="Channel",
                    height=400,
                    showlegend=False,
                    yaxis=dict(autorange="reversed")
                )
                
                st.plotly_chart(fig, use_container_width=True)

    # --- Batch Prediction flow ---
    elif st.session_state.get('prediction_type') == 'batch':
        if st.button("Back to Selection", type="secondary"):
            st.session_state['prediction_type'] = None
            st.session_state['batch_predictions'] = None # Clear results
            st.rerun()
        
        st.markdown("---")
        st.markdown('<div class="section-header">Batch Customer Prediction</div>', unsafe_allow_html=True)
        
        total_customers = len(predictor.df) if predictor else 0
        
        st.info(f"Total customers in dataset: **{total_customers}**")
        
        prediction_mode = st.radio(
            "Select Prediction Mode:",
            ["Full Dataset", "Customer Range by Index"],
            horizontal=True
        )
        
        if prediction_mode == "Customer Range by Index":
            st.markdown("### Select Customer Index Range")
            col1, col2 = st.columns(2)
            
            with col1:
                start_idx = st.number_input(
                    "Start Index",
                    min_value=0,
                    max_value=total_customers-1,
                    value=0,
                    step=1
                )
            
            with col2:
                end_idx = st.number_input(
                    "End Index",
                    min_value=start_idx+1,
                    max_value=total_customers,
                    value=min(start_idx+10, total_customers),
                    step=1
                )
            
            st.caption(f"Will process {end_idx - start_idx} customers (Index {start_idx} to {end_idx-1})")
        else:
            start_idx = 0
            end_idx = total_customers
        
        # This button block ONLY calculates and saves to session state
        if st.button("Start Batch Prediction", type="primary", use_container_width=True):
            try:
                with st.spinner(f"Processing {end_idx - start_idx} customers..."):
                    st.session_state['batch_predictions'] = predictor.predict_multiple_customers(start_idx, end_idx)
                
                if st.session_state['batch_predictions']:
                    st.success(f"Predictions completed for {len(st.session_state['batch_predictions'])} customers!")
                
            except Exception as e:
                st.error(f"Error in batch processing: {str(e)}")
                st.session_state['batch_predictions'] = None # Clear on error

        # This block runs IF data exists in session state (i.e., after button click OR filter change)
        if st.session_state.get('batch_predictions'):
            predictions = st.session_state['batch_predictions']

            st.markdown("---")
            st.markdown(f'<div class="section-header">{icon_list} Prediction Results</div>', unsafe_allow_html=True)
            
            # Prepare data for table
            results_data = []
            for customer_id, customer_predictions in predictions.items():
                row = {'Customer_ID': customer_id}
                for i, pred in enumerate(customer_predictions, 1):
                    row[f'Preferred_Channel_{i}'] = pred['channel']
                results_data.append(row)
            
            results_df = pd.DataFrame(results_data)
            
            # --- START UI UPDATE: Histogram and Metrics moved ---
            
            if not results_df.empty:
                st.markdown("### Top Channel Distribution")
                channel_counts = results_df['Preferred_Channel_1'].value_counts()
                fig_dist = px.bar(
                    channel_counts,
                    x=channel_counts.index,
                    y=channel_counts.values,
                    labels={'x': 'Channel', 'y': 'Number of Customers'},
                    color_discrete_sequence=['#0473ea'] 
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            # Calculate metrics from the full results_df
            col1, col2, col3 = st.columns(3)
            
            if not results_df.empty:
                top_channels = results_df['Preferred_Channel_1']
                channel_counts = top_channels.value_counts()
                most_preferred = channel_counts.index[0] if not channel_counts.empty else "N/A"
                unique_channels = len(channel_counts)
            else:
                most_preferred = "N/A"
                unique_channels = 0

            with col1:
                st.metric("Total Processed", len(results_df))
            with col2:
                st.metric("Most Preferred", most_preferred)
            with col3:
                st.metric("Unique Channels", unique_channels)
            
            st.markdown("---") # Add separator

            # --- Add the filter dropdown ---
            channel_options = ['All'] + predictor.channels
            selected_channel = st.selectbox(
                "Filter by Preferred_Channel_1:",
                options=channel_options,
                index=0 # 'All' is default
            )

            # --- Apply the filter ---
            if selected_channel == 'All':
                filtered_df = results_df
            else:
                filtered_df = results_df[results_df['Preferred_Channel_1'] == selected_channel]

            # Display filtered table
            st.dataframe(filtered_df, use_container_width=True, height=400)
            
            st.markdown("---") 
            
            # --- Download button for filtered_df ---
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name=f"batch_predictions_{start_idx}_{end_idx}_filtered.csv",
                mime="text/csv",
                type="primary" 
            )
            # --- END UI UPDATE ---

    # Initial state - no model loaded
    elif not st.session_state.get('show_prediction_buttons'):
        st.info("Please load the model and data from the sidebar to begin predictions.")

    # Footer
    st.markdown("---")
    st.caption("© Standard Chartered Bank")

if __name__ == "__main__":
    main()