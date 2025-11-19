import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Channel Preference Predictor",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .insights-card {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff6b6b;
        margin-bottom: 1rem;
    }
    .channel-score {
        font-weight: bold;
        color: #1f77b4;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class LightGBMPredictor:
    """LightGBM model predictor for channel preference ranking"""

    def __init__(self, model_path='lightgbm_channel_ranker.pkl'):
        """
        Initialize the LightGBM predictor
        """
        try:
            # Load the model
            self.model = joblib.load(model_path)
            
            # Define channels (same as training)
            self.channels = ['Call', 'SMS', 'WhatsApp', 'Email', 'IVR', 'Field_Agent']
            
        except FileNotFoundError:
            st.error("❌ Model file not found. Please ensure the model file exists.")
            raise
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            raise

    def load_customer_data(self, data_path, customer_id=None):
        """
        Load customer data for prediction
        """
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
            st.error(f"❌ Error loading customer data: {e}")
            raise

    def load_processed_data(self, data_path):
        """
        Load processed data for age and income band information
        """
        try:
            self.processed_df = pd.read_csv(data_path)
        except Exception as e:
            st.error(f"❌ Error loading processed data: {e}")
            raise

    def prepare_features(self, customer_data):
        """
        Prepare features for prediction
        """
        # Ensure we have a DataFrame
        if isinstance(customer_data, pd.Series):
            customer_data = customer_data.to_frame().T

        # Get feature columns (exclude label and ID columns)
        exclude_cols = ['Customer_id', 'Channel_Preference_Order', 'Preference_Label', 'Top_Channel']
        exclude_cols.extend([col for col in customer_data.columns if 'Prefers_' in col])
        exclude_cols.extend(['Last_Successful_Agent_ID', 'Best_Contact_Agent_IDs'])
        
        feature_cols = [col for col in customer_data.columns if col not in exclude_cols]

        # Extract features
        X_customer = customer_data[feature_cols].copy()

        # Encode categorical features
        categorical_cols = X_customer.select_dtypes(include=['object']).columns.tolist()

        if categorical_cols:
            for col in categorical_cols:
                try:
                    # Use LabelEncoder for categorical columns
                    le = LabelEncoder()
                    X_customer[col] = le.fit_transform(X_customer[col].astype(str))
                except Exception as e:
                    # Use default value for encoding issues
                    X_customer[col] = 0

        # Return flattened array for single customer
        return X_customer.values.flatten(), feature_cols

    def predict_single_customer(self, customer_data):
        """
        Predict channel preferences for a single customer with 0-1 normalized scores
        """
        # Ensure proper DataFrame format
        if isinstance(customer_data, pd.Series):
            customer_data = customer_data.to_frame().T

        customer_id = customer_data['Customer_id'].iloc[0]

        # Prepare customer features
        customer_features, feature_cols = self.prepare_features(customer_data)

        # Create samples for each channel
        channel_predictions = []

        for channel in self.channels:
            # Channel one-hot encoding
            channel_features = np.zeros(len(self.channels))
            channel_idx = self.channels.index(channel)
            channel_features[channel_idx] = 1

            # Combine customer and channel features
            combined_features = np.concatenate([customer_features, channel_features])

            # Make prediction
            try:
                prediction_score = self.model.predict(combined_features.reshape(1, -1))[0]
                channel_predictions.append({
                    'channel': channel,
                    'score': prediction_score,
                    'customer_id': customer_id
                })
            except Exception as e:
                channel_predictions.append({
                    'channel': channel,
                    'score': 0.0,
                    'customer_id': customer_id
                })

        # Sort channels by prediction score (descending)
        channel_predictions.sort(key=lambda x: x['score'], reverse=True)
        
        # Normalize scores to 0-1 range using softmax
        scores = np.array([pred['score'] for pred in channel_predictions])
        
        # Apply softmax to get probabilities between 0-1
        exp_scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
        probabilities = exp_scores / np.sum(exp_scores)
        
        # Update scores with normalized values
        for i, pred in enumerate(channel_predictions):
            pred['score'] = probabilities[i]

        return channel_predictions

    def predict_multiple_customers(self, customer_ids=None):
        """
        Predict channel preferences for multiple customers
        """
        if customer_ids:
            customers_to_predict = self.df[self.df['Customer_id'].isin(customer_ids)]
            if len(customers_to_predict) == 0:
                st.error("❌ No matching customer IDs found!")
                return {}
        else:
            customers_to_predict = self.customer_data

        all_predictions = {}
        progress_bar = st.progress(0)

        for idx, (_, customer_row) in enumerate(customers_to_predict.iterrows()):
            customer_df = pd.DataFrame([customer_row])
            predictions = self.predict_single_customer(customer_df)
            all_predictions[customer_row['Customer_id']] = predictions

            # Update progress bar
            progress = (idx + 1) / len(customers_to_predict)
            progress_bar.progress(progress)

        progress_bar.empty()
        return all_predictions

    def get_customer_insights(self, customer_id):
        """
        Get additional insights about a customer from processed_data.csv
        """
        if not hasattr(self, 'processed_df'):
            return None
            
        customer_row = self.processed_df[self.processed_df['Customer_id'] == customer_id]
        if customer_row.empty:
            return None

        insights = {
            'customer_id': customer_id,
            'age': customer_row['Age'].iloc[0] if 'Age' in customer_row.columns else 'N/A',
            'income_band': customer_row['Income_Band_SGD'].iloc[0] if 'Income_Band_SGD' in customer_row.columns else 'N/A'
        }

        return insights

def main():
    # Header
    st.markdown('<div class="main-header">📊 Channel Preference Predictor</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("This predictor uses LightGBM to rank customer channel preferences based on their profile and behavior data.")
        
        st.markdown("---")
        st.subheader("Data Sources")
        st.write("**Prediction Data:** features_with_channel_labels.csv")
        st.write("**Customer Insights:** processed_data.csv")
        st.write("**Scores:** Normalized 0-1 using softmax")
        
        st.markdown("---")
        st.subheader("Available Channels")
        channels = ['Call', 'SMS', 'WhatsApp', 'Email', 'IVR', 'Field_Agent']
        for channel in channels:
            st.write(f"• {channel}")

    # Initialize predictor
    try:
        predictor = LightGBMPredictor('lightgbm_channel_ranker.pkl')
        predictor.load_customer_data('features_with_channel_labels.csv')
        predictor.load_processed_data('processed_data.csv')
        
        # Get sample customer IDs for suggestions
        sample_customers = predictor.df['Customer_id'].head(20).tolist()
        
    except Exception as e:
        st.error(f"Failed to initialize predictor: {e}")
        return

    # Main content - Individual and Batch Prediction Tabs
    tab1, tab2 = st.tabs(["🔍 Individual Prediction", "📊 Batch Prediction"])
    
    with tab1:
        st.subheader("Individual Customer Prediction")
        
        # Create two columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            customer_id = st.text_input("Enter Customer ID:", placeholder="e.g., SCB892183816")
            
        with col2:
            st.write("**Sample Customer IDs:**")
            for i, cust_id in enumerate(sample_customers[:5]):
                st.caption(f"{i+1}. {cust_id}")

        if customer_id:
            try:
                # Check if customer exists
                customer_data = predictor.df[predictor.df['Customer_id'] == customer_id]
                if len(customer_data) == 0:
                    st.error(f"❌ Customer ID '{customer_id}' not found!")
                else:
                    with st.spinner("🔮 Predicting channel preferences..."):
                        # Convert to DataFrame and predict
                        customer_df = pd.DataFrame([customer_data.iloc[0]])
                        predictions = predictor.predict_single_customer(customer_df)
                        
                        # Display results in two columns
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Display predictions
                            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                            st.subheader("🎯 Channel Preference Ranking")
                            st.write(f"**Customer:** {customer_id}")
                            
                            # Create a nice visualization of scores
                            for i, pred in enumerate(predictions, 1):
                                col_a, col_b, col_c = st.columns([1, 3, 1])
                                
                                with col_a:
                                    st.write(f"**#{i}** {pred['channel']}")
                                with col_b:
                                    st.progress(float(pred['score']))
                                with col_c:
                                    st.write(f"`{pred['score']:.4f}`")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Confidence analysis
                            confidence_gap = predictions[0]['score'] - predictions[1]['score']
                            if confidence_gap > 0.3:
                                confidence_level = "🟢 HIGH (Clear preference)"
                            elif confidence_gap > 0.15:
                                confidence_level = "🟡 MEDIUM (Strong preference)"
                            else:
                                confidence_level = "🔴 LOW (Consider multiple channels)"
                            
                            st.info(f"**Confidence Level:** {confidence_level}")
                            st.success(f"**🚀 RECOMMENDATION:** Start with **{predictions[0]['channel']}**")

                        with col2:
                            # Customer insights
                            insights = predictor.get_customer_insights(customer_id)
                            if insights:
                                st.markdown('<div class="insights-card">', unsafe_allow_html=True)
                                st.subheader("📈 Customer Profile")
                                st.metric("Age", insights['age'])
                                st.metric("Income Band", insights['income_band'])
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Quick stats
                            st.markdown("---")
                            st.subheader("📋 Quick Stats")
                            st.write(f"**Top Channel:** {predictions[0]['channel']}")
                            st.write(f"**Score:** {predictions[0]['score']:.4f}")
                            st.write(f"**Score Gap:** {confidence_gap:.4f}")
                        
            except Exception as e:
                st.error(f"Error predicting for customer {customer_id}: {str(e)}")
    
    with tab2:
        st.subheader("Batch Customer Prediction")
        
        # Two options for batch prediction
        option = st.radio("Choose batch prediction method:", 
                         ["Select Multiple Customers", "Process All Customers"])
        
        if option == "Select Multiple Customers":
            st.write("**Enter multiple Customer IDs (comma-separated):**")
            customer_ids_input = st.text_area(
                "",
                placeholder="SCB892183816, SCB892183817, SCB892183818",
                height=100,
                label_visibility="collapsed"
            )
            
            if customer_ids_input:
                customer_ids = [cid.strip() for cid in customer_ids_input.split(',') if cid.strip()]
                
                if st.button("🚀 Predict for Selected Customers", type="primary", use_container_width=True):
                    try:
                        with st.spinner(f"🔮 Predicting for {len(customer_ids)} customers..."):
                            predictions = predictor.predict_multiple_customers(customer_ids)
                            
                            if predictions:
                                st.success(f"✅ Predictions completed for {len(predictions)} customers")
                                
                                # Display summary metrics
                                st.subheader("📈 Prediction Summary")
                                col1, col2, col3 = st.columns(3)
                                
                                top_channels = [preds[0]['channel'] for preds in predictions.values()]
                                channel_counts = pd.Series(top_channels).value_counts()
                                
                                with col1:
                                    st.metric("Customers Processed", len(predictions))
                                with col2:
                                    most_common = channel_counts.index[0]
                                    st.metric("Most Preferred", most_common)
                                with col3:
                                    st.metric("Unique Preferences", len(channel_counts))
                                
                                # Display top results in expandable sections
                                st.subheader("🎯 Top Customer Recommendations")
                                for i, (customer_id, customer_predictions) in enumerate(list(predictions.items())[:15]):
                                    with st.expander(f"👤 {customer_id} | 🥇 {customer_predictions[0]['channel']} (Score: {customer_predictions[0]['score']:.4f})", expanded=i<3):
                                        col_left, col_right = st.columns([2, 1])
                                        
                                        with col_left:
                                            st.write("**Channel Rankings:**")
                                            for j, pred in enumerate(customer_predictions[:4], 1):
                                                st.write(f"{j}. **{pred['channel']}** - `{pred['score']:.4f}`")
                                        
                                        with col_right:
                                            insights = predictor.get_customer_insights(customer_id)
                                            if insights:
                                                st.write("**Customer Insights:**")
                                                st.write(f"**Age:** {insights['age']}")
                                                st.write(f"**Income:** {insights['income_band']}")
                                
                                # Download option
                                st.subheader("💾 Download Results")
                                all_predictions = []
                                for customer_id, customer_predictions in predictions.items():
                                    for i, pred in enumerate(customer_predictions, 1):
                                        pred_data = pred.copy()
                                        pred_data['rank'] = i
                                        all_predictions.append(pred_data)
                                
                                df_output = pd.DataFrame(all_predictions)
                                csv = df_output.to_csv(index=False)
                                
                                st.download_button(
                                    label="📥 Download All Predictions as CSV",
                                    data=csv,
                                    file_name="channel_predictions.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                    
                    except Exception as e:
                        st.error(f"Error in batch prediction: {str(e)}")
        
        else:  # Process All Customers
            st.warning("⚠️ This will process ALL customers and may take significant time for large datasets.")
            
            if st.button("🌐 Predict for All Customers", type="primary", use_container_width=True):
                try:
                    with st.spinner(f"🔮 Predicting for all {len(predictor.df)} customers..."):
                        predictions = predictor.predict_multiple_customers()
                        
                        if predictions:
                            st.success(f"✅ Predictions completed for {len(predictions)} customers")
                            
                            # Comprehensive summary
                            st.subheader("📊 Batch Prediction Summary")
                            
                            # Top metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            top_channels = [preds[0]['channel'] for preds in predictions.values()]
                            channel_counts = pd.Series(top_channels).value_counts()
                            total_score = sum([preds[0]['score'] for preds in predictions.values()])
                            avg_top_score = total_score / len(predictions)
                            
                            with col1:
                                st.metric("Total Customers", len(predictions))
                            with col2:
                                most_common_channel = channel_counts.index[0]
                                st.metric("Top Channel", most_common_channel)
                            with col3:
                                st.metric("Channel Diversity", len(channel_counts))
                            with col4:
                                st.metric("Avg Top Score", f"{avg_top_score:.4f}")
                            
                            # Channel distribution
                            st.subheader("📈 Channel Preference Distribution")
                            chart_data = pd.DataFrame({
                                'Channel': channel_counts.index,
                                'Count': channel_counts.values
                            })
                            st.bar_chart(chart_data.set_index('Channel'))
                            
                            # Sample of results
                            st.subheader("🎯 Sample Predictions (First 10 Customers)")
                            sample_df = pd.DataFrame([
                                {
                                    'Customer_ID': cust_id,
                                    'Top_Channel': preds[0]['channel'],
                                    'Top_Score': f"{preds[0]['score']:.4f}",
                                    'Second_Channel': preds[1]['channel'],
                                    'Score_Gap': f"{(preds[0]['score'] - preds[1]['score']):.4f}"
                                }
                                for cust_id, preds in list(predictions.items())[:10]
                            ])
                            st.dataframe(sample_df, use_container_width=True)
                            
                            # Download all predictions
                            st.subheader("💾 Download Complete Results")
                            all_predictions = []
                            for customer_id, customer_predictions in predictions.items():
                                for i, pred in enumerate(customer_predictions, 1):
                                    pred_data = pred.copy()
                                    pred_data['rank'] = i
                                    all_predictions.append(pred_data)
                            
                            df_output = pd.DataFrame(all_predictions)
                            csv = df_output.to_csv(index=False)
                            
                            st.download_button(
                                label="📥 Download Complete Dataset as CSV",
                                data=csv,
                                file_name="complete_channel_predictions.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                
                except Exception as e:
                    st.error(f"Error in batch processing: {str(e)}")

    # Footer
    st.markdown("---")
    st.caption("LightGBM Channel Preference Prediction System | Scores normalized to 0-1 range using softmax")

if __name__ == "__main__":
    main()