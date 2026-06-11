import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Flow Predictor", layout="wide")

# --- 2. CUSTOM HEADER ---
st.markdown("""
    <h1 style='text-align: center; color: #F39C12; font-family: sans-serif;'>
        Multiphase Flow Correlation Predictor
    </h1>
    <p style='text-align: center; color: gray; font-size: 18px;'>
        Upload your well or pipeline data to instantly generate AI-driven predictions.
    </p>
    <hr>
""", unsafe_allow_html=True)

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_artifacts(model_type):
    # Dynamically find the folder where app.py is currently located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Safely attach the 'models' folder to that path
    path = os.path.join(current_dir, "Models") + "/"
    
    model = joblib.load(f"{path}xgb_model_{model_type}.pkl")
    scaler = joblib.load(f"{path}scaler_{model_type}.pkl")
    le = joblib.load(f"{path}le_{model_type}.pkl")
    return model, scaler, le

base_features = [
    'MD', 'TVD', 'Tubing_ID', 'Deviation_Angle',
    'Oil_Rate', 'Water_Rate', 'Gas_Rate', 'Water_Cut', 'GOR',
    'Oil_API', 'Oil_Viscosity', 'Gas_SG', 'WHP', 'WHT'
]

# --- 4. TABS LAYOUT ---
tab_vertical, tab_horizontal = st.tabs(["🛢️ Vertical (Well) Model", "🛤️ Horizontal (Pipeline) Model"])

# ==========================================
#           VERTICAL MODEL TAB
# ==========================================
with tab_vertical:
    col1_v, col2_v = st.columns([1, 2])
    model_type_v = "vertical"
    memory_tag_v = f"{model_type_v}_results"
    
    with col1_v:
        with st.container(border=True):
            st.markdown("### 📥 Data Input")
            st.info("Upload your Vertical Well parameters.")
            
            with st.expander("ℹ️ View Required Columns"):
                st.code(", ".join(base_features))
                
            uploaded_file_v = st.file_uploader("Upload dataset", type=['csv', 'xlsx', 'xls'], key="upload_v")
            
            if uploaded_file_v is not None:
                try:
                    if uploaded_file_v.name.endswith('.csv'):
                        df_v = pd.read_csv(uploaded_file_v)
                    elif uploaded_file_v.name.endswith('.xlsx'):
                        df_v = pd.read_excel(uploaded_file_v, engine='openpyxl')
                    elif uploaded_file_v.name.endswith('.xls'):
                        df_v = pd.read_excel(uploaded_file_v, engine='xlrd')
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    st.stop()
                    
                missing_cols = [col for col in base_features if col not in df_v.columns]
                if missing_cols:
                    st.error(f"Missing columns: {', '.join(missing_cols)}")
                else:
                    st.success("✅ Validation Passed!")
                    with st.spinner("Running AI Model..."):
                        model, scaler, le = load_artifacts(model_type_v)
                        processing_df = df_v.copy()
                        
                        processing_df['Total_Liquid'] = processing_df['Oil_Rate'] + processing_df['Water_Rate']
                        processing_df['Gas_Liquid_Ratio'] = (processing_df['Gas_Rate'] * 1000) / (processing_df['Total_Liquid'] + 1)
                        processing_df['Tubing_Area'] = 3.14159 * (processing_df['Tubing_ID'] / 2)**2
                        processing_df['Liquid_Velocity_Proxy'] = processing_df['Total_Liquid'] / processing_df['Tubing_Area']
                        processing_df['Gas_Velocity_Proxy'] = (processing_df['Gas_Rate'] * 1000) / processing_df['Tubing_Area']
                        
                        final_features = base_features + ['Total_Liquid', 'Gas_Liquid_Ratio', 'Liquid_Velocity_Proxy', 'Gas_Velocity_Proxy']
                        X = processing_df[final_features].copy()
                        X.fillna(X.median(numeric_only=True), inplace=True)
                        
                        predictions = model.predict(scaler.transform(X))
                        df_v['Recommended_Correlation'] = le.inverse_transform(predictions)
                        st.session_state[memory_tag_v] = df_v

    with col2_v:
        with st.container(border=True):
            st.markdown("### 📊 Vertical Predictions Dashboard")
            if memory_tag_v in st.session_state:
                saved_df_v = st.session_state[memory_tag_v]
                
                m1, m2 = st.columns(2)
                m1.metric("Data Points Processed", len(saved_df_v))
                m2.metric("Top Recommendation", saved_df_v['Recommended_Correlation'].mode()[0])
                
                st.dataframe(saved_df_v, use_container_width=True)
                
                csv_v = saved_df_v.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Vertical Predictions", data=csv_v, file_name='vertical_predictions.csv', mime='text/csv', key="dl_v")
            else:
                st.write("Awaiting dataset upload. Your predictions will appear here.")


# ==========================================
#           HORIZONTAL MODEL TAB
# ==========================================
with tab_horizontal:
    col1_h, col2_h = st.columns([1, 2])
    model_type_h = "horizontal"
    memory_tag_h = f"{model_type_h}_results"
    
    with col1_h:
        with st.container(border=True):
            st.markdown("### 📥 Data Input")
            st.info("Upload your Horizontal Pipeline parameters.")
            
            with st.expander("ℹ️ View Required Columns"):
                st.code(", ".join(base_features))
                
            uploaded_file_h = st.file_uploader("Upload dataset", type=['csv', 'xlsx', 'xls'], key="upload_h")
            
            if uploaded_file_h is not None:
                try:
                    if uploaded_file_h.name.endswith('.csv'):
                        df_h = pd.read_csv(uploaded_file_h)
                    elif uploaded_file_h.name.endswith('.xlsx'):
                        df_h = pd.read_excel(uploaded_file_h, engine='openpyxl')
                    elif uploaded_file_h.name.endswith('.xls'):
                        df_h = pd.read_excel(uploaded_file_h, engine='xlrd')
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    st.stop()
                    
                missing_cols = [col for col in base_features if col not in df_h.columns]
                if missing_cols:
                    st.error(f"Missing columns: {', '.join(missing_cols)}")
                else:
                    st.success("✅ Validation Passed!")
                    with st.spinner("Running AI Model..."):
                        model, scaler, le = load_artifacts(model_type_h)
                        processing_df = df_h.copy()
                        
                        processing_df['Total_Liquid'] = processing_df['Oil_Rate'] + processing_df['Water_Rate']
                        processing_df['Gas_Liquid_Ratio'] = (processing_df['Gas_Rate'] * 1000) / (processing_df['Total_Liquid'] + 1)
                        processing_df['Tubing_Area'] = 3.14159 * (processing_df['Tubing_ID'] / 2)**2
                        processing_df['Liquid_Velocity_Proxy'] = processing_df['Total_Liquid'] / processing_df['Tubing_Area']
                        processing_df['Gas_Velocity_Proxy'] = (processing_df['Gas_Rate'] * 1000) / processing_df['Tubing_Area']
                        
                        final_features = base_features + ['Total_Liquid', 'Gas_Liquid_Ratio', 'Liquid_Velocity_Proxy', 'Gas_Velocity_Proxy']
                        X = processing_df[final_features].copy()
                        X.fillna(X.median(numeric_only=True), inplace=True)
                        
                        predictions = model.predict(scaler.transform(X))
                        df_h['Recommended_Correlation'] = le.inverse_transform(predictions)
                        st.session_state[memory_tag_h] = df_h

    with col2_h:
        with st.container(border=True):
            st.markdown("### 📊 Horizontal Predictions Dashboard")
            if memory_tag_h in st.session_state:
                saved_df_h = st.session_state[memory_tag_h]
                
                m1, m2 = st.columns(2)
                m1.metric("Data Points Processed", len(saved_df_h))
                m2.metric("Top Recommendation", saved_df_h['Recommended_Correlation'].mode()[0])
                
                st.dataframe(saved_df_h, use_container_width=True)
                
                csv_h = saved_df_h.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Horizontal Predictions", data=csv_h, file_name='horizontal_predictions.csv', mime='text/csv', key="dl_h")
            else:
                st.write("Awaiting dataset upload. Your predictions will appear here.")
