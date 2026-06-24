import streamlit as st
import pandas as pd
import numpy as np  # <-- NEW: Required for sorting probabilities
import joblib
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Flow Predictor", layout="wide")

# --- 2. CUSTOM HEADER ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Teko:wght@500;600&display=swap');
    </style>

    <div style='text-align: center; color: #FF6B00; font-family: "Teko", sans-serif; font-size: 4.5rem; font-weight: 600; letter-spacing: 2px; margin-bottom: -10px; line-height: 1;'>
        MULTIPHASE FLOW CORRELATION PREDICTOR
    </div>
    <div style='text-align: center; color: #A3A3A3; font-size: 18px; margin-bottom: 20px; font-family: sans-serif;'>
        Upload a dataset or manually enter parameters to instantly generate AI-driven predictions.
    </div>
    <hr style='border-color: #333333;'>
""", unsafe_allow_html=True)
# --- 3. MODEL LOADING ---
@st.cache_resource
def load_artifacts(model_type):
    current_dir = os.path.dirname(os.path.abspath(__file__))
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

# --- HELPER FUNCTION: RUN PREDICTION ON A SINGLE ROW ---
def predict_single_row_top3(input_dict, model_type):
    df = pd.DataFrame([input_dict])
    
    baseline_defaults = {
        'MD': 10000.0, 'TVD': 8000.0, 'Tubing_ID': 2.992, 'Deviation_Angle': 0.0 if model_type == 'vertical' else 90.0,
        'Oil_Rate': 500.0, 'Water_Rate': 100.0, 'Gas_Rate': 250.0, 'Water_Cut': 15.0,
        'GOR': 500.0, 'Oil_API': 35.0, 'Oil_Viscosity': 2.5, 'Gas_SG': 0.7,
        'WHP': 250.0, 'WHT': 140.0
    }
    
    df.fillna(value=baseline_defaults, inplace=True)
    
    df['Total_Liquid'] = df['Oil_Rate'] + df['Water_Rate']
    df['Gas_Liquid_Ratio'] = (df['Gas_Rate'] * 1000) / (df['Total_Liquid'] + 1)
    df['Tubing_Area'] = 3.14159 * (df['Tubing_ID'] / 2)**2
    df['Liquid_Velocity_Proxy'] = df['Total_Liquid'] / df['Tubing_Area']
    df['Gas_Velocity_Proxy'] = (df['Gas_Rate'] * 1000) / df['Tubing_Area']
    
    final_features = base_features + ['Total_Liquid', 'Gas_Liquid_Ratio', 'Liquid_Velocity_Proxy', 'Gas_Velocity_Proxy']
    X = df[final_features].copy()
    
    model, scaler, le = load_artifacts(model_type)
    
    # --- NEW: Get Probabilities instead of just the top prediction ---
    probs = model.predict_proba(scaler.transform(X))[0]
    
    # Sort to find the indices of the top 3 highest probabilities
    top3_idx = np.argsort(-probs)[:3]
    
    # Get the names and confidence scores of the top 3
    top3_classes = le.inverse_transform(top3_idx)
    top3_confidences = probs[top3_idx] * 100
    
    return top3_classes, top3_confidences


# --- 4. TABS LAYOUT ---
tab_vertical, tab_horizontal = st.tabs(["🛢️ Vertical (Well) Model", "🛤️ Horizontal (Pipeline) Model"])

# ==========================================
#           VERTICAL MODEL TAB
# ==========================================
with tab_vertical:
    model_type_v = "vertical"
    input_mode_v = st.radio("Choose Input Method:", ["📁 Upload Dataset", "✍️ Manual Entry"], horizontal=True, key="radio_v")
    st.markdown("<br>", unsafe_allow_html=True)
    
    if input_mode_v == "✍️ Manual Entry":
        with st.form("manual_form_v", clear_on_submit=False):
            st.subheader("Well Geometry")
            g1, g2, g3, g4 = st.columns(4)
            md = g1.number_input("MD (ft)", value=None, placeholder="e.g. 10000")
            tvd = g2.number_input("TVD (ft)", value=None, placeholder="e.g. 8000")
            tubing_id = g3.number_input("Tubing ID (in)", value=None, placeholder="e.g. 2.99")
            dev_angle = g4.number_input("Deviation Angle (°)", value=None, placeholder="e.g. 0")

            st.subheader("Production Rates")
            r1, r2, r3, r4 = st.columns(4)
            oil_rate = r1.number_input("Oil Rate (bopd)", value=None, placeholder="e.g. 500")
            water_rate = r2.number_input("Water Rate (bwpd)", value=None, placeholder="e.g. 100")
            gas_rate = r3.number_input("Gas Rate (mscfd)", value=None, placeholder="e.g. 250")
            water_cut = r4.number_input("Water Cut (%)", value=None, placeholder="e.g. 16.6")

            st.subheader("Fluid & Surface Properties")
            f1, f2, f3, f4, f5, f6 = st.columns(6)
            gor = f1.number_input("GOR (scf/stb)", value=None, placeholder="e.g. 500")
            oil_api = f2.number_input("Oil API", value=None, placeholder="e.g. 35")
            oil_visc = f3.number_input("Oil Viscosity (cP)", value=None, placeholder="e.g. 2.5")
            gas_sg = f4.number_input("Gas SG", value=None, placeholder="e.g. 0.7")
            whp = f5.number_input("WHP (psi)", value=None, placeholder="e.g. 250")
            wht = f6.number_input("WHT (°F)", value=None, placeholder="e.g. 140")

            submit_v = st.form_submit_button("🚀 Generate Prediction", type="primary", use_container_width=True)

        if submit_v:
            manual_data_v = {
                'MD': md, 'TVD': tvd, 'Tubing_ID': tubing_id, 'Deviation_Angle': dev_angle,
                'Oil_Rate': oil_rate, 'Water_Rate': water_rate, 'Gas_Rate': gas_rate, 'Water_Cut': water_cut,
                'GOR': gor, 'Oil_API': oil_api, 'Oil_Viscosity': oil_visc, 'Gas_SG': gas_sg, 'WHP': whp, 'WHT': wht
            }
            
            if all(v is None for v in manual_data_v.values()):
                st.warning("⚠️ You didn't enter any data! Please fill in at least one field to generate a prediction.")
            else:
                with st.spinner("Analyzing Parameters..."):
                    # Capture the new Top 3 outputs
                    top_classes, top_conf = predict_single_row_top3(manual_data_v, model_type_v)
                    
                    st.success("### 🎯 Recommended Vertical Correlations")
                    
                    # Display Top 3 elegantly using columns
                    c1, c2, c3 = st.columns(3)
                    c1.metric("🥇 1st Choice", top_classes[0], f"{top_conf[0]:.1f}% Confidence")
                    c2.metric("🥈 2nd Choice", top_classes[1], f"{top_conf[1]:.1f}% Confidence")
                    c3.metric("🥉 3rd Choice", top_classes[2], f"{top_conf[2]:.1f}% Confidence")
                    
                    missing_keys = [k for k, v in manual_data_v.items() if v is None]
                    if missing_keys:
                        st.info(f"💡 **Note:** You left some fields blank. The AI automatically assumed standard baseline values for: {', '.join(missing_keys)}")

    else:
        col1_v, col2_v = st.columns([1, 2])
        memory_tag_v = f"{model_type_v}_results"
        
        with col1_v:
            with st.container(border=True):
                st.markdown("### 📥 Data Input")
                uploaded_file_v = st.file_uploader("Upload Vertical dataset", type=['csv', 'xlsx', 'xls'], key="upload_v")
                
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
                            
                            # --- NEW: Process top 3 for the entire dataset ---
                            probs = model.predict_proba(scaler.transform(X))
                            top3_idx = np.argsort(-probs, axis=1)[:, :3]
                            
                            df_v['1st_Choice'] = le.inverse_transform(top3_idx[:, 0])
                            df_v['2nd_Choice'] = le.inverse_transform(top3_idx[:, 1])
                            df_v['3rd_Choice'] = le.inverse_transform(top3_idx[:, 2])
                            
                            st.session_state[memory_tag_v] = df_v

        with col2_v:
            with st.container(border=True):
                st.markdown("### 📊 Vertical Predictions Dashboard")
                if memory_tag_v in st.session_state:
                    saved_df_v = st.session_state[memory_tag_v]
                    m1, m2 = st.columns(2)
                    m1.metric("Data Points Processed", len(saved_df_v))
                    m2.metric("Top Recommended Overall", saved_df_v['1st_Choice'].mode()[0])
                    st.dataframe(saved_df_v, use_container_width=True)
                    csv_v = saved_df_v.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Vertical Predictions", data=csv_v, file_name='vertical_predictions.csv', mime='text/csv', key="dl_v")
                else:
                    st.write("Awaiting dataset upload. Your predictions will appear here.")


# ==========================================
#           HORIZONTAL MODEL TAB
# ==========================================
with tab_horizontal:
    model_type_h = "horizontal"
    input_mode_h = st.radio("Choose Input Method:", ["📁 Upload Dataset", "✍️ Manual Entry"], horizontal=True, key="radio_h")
    st.markdown("<br>", unsafe_allow_html=True)
    
    if input_mode_h == "✍️ Manual Entry":
        with st.form("manual_form_h", clear_on_submit=False):
            st.subheader("Pipeline / Well Geometry")
            g1, g2, g3, g4 = st.columns(4)
            md = g1.number_input("MD (ft)", value=None, placeholder="e.g. 15000", key="h_md")
            tvd = g2.number_input("TVD (ft)", value=None, placeholder="e.g. 8000", key="h_tvd")
            tubing_id = g3.number_input("Tubing ID (in)", value=None, placeholder="e.g. 2.99", key="h_id")
            dev_angle = g4.number_input("Deviation Angle (°)", value=None, placeholder="e.g. 90", key="h_dev") 

            st.subheader("Production Rates")
            r1, r2, r3, r4 = st.columns(4)
            oil_rate = r1.number_input("Oil Rate (bopd)", value=None, placeholder="e.g. 1200", key="h_oil")
            water_rate = r2.number_input("Water Rate (bwpd)", value=None, placeholder="e.g. 300", key="h_wat")
            gas_rate = r3.number_input("Gas Rate (mscfd)", value=None, placeholder="e.g. 1000", key="h_gas")
            water_cut = r4.number_input("Water Cut (%)", value=None, placeholder="e.g. 20", key="h_wc")

            st.subheader("Fluid & Surface Properties")
            f1, f2, f3, f4, f5, f6 = st.columns(6)
            gor = f1.number_input("GOR (scf/stb)", value=None, placeholder="e.g. 833", key="h_gor")
            oil_api = f2.number_input("Oil API", value=None, placeholder="e.g. 35", key="h_api")
            oil_visc = f3.number_input("Oil Viscosity (cP)", value=None, placeholder="e.g. 3.5", key="h_visc")
            gas_sg = f4.number_input("Gas SG", value=None, placeholder="e.g. 0.65", key="h_sg")
            whp = f5.number_input("WHP (psi)", value=None, placeholder="e.g. 450", key="h_whp")
            wht = f6.number_input("WHT (°F)", value=None, placeholder="e.g. 155", key="h_wht")

            submit_h = st.form_submit_button("🚀 Generate Prediction", type="primary", use_container_width=True)

        if submit_h:
            manual_data_h = {
                'MD': md, 'TVD': tvd, 'Tubing_ID': tubing_id, 'Deviation_Angle': dev_angle,
                'Oil_Rate': oil_rate, 'Water_Rate': water_rate, 'Gas_Rate': gas_rate, 'Water_Cut': water_cut,
                'GOR': gor, 'Oil_API': oil_api, 'Oil_Viscosity': oil_visc, 'Gas_SG': gas_sg, 'WHP': whp, 'WHT': wht
            }
            
            if all(v is None for v in manual_data_h.values()):
                st.warning("⚠️ You didn't enter any data! Please fill in at least one field to generate a prediction.")
            else:
                with st.spinner("Analyzing Parameters..."):
                    # Capture the new Top 3 outputs
                    top_classes, top_conf = predict_single_row_top3(manual_data_h, model_type_h)
                    
                    st.success("### 🎯 Recommended Horizontal Correlations")
                    
                    # Display Top 3 elegantly using columns
                    c1, c2, c3 = st.columns(3)
                    c1.metric("🥇 1st Choice", top_classes[0], f"{top_conf[0]:.1f}% Confidence")
                    c2.metric("🥈 2nd Choice", top_classes[1], f"{top_conf[1]:.1f}% Confidence")
                    c3.metric("🥉 3rd Choice", top_classes[2], f"{top_conf[2]:.1f}% Confidence")
                    
                    missing_keys = [k for k, v in manual_data_h.items() if v is None]
                    if missing_keys:
                        st.info(f"💡 **Note:** You left some fields blank. The AI automatically assumed standard baseline values for: {', '.join(missing_keys)}")

    else:
        col1_h, col2_h = st.columns([1, 2])
        memory_tag_h = f"{model_type_h}_results"
        
        with col1_h:
            with st.container(border=True):
                st.markdown("### 📥 Data Input")
                uploaded_file_h = st.file_uploader("Upload Horizontal dataset", type=['csv', 'xlsx', 'xls'], key="upload_h")
                
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
                            
                            # --- NEW: Process top 3 for the entire dataset ---
                            probs = model.predict_proba(scaler.transform(X))
                            top3_idx = np.argsort(-probs, axis=1)[:, :3]
                            
                            df_h['1st_Choice'] = le.inverse_transform(top3_idx[:, 0])
                            df_h['2nd_Choice'] = le.inverse_transform(top3_idx[:, 1])
                            df_h['3rd_Choice'] = le.inverse_transform(top3_idx[:, 2])
                            
                            st.session_state[memory_tag_h] = df_h

        with col2_h:
            with st.container(border=True):
                st.markdown("### 📊 Horizontal Predictions Dashboard")
                if memory_tag_h in st.session_state:
                    saved_df_h = st.session_state[memory_tag_h]
                    m1, m2 = st.columns(2)
                    m1.metric("Data Points Processed", len(saved_df_h))
                    m2.metric("Top Recommended Overall", saved_df_h['1st_Choice'].mode()[0])
                    st.dataframe(saved_df_h, use_container_width=True)
                    csv_h = saved_df_h.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Horizontal Predictions", data=csv_h, file_name='horizontal_predictions.csv', mime='text/csv', key="dl_h")
                else:
                    st.write("Awaiting dataset upload. Your predictions will appear here.")
