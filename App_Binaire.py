import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(page_title="Binary Maintenance (Stage 1)", page_icon="⚖️", layout="wide")

@st.cache_resource
def load_binary_models():
    rf = joblib.load("best_rf_model.pkl")
    xgb = joblib.load("best_xgb_model.pkl")
    lgbm = joblib.load("best_lgbm_model.pkl")
    return rf, xgb, lgbm

try:
    rf_bin, xgb_bin, lgbm_bin = load_binary_models()
except Exception as e:
    st.error(f"Erreur de chargement : {e}")
    st.stop()

# --- INTERFACE ---
st.title("⚖️ Binary Classification - Stage 1 Winner")

# --- SCÉNARIOS DE TEST (Sidebar) ---
st.sidebar.header("🚀 Quick Test Scenarios")

# Initialisation des valeurs par défaut
if 'val_type' not in st.session_state:
    st.session_state.val_type = 'L'
    st.session_state.val_air = 300.0
    st.session_state.val_proc = 310.0
    st.session_state.val_speed = 1500
    st.session_state.val_torque = 40.0
    st.session_state.val_wear = 0

# Bouton pour charger le cas "Normal"
if st.sidebar.button("✅ Load Normal Case"):
    st.session_state.val_type = 'L'
    st.session_state.val_air = 298.1
    st.session_state.val_proc = 308.6
    st.session_state.val_speed = 1551
    st.session_state.val_torque = 42.8
    st.session_state.val_wear = 0

# Bouton pour charger le cas "Panne (Overstrain)"
if st.sidebar.button("⚠️ Load Failure Case"):
    st.session_state.val_type = 'L'
    st.session_state.val_air = 301.0
    st.session_state.val_proc = 310.5
    st.session_state.val_speed = 1379
    st.session_state.val_torque = 57.3
    st.session_state.val_wear = 204

st.sidebar.divider()

# --- SAISIE MANUELLE ---
st.sidebar.header("📡 Manual Adjustments")
type_map = {'L': 0, 'M': 1, 'H': 2}
selected_type = st.sidebar.selectbox('Machine Type', options=['L', 'M', 'H'], index=['L', 'M', 'H'].index(st.session_state.val_type))

air = st.sidebar.number_input('Air temperature [K]', value=st.session_state.val_air)
proc = st.sidebar.number_input('Process temperature [K]', value=st.session_state.val_proc)
speed = st.sidebar.number_input('Rotational speed [rpm]', value=st.session_state.val_speed)
torque = st.sidebar.number_input('Torque [Nm]', value=st.session_state.val_torque)
wear = st.sidebar.number_input('Tool wear [min]', value=st.session_state.val_wear)

# --- PRÉPARATION DES DONNÉES ---
model_features = ['Type', 'Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min']
input_df = pd.DataFrame([[type_map[selected_type], air, proc, speed, torque, wear]], columns=model_features)

# --- AFFICHAGE ---
st.header("🏆 Performance Comparison")
col1, col2, col3 = st.columns(3)

def show_result(name, model, is_winner=False):
    try:
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        header = f"⭐ {name} (WINNER)" if is_winner else name
        if pred == 1:
            st.error(f"**{header}**")
            st.metric("DIAGNOSTIC", "FAILURE", delta=f"{proba:.1%} risk", delta_color="inverse")
        else:
            st.success(f"**{header}**")
            st.metric("DIAGNOSTIC", "NORMAL", delta=f"{proba:.1%} risk")
    except Exception as e:
        st.warning(f"Erreur avec {name}: {e}")

with col1: show_result("Random Forest", rf_bin)
with col2: show_result("XGBoost", xgb_bin)
with col3: show_result("LightGBM", lgbm_bin, is_winner=True)