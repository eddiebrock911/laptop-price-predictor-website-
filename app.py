import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ----------------- Load Model and Data -----------------
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# 🛠 Fix: Agar XGBModel ke andar gpu_id attribute hai to usse hata do
try:
    if hasattr(pipe.named_steps['xgbmodel'], 'gpu_id'):
        pipe.named_steps['xgbmodel'].gpu_id = None
except Exception:
    pass

# ----------------- Page Configuration -----------------
st.set_page_config(
    page_title="Laptop Price Predictor 💻",
    page_icon="💻",
    layout="centered"
)

# ----------------- Custom CSS -----------------
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #ece9e6, #ffffff);
        }
        .main {
            background-color: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.12);
        }
        h1, h2, h3 {
            color: #2e86de;
            font-family: 'Trebuchet MS', sans-serif;
        }
        .stButton button {
            background: linear-gradient(90deg, #2e86de, #48c6ef);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6em 1.2em;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton button:hover {
            background: linear-gradient(90deg, #48c6ef, #2e86de);
            transform: scale(1.05);
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ----------------- Title -----------------
st.markdown("<h1 style='text-align: center;'>💻 Laptop Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Get instant price predictions based on laptop specs</p>", unsafe_allow_html=True)
st.markdown("---")

st.subheader("📊 Enter Laptop Specifications")

# ----------------- Layout -----------------
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox('Brand', sorted(df['Company'].unique()))
    type_name = st.selectbox('Type', sorted(df['TypeName'].unique()))
    ram = st.selectbox('RAM (in GB)', sorted(df['Ram'].unique()))
    weight = st.number_input('Weight (in Kg)', min_value=0.5, max_value=5.0, step=0.1)
    touchscreen = st.radio('Touchscreen', ['No', 'Yes'], horizontal=True)
    ips_val = st.radio('IPS Display', ['No', 'Yes'], horizontal=True)

with col2:
    screen_size = st.number_input('Screen Size (inches)', min_value=10.0, max_value=20.0, step=0.1)
    resolution = st.selectbox(
        'Screen Resolution',
        ['1920x1080', '1366x768', '1600x900', '3840x2160',
         '2560x1600', '2736x1824', '2560x1440']
    )
    cpu = st.selectbox('CPU Brand', sorted(df['Cpu brand'].unique()))
    hdd = st.selectbox('HDD (in GB)', sorted(df['HDD'].unique()))
    ssd = st.selectbox('SSD (in GB)', sorted(df['SSD'].unique()))
    gpu = st.selectbox('GPU Brand', sorted(df['Gpu brand'].unique()))
    os = st.selectbox('Operating System', sorted(df['os'].unique()))

# ----------------- Prediction -----------------
if st.button('💰 Predict Price'):
    # Convert Yes/No to binary
    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips_val == 'Yes' else 0

    # Calculate PPI
    try:
        X_res, Y_res = map(int, resolution.split('x'))
        ppi = ((X_res ** 2 + Y_res ** 2) ** 0.5) / screen_size
    except Exception:
        st.error("⚠️ Error in calculating PPI. Please check resolution and screen size.")
        ppi = 0

    # Prepare input DataFrame
    input_df = pd.DataFrame([{
        'Company': brand,
        'TypeName': type_name,
        'Ram': ram,
        'Weight': weight,
        'Touchscreen': touchscreen_val,
        'IPS': ips_val,
        'ppi': ppi,
        'Cpu brand': cpu,
        'HDD': hdd,
        'SSD': ssd,
        'Gpu brand': gpu,
        'os': os
    }])

    # Predict
    try:
        predicted_price = int(np.exp(pipe.predict(input_df)[0]))
        low = int(predicted_price * 0.9)
        high = int(predicted_price * 1.1)

        st.success(f"💸 Estimated Laptop Price Range: ₹ {low:,} - ₹ {high:,}")
        st.balloons()  # 🎉 Stylish animation effect

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# ----------------- Footer -----------------
st.markdown("---")
st.markdown("<h5 style='text-align: center;'>🚀 Made with ❤️ by Ankit</h5>", unsafe_allow_html=True)
