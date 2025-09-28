import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime

# Optional imports - install if needed: pip install plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Configuration
MODEL_PATH = os.path.join("model", "logistic_model.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")

# Function to encode image for display
def get_base64_image(image_path):
    try:
        import base64
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# Load model and scaler with error handling
@st.cache_resource
def load_models():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure the model is properly trained and saved.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

model, scaler = load_models()

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with dark mode support and animations
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary-color: #ff6b6b;
    --secondary-color: #4ecdc4;
    --accent-color: #45b7d1;
    --success-color: #96ceb4;
    --warning-color: #ffeaa7;
    --danger-color: #fab1a0;
    --bg-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --glass-bg: rgba(255, 255, 255, 0.1);
    --glass-border: rgba(255, 255, 255, 0.2);
    --text-color: #2d3436;
    --shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
}

[data-theme="dark"] {
    --bg-gradient: linear-gradient(135deg, #232526 0%, #414345 100%);
    --glass-bg: rgba(0, 0, 0, 0.2);
    --glass-border: rgba(255, 255, 255, 0.1);
    --text-color: #ddd;
}

body {
    background: var(--bg-gradient);
    font-family: 'Inter', sans-serif;
    color: var(--text-color);
}

.main-container {
    background: var(--glass-bg);
    backdrop-filter: blur(15px);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem;
    box-shadow: var(--shadow);
    animation: slideIn 0.6s ease-out;
}

@keyframes slideIn {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}

.header-container {
    text-align: center;
    padding: 2rem 0;
    background: var(--glass-bg);
    border-radius: 15px;
    margin-bottom: 2rem;
    backdrop-filter: blur(10px);
}

.title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(45deg, var(--primary-color), var(--accent-color));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.subtitle {
    font-size: 1.2rem;
    color: var(--text-color);
    opacity: 0.8;
    margin-top: 0.5rem;
}

.input-section {
    background: var(--glass-bg);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    backdrop-filter: blur(10px);
    border: 1px solid var(--glass-border);
    transition: all 0.3s ease;
}

.input-section:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.15);
}

.risk-gauge {
    background: var(--glass-bg);
    border-radius: 15px;
    padding: 2rem;
    text-align: center;
    backdrop-filter: blur(15px);
    border: 1px solid var(--glass-border);
    margin: 2rem 0;
}

.metric-card {
    background: var(--glass-bg);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem;
    backdrop-filter: blur(10px);
    border: 1px solid var(--glass-border);
    text-align: center;
    transition: transform 0.3s ease;
}

.metric-card:hover {
    transform: scale(1.05);
}

.stButton > button {
    background: linear-gradient(45deg, var(--primary-color), var(--accent-color));
    border: none;
    border-radius: 10px;
    color: white;
    font-weight: 600;
    padding: 0.75rem 2rem;
    font-size: 1.1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
}

.info-card {
    background: var(--glass-bg);
    border-left: 4px solid var(--accent-color);
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    backdrop-filter: blur(10px);
}

.disclaimer {
    background: var(--glass-bg);
    border-radius: 10px;
    padding: 1rem;
    margin-top: 2rem;
    backdrop-filter: blur(10px);
    border: 1px solid var(--warning-color);
    color: #856404;
}
</style>
""", unsafe_allow_html=True)

# Sidebar for additional information
with st.sidebar:
    st.markdown("### 🔍 Risk Factor Information")
    
    risk_factors = {
        "Age": "Risk increases with age, especially after 45 for men and 55 for women",
        "Smoking": "Doubles the risk of heart disease",
        "Cholesterol": "High levels (>240 mg/dL) significantly increase risk",
        "Blood Pressure": "High BP (>140/90) is a major risk factor",
        "Diabetes": "Increases risk by 2-4 times",
        "BMI": "Obesity (BMI >30) increases risk substantially"
    }
    
    for factor, description in risk_factors.items():
        with st.expander(f"📊 {factor}"):
            st.write(description)
    
    st.markdown("---")
    st.markdown("### 📈 Prediction History")
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    if st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df[['timestamp', 'risk_percentage']].tail(5))
    else:
        st.write("No predictions made yet.")

# Main header
heart_icon_path = r"C:\Users\eshra\Downloads\11424092.png"  # Update this path to your image
heart_icon_base64 = get_base64_image(heart_icon_path)

if heart_icon_base64:
    st.markdown(f"""
    <div class="header-container">
        <img src="data:image/png;base64,{heart_icon_base64}" width="60" style="vertical-align:middle; margin-right: 15px;">
        <h1 class="title" style="display: inline;">Heart Disease Risk Assessment</h1>
        <p class="subtitle">AI-Powered 10-Year Cardiovascular Risk Prediction</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="header-container">
        <h1 class="title">❤️ Heart Disease Risk Assessment</h1>
        <p class="subtitle">AI-Powered 10-Year Cardiovascular Risk Prediction</p>
    </div>
    """, unsafe_allow_html=True)

# Health motivation section with glassmorphism
st.markdown("""
<div style="
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 20px;
    padding: 2rem;
    margin: 2rem 0;
    text-align: center;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    animation: slideIn 0.8s ease-out;
">
    <h3 style="color: #2d3436; margin-bottom: 1.5rem; font-weight: 600; font-size: 1.8rem;">
        💪 Your Heart, Your Health, Your Future
    </h3>
    <p style="font-size: 1.1rem; color: #636e72; line-height: 1.6; margin-bottom: 2rem; font-weight: 400;">
        Every heartbeat is a reminder of life's precious gift. Taking charge of your cardiovascular health today 
        means more quality moments with loved ones tomorrow.
    </p>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; margin-top: 2rem;">
        <div style="
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        " onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 12px 40px rgba(0,0,0,0.15)'" 
           onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none'">
            <span style="font-size: 3rem; display: block; margin-bottom: 1rem;">🏃‍♂️</span>
            <p style="margin: 0.5rem 0; font-weight: 600; color: #2d3436; font-size: 1.1rem;">Stay Active</p>
            <small style="color: #636e72; line-height: 1.4;">30 minutes of daily exercise keeps your heart strong</small>
        </div>
        <div style="
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        " onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 12px 40px rgba(0,0,0,0.15)'" 
           onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none'">
            <span style="font-size: 3rem; display: block; margin-bottom: 1rem;">🥗</span>
            <p style="margin: 0.5rem 0; font-weight: 600; color: #2d3436; font-size: 1.1rem;">Eat Smart</p>
            <small style="color: #636e72; line-height: 1.4;">Heart-healthy nutrition fuels your body</small>
        </div>
        <div style="
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        " onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 12px 40px rgba(0,0,0,0.15)'" 
           onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none'">
            <span style="font-size: 3rem; display: block; margin-bottom: 1rem;">😌</span>
            <p style="margin: 0.5rem 0; font-weight: 600; color: #2d3436; font-size: 1.1rem;">manage Stress</p>
            <small style="color: #636e72; line-height: 1.4;">Mental wellness protects your heart</small>
        </div>
        <div style="
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        " onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 12px 40px rgba(0,0,0,0.15)'" 
           onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none'">
            <span style="font-size: 3rem; display: block; margin-bottom: 1rem;">🩺</span>
            <p style="margin: 0.5rem 0; font-weight: 600; color: #2d3436; font-size: 1.1rem;">Regular Checkups</p>
            <small style="color: #636e72; line-height: 1.4;">Prevention is the best medicine</small>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.form(key="enhanced_prediction_form"):
    st.markdown("### 📝 Patient Information")
    
    # Basic Demographics
    st.markdown("#### 👤 Demographics")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=50, 
                            help="Age is a significant risk factor for heart disease")
        male = st.selectbox("Gender", ["Male", "Female"], 
                          help="Men typically have higher risk at younger ages")
    
    with col2:
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=300, value=70)
    
    # Calculate BMI automatically
    BMI = weight / ((height/100) ** 2)
    st.info(f"Calculated BMI: {BMI:.1f} kg/m²")
    
    # Lifestyle Factors
    st.markdown("#### 🚬 Lifestyle Factors")
    col1, col2 = st.columns(2)
    
    with col1:
        currentSmoker = st.selectbox("Current Smoker?", ["No", "Yes"])
        
        # Always show the input field, but set constraints based on smoking status
        if currentSmoker == "Yes":
            cigsPerDay = st.number_input("Cigarettes per day", 
                                       min_value=0, 
                                       max_value=100, 
                                       value=1,
                                       key="cigs_smoker",
                                       help="Average number of cigarettes smoked per day")
        else:
            cigsPerDay = st.number_input("Cigarettes per day", 
                                       min_value=0, 
                                       max_value=100, 
                                       value=0,
                                       key="cigs_non_smoker",
                                       help="Set to 0 for non-smokers")
    
    with col2:
        exercise_frequency = st.selectbox("Exercise Frequency", 
                                        ["Sedentary", "Light (1-2x/week)", "Moderate (3-4x/week)", "Active (5+x/week)"])
        family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
    
    # Medical Parameters
    st.markdown("#### 🩺 Medical Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        totChol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=400, value=200,
                                help="Normal: <200, Borderline: 200-239, High: ≥240")
        sysBP = st.number_input("Systolic BP (mm Hg)", min_value=80, max_value=250, value=120,
                              help="Normal: <120, Elevated: 120-129, High: ≥130")
        diaBP = st.number_input("Diastolic BP (mm Hg)", min_value=60, max_value=200, value=80,
                              help="Normal: <80, High: ≥80")
    
    with col2:
        heartRate = st.number_input("Resting Heart Rate (bpm)", min_value=40, max_value=200, value=70,
                                  help="Normal: 60-100 bpm")
        glucose = st.number_input("Fasting Glucose (mg/dL)", min_value=50, max_value=400, value=90,
                                help="Normal: <100, Prediabetes: 100-125, Diabetes: ≥126")
    
    # Medical Conditions
    st.markdown("#### 🏥 Medical Conditions")
    col1, col2 = st.columns(2)
    
    with col1:
        diabetes = st.selectbox("Diabetes?", ["No", "Yes"])
        prevalentHyp = st.selectbox("Hypertension?", ["No", "Yes"])
    
    with col2:
        medications = st.multiselect("Current Medications", 
                                   ["Blood Pressure Medication", "Cholesterol Medication", 
                                    "Blood Thinners", "Diabetes Medication", "Other"])
    
    # Submit button with loading state
    submit_button = st.form_submit_button("🔍 Analyze Risk", use_container_width=True)

# Enhanced prediction and results
if submit_button:
    # Map inputs to numeric values
    male_val = 1 if male == "Male" else 0
    currentSmoker_val = 1 if currentSmoker == "Yes" else 0
    diabetes_val = 1 if diabetes == "Yes" else 0
    prevalentHyp_val = 1 if prevalentHyp == "Yes" else 0
    
    # Create input array
    X = np.array([[age, male_val, currentSmoker_val, cigsPerDay, totChol,
                   sysBP, diaBP, BMI, heartRate, glucose,
                   diabetes_val, prevalentHyp_val]])
    
    # Make prediction
    with st.spinner("Analyzing patient data..."):
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0][1] * 100
    
    # Store in history
    st.session_state.prediction_history.append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'age': age,
        'risk_percentage': f"{prob:.1f}%",
        'risk_level': 'High' if prob > 50 else 'Moderate' if prob > 20 else 'Low'
    })
    
    # Results display
    st.markdown("### 📊 Risk Assessment Results")
    
    # Risk gauge visualization
    if PLOTLY_AVAILABLE:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = prob,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "10-Year Heart Disease Risk"},
            delta = {'reference': 30},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 20], 'color': "lightgreen"},
                    {'range': [20, 50], 'color': "yellow"},
                    {'range': [50, 80], 'color': "orange"},
                    {'range': [80, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "darkblue"},
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Fallback progress bar visualization
        st.markdown(f"""
        <div style="background-color: rgba(255,255,255,0.25); border-radius:15px; padding:20px; backdrop-filter: blur(10px);">
            <h3 style="text-align:center; margin-bottom:20px;">10-Year Heart Disease Risk: {prob:.1f}%</h3>
            <div style="
                width:100%;
                background-color:#e0e0e0;
                border-radius:10px;
                height:30px;
                position:relative;">
                <div style="
                    width:{prob}%;
                    background: {'linear-gradient(to right, #ff6b6b, #fab1a0)' if prob > 50 else 'linear-gradient(to right, #ffeaa7, #fdcb6e)' if prob > 20 else 'linear-gradient(to right, #00b894, #55a3ff)'};
                    height:30px;
                    border-radius:10px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    color:white;
                    font-weight:bold;">
                    {prob:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk interpretation
    if prob < 20:
        risk_level = "Low Risk"
        risk_color = "#96ceb4"
        risk_icon = "✅"
        recommendations = [
            "Continue maintaining healthy lifestyle habits",
            "Regular exercise and balanced diet",
            "Annual health check-ups"
        ]
    elif prob < 50:
        risk_level = "Moderate Risk"
        risk_color = "#ffeaa7"
        risk_icon = "⚠️"
        recommendations = [
            "Consider lifestyle modifications",
            "Monitor blood pressure and cholesterol regularly",
            "Discuss prevention strategies with your doctor"
        ]
    else:
        risk_level = "High Risk"
        risk_color = "#fab1a0"
        risk_icon = "🚨"
        recommendations = [
            "Immediate medical consultation recommended",
            "Consider medication if advised by physician",
            "Urgent lifestyle changes needed"
        ]
    
    # Display results with enhanced styling
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid {risk_color};">
            <h3>{risk_icon} {risk_level}</h3>
            <h2 style="color: {risk_color};">{prob:.1f}%</h2>
            <p>10-Year Risk</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        relative_risk = "High" if prob > np.mean([20, 50]) else "Average"
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Relative Risk</h3>
            <h2>{relative_risk}</h2>
            <p>Compared to population</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        confidence = "High" if abs(prob - 50) > 20 else "Moderate"
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Confidence</h3>
            <h2>{confidence}</h2>
            <p>Prediction reliability</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("### 💡 Personalized Recommendations")
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"**{i}.** {rec}")
    
    # Risk factors analysis
    st.markdown("### 📈 Risk Factor Analysis")
    
    risk_factors_data = {
        'Factor': ['Age', 'Gender', 'Smoking', 'Cholesterol', 'Blood Pressure', 'BMI', 'Diabetes'],
        'Your Value': [age, male, currentSmoker, totChol, f"{sysBP}/{diaBP}", f"{BMI:.1f}", diabetes],
        'Risk Level': [
            'High' if age > 65 else 'Moderate' if age > 45 else 'Low',
            'Higher' if male == 'Male' else 'Lower',
            'High' if currentSmoker == 'Yes' else 'Low',
            'High' if totChol > 240 else 'Moderate' if totChol > 200 else 'Low',
            'High' if sysBP > 140 or diaBP > 90 else 'Moderate' if sysBP > 120 else 'Low',
            'High' if BMI > 30 else 'Moderate' if BMI > 25 else 'Low',
            'High' if diabetes == 'Yes' else 'Low'
        ]
    }
    
    df_factors = pd.DataFrame(risk_factors_data)
    st.dataframe(df_factors, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("*Heart Disease Risk Assessment Tool - Powered by Machine Learning*")