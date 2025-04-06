import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import plotly.io as pio
from fpdf import FPDF
import base64
import sqlite3

from database import setup_tables
setup_tables()

# Load datasets
farmer_df = pd.read_csv('farmer_advisor_dataset.csv')
market_df = pd.read_csv('market_researcher_dataset.csv')

# Encode Crop Type
le_crop = LabelEncoder()
farmer_df['Crop_Type_Encoded'] = le_crop.fit_transform(farmer_df['Crop_Type'])

# Features and models
features = ['Soil_pH', 'Soil_Moisture', 'Temperature_C', 'Rainfall_mm',
            'Crop_Type_Encoded', 'Fertilizer_Usage_kg', 'Pesticide_Usage_kg']
X = farmer_df[features]
y_yield = farmer_df['Crop_Yield_ton']
y_sustain = farmer_df['Sustainability_Score']

model_yield = RandomForestRegressor().fit(X, y_yield)
model_sustain = RandomForestRegressor().fit(X, y_sustain)

# Market Score
market_df['Seasonal_Factor'] = market_df['Seasonal_Factor'].map({'Low': 0, 'Medium': 1, 'High': 2})
weights = {
    'Demand_Index': 0.2,
    'Supply_Index': 0.1,
    'Market_Price_per_ton': 0.2,
    'Competitor_Price_per_ton': 0.1,
    'Economic_Indicator': 0.1,
    'Weather_Impact_Score': 0.1,
    'Seasonal_Factor': 0.1,
    'Consumer_Trend_Index': 0.1
}
market_df['Profitability_Score'] = sum(
    market_df[col] * weight for col, weight in weights.items()
)
profitability_by_crop = market_df.groupby('Product')['Profitability_Score'].mean()

# PDF generator
def generate_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(34, 139, 34)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="AgroSense AI - Crop Recommendation Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    for key, value in data.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
    pdf.ln(10)
    try:
        pdf.image("chart.png", x=10, w=180)
    except:
        pdf.cell(200, 10, txt="(Chart image not found)", ln=True)
    pdf.set_y(-20)
    pdf.set_font("Arial", 'I', 10)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 10, "Agrisense Report - Created by Sumit and Nilesh", 0, 0, 'C')
    pdf_output = pdf.output(dest='S').encode('latin1')
    b64 = base64.b64encode(pdf_output).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="AgroSense_Report.pdf">Download Report as PDF</a>'
    return href

# Save to database
def save_to_db(name, crop, yield_ton, score):
    conn = sqlite3.connect("agrisense.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO farmer_history (name, crop, yield, sustainability_score)
        VALUES (?, ?, ?, ?)
    """, (name, crop, yield_ton, score))
    conn.commit()
    conn.close()

# View history from database
def view_history():
    conn = sqlite3.connect("agrisense.db")
    df = pd.read_sql_query("SELECT * FROM farmer_history", conn)
    conn.close()
    return df

# UI
st.set_page_config(page_title="AgroSense AI", layout="centered")
st.markdown("""
    <style>
    .title { font-size:32px !important; font-weight: bold; color: #2E8B57; }
    .subtitle { font-size:20px !important; color: #555; }
    </style>
""", unsafe_allow_html=True)
st.markdown('<div class="title">AgroSense AI - Smart Crop Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Empowering farmers with AI-driven insights</div>', unsafe_allow_html=True)
st.write("---")

# Inputs
farmer_name = st.text_input("Enter Farmer's Name")
crop_options = list(le_crop.classes_)
crop_type = st.selectbox("Select Crop Type", crop_options)
soil_ph = st.slider("Soil pH", 4.0, 9.0, 6.5)
soil_moisture = st.slider("Soil Moisture (%)", 5.0, 50.0, 30.0)
temperature = st.slider("Temperature (°C)", 10.0, 45.0, 25.0)
rainfall = st.slider("Rainfall (mm)", 0.0, 500.0, 100.0)
fertilizer = st.slider("Fertilizer Usage (kg)", 0.0, 300.0, 150.0)
pesticide = st.slider("Pesticide Usage (kg)", 0.0, 50.0, 10.0)

compare_mode = st.checkbox("Compare all crops before deciding?")
if compare_mode:
    st.subheader("Crop Profitability Comparison")
    market_avg = market_df.groupby('Product')['Profitability_Score'].mean().reset_index()
    fig2 = go.Figure(data=[go.Bar(x=market_avg['Product'], y=market_avg['Profitability_Score'], marker_color='darkorange')])
    fig2.update_layout(title='Average Profitability Score by Crop',
                       xaxis_title='Crop',
                       yaxis_title='Profitability Score',
                       xaxis_tickangle=-45)
    st.plotly_chart(fig2)

# Prediction & Recommendation
if st.button("Predict & Recommend"):
    crop_encoded = le_crop.transform([crop_type])[0]
    input_data = np.array([[soil_ph, soil_moisture, temperature, rainfall,
                            crop_encoded, fertilizer, pesticide]])
    pred_yield = model_yield.predict(input_data)[0]
    pred_sustain = model_sustain.predict(input_data)[0]
    market_score = profitability_by_crop.get(crop_type, 0)
    confidence = "High" if pred_sustain > 75 and market_score > 100 else "Moderate"

    st.subheader("Recommendation Summary:")
    st.write(f"**Crop:** {crop_type}")
    st.write(f"**Predicted Yield:** {pred_yield:.2f} tons")
    st.write(f"**Sustainability Score:** {pred_sustain:.2f}")
    st.write(f"**Profitability Score:** {market_score:.2f}")
    st.write(f"**Decision Confidence:** {confidence}")

    # Save to DB
    save_to_db(farmer_name, crop_type, pred_yield, pred_sustain)

    # Chart
    fig = go.Figure(data=[
        go.Bar(name='Predicted Yield (tons)', x=['Crop'], y=[pred_yield], marker_color='green'),
        go.Bar(name='Sustainability Score', x=['Crop'], y=[pred_sustain], marker_color='blue'),
        go.Bar(name='Profitability Score', x=['Crop'], y=[market_score], marker_color='orange')
    ])
    fig.update_layout(barmode='group', title='Crop Analysis Summary', yaxis=dict(title='Score / Yield'))
    st.subheader("Visual Summary")
    st.plotly_chart(fig)

    # Save chart
    pio.write_image(fig, "chart.png", format='png')

    # PDF Report
    st.subheader("Downloadable Report")
    report_data = {
        "Farmer Name": farmer_name if farmer_name else "N/A",
        "Crop Recommended": crop_type,
        "Predicted Yield (tons)": f"{pred_yield:.2f}",
        "Sustainability Score": f"{pred_sustain:.2f}",
        "Profitability Score": f"{market_score:.2f}",
        "Decision Confidence": confidence
    }
    pdf_link = generate_pdf(report_data)
    st.markdown(pdf_link, unsafe_allow_html=True)

    # Show History
    with st.expander("📜 View Past Recommendations"):
        history_df = view_history()
        if not history_df.empty:
            st.dataframe(history_df)
        else:
            st.write("No records found yet.")

# Footer
st.markdown("""
<hr>
<div style='text-align: center; font-size: 14px;'>
    Agrisense - Created by Sumit and Nilesh
</div>
""", unsafe_allow_html=True)
