import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Clinical Risk System", page_icon="🩺")

st.title("🩺 AI Clinical Risk Assessment System")
st.write("Machine Learning powered early disease risk detection tool.")

# ---------------------------
# LOAD DATA
# ---------------------------
data = pd.read_csv("diabetes.csv")

# ---------------------------
# TRAIN MODEL
# ---------------------------
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

model = RandomForestClassifier()
model.fit(X, y)

# ---------------------------
# INPUT FORM
# ---------------------------
st.markdown("## 👤 Patient Input")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0)

with col2:
    insulin = st.number_input("Insulin", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=0)

# ---------------------------
# PREDICTION
# ---------------------------
if st.button("Generate Clinical Report"):

    input_data = [[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]]

    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1]

    # ---------------------------
    # RISK CLASSIFICATION (3 LEVELS)
    # ---------------------------
    if prob >= 0.7:
        risk_level = "🔴 HIGH RISK"
        risk_color = "error"

    elif prob >= 0.4:
        risk_level = "🟠 MODERATE RISK"
        risk_color = "warning"

    else:
        risk_level = "🟢 LOW RISK"
        risk_color = "success"

    # ---------------------------
    # REPORT HEADER
    # ---------------------------
    st.markdown("## 🧾 Clinical Risk Assessment Report")
    st.markdown("---")

    # ---------------------------
    # SUMMARY
    # ---------------------------
    st.markdown("### 👤 Patient Summary")
    st.write(f"Age: {age}")
    st.write(f"Glucose: {glucose}")
    st.write(f"Blood Pressure: {blood_pressure}")
    st.write(f"BMI: {bmi}")

    st.markdown("---")

    # ---------------------------
    # RESULT
    # ---------------------------
    st.markdown("### ⚕️ Diagnostic Assessment")

    st.write(f"**Risk Level:** {risk_level}")
    st.write(f"**Risk Probability:** {round(prob * 100, 2)}%")

    # ---------------------------
    # KEY FACTORS (simple explainability)
    # ---------------------------
    st.markdown("### 🧠 Key Clinical Indicators")

    if glucose > 140:
        st.write("⚠ Elevated glucose level (major risk factor)")
    if bmi > 30:
        st.write("⚠ High BMI (obesity risk)")
    if blood_pressure > 90:
        st.write("⚠ High blood pressure detected")
    if age > 50:
        st.write("⚠ Age-related risk factor present")

    st.markdown("---")

    # ---------------------------
    # RECOMMENDATION SYSTEM
    # ---------------------------
    st.markdown("### 💡 Medical Recommendation")

    if prob >= 0.7:
        st.error("""
🚨 HIGH RISK  
- Immediate physician consultation recommended  
- Further tests required (HbA1c, glucose tolerance test)  
- Lifestyle intervention needed
""")

    elif prob >= 0.4:
        st.warning("""
⚠ MODERATE RISK  
- Regular monitoring recommended  
- Dietary improvements advised  
- Follow-up screening suggested
""")

    else:
        st.success("""
✅ LOW RISK  
- No immediate concern  
- Maintain healthy lifestyle  
- Routine check-ups sufficient
""")