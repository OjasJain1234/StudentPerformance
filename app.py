import streamlit as st
import pandas as pd
import joblib, json

# Load trained artifacts
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
with open("learning_paths.json", "r") as f:
    learning_paths = json.load(f)

st.set_page_config(page_title="🎓 Personalized Learning Path Generator", layout="centered")

st.title("🎓 Personalized Learning Path Recommendation System")
st.markdown("##### Get a learning plan designed just for you based on your study habits and performance!")

# Input Form
with st.form("student_form"):
    st.subheader("Enter your academic details:")

    hours = st.number_input("Hours Studied per Day", min_value=0.0, step=0.5)
    prev = st.number_input("Previous Score (%)", min_value=0.0, max_value=100.0, step=1.0)
    extra = st.number_input("Extracurricular Activities (hrs/week)", min_value=0.0, step=1.0)
    sleep = st.number_input("Sleep Hours (per day)", min_value=0.0, max_value=24.0, step=0.5)
    papers = st.number_input("Sample Question Papers Practiced", min_value=0, step=1)
    submitted = st.form_submit_button("📊 Generate My Learning Path")

if submitted:
    try:
        # Prepare input data
        input_data = pd.DataFrame([[hours, prev, extra, sleep, papers]],
            columns=['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced'])
        scaled = scaler.transform(input_data)

        # Predict
        pred = model.predict(scaled)[0]
        path = learning_paths.get(pred, {})

        st.success(f"🎯 Based on your profile, your learning level is: **{pred.upper()}**")
        st.markdown(f"### 📘 Recommended Path: {path.get('title','')}")
        st.write("**Modules to Focus On:**")
        for m in path.get('modules', []):
            st.write(f"- {m}")
        st.write(f"**Recommended Study Hours:** {path.get('hours','')}")

        st.balloons()

    except Exception as e:
        st.error(f"Error: {e}")
