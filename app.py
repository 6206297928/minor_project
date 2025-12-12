import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Page Config
st.set_page_config(page_title="Student Performance Predictor", layout="wide")

st.title("🎓 Student Performance Prediction System")
st.markdown("Enter student details to predict their final **Grade**.")

# --- PART 1: LOAD & TRAIN (Cached) ---
@st.cache_resource
def load_and_train_model():

    try:
        df = pd.read_csv("masked_data.csv")
    except FileNotFoundError:
        return None, None, None, None, None

    # ========== FIX 1: ADD DROP COLUMNS ==========
    drop_cols = []   # ← add your columns to drop
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Fill missing
    df["Parent_Education_Level"] = df["Parent_Education_Level"].fillna("Bachelor's")

    # Encode Binary
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    df["Extracurricular_Activities"] = df["Extracurricular_Activities"].map({"No": 0, "Yes": 1})
    df["Internet_Access_at_Home"] = df["Internet_Access_at_Home"].map({"No": 0, "Yes": 1})

    # Encode Ordinal
    edu_map = {"High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}
    df["Parent_Education_Level"] = df["Parent_Education_Level"].map(edu_map)

    income_map = {"Low": 1, "Medium": 2, "High": 3}
    df["Family_Income_Level"] = df["Family_Income_Level"].map(income_map)

    # ========== FIX 2: ONE-HOT ENCODING ==========
    department_columns = ["Department"]  # ← UPDATE BASED ON YOUR DATASET
    df = pd.get_dummies(df, columns=department_columns, drop_first=True)

    # Outlier handling
    numeric_cols = [
        col for col in df.select_dtypes(include=["int64", "float64"]).columns
        if col != "Grade"
    ]
    for col in numeric_cols:
        low, high = df[col].quantile([0.05, 0.95])
        df[col] = df[col].clip(lower=low, upper=high)

    # Label Encoding
    encoder = LabelEncoder()
    df["Grade"] = encoder.fit_transform(df["Grade"])

    # Scaling
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    scale_cols = [c for c in numeric_cols if c not in bool_cols]

    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    # Train
    X = df.drop(columns="Grade", axis=1)
    Y = df["Grade"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, Y)

    return model, scaler, encoder, X.columns, scale_cols


# Train model
model, scaler, encoder, model_columns, scale_cols = load_and_train_model()

if model is None:
    st.error("Error: 'masked_data.csv' not found. Please upload the CSV file.")
    st.stop()


# --- PART 2: USER INPUT ---
st.sidebar.header("Student Parameters")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    dept = st.sidebar.selectbox("Department", ("Science", "Engineering", "Arts", "Commerce"))
    parent_edu = st.sidebar.selectbox("Parent Education", ("High School", "Bachelor's", "Master's", "PhD"))
    income = st.sidebar.selectbox("Family Income", ("Low", "Medium", "High"))
    extra = st.sidebar.selectbox("Extracurricular Activities", ("Yes", "No"))
    internet = st.sidebar.selectbox("Internet Access", ("Yes", "No"))

    attendance = st.sidebar.slider("Attendance (%)", 0, 100, 80)
    study_hours = st.sidebar.number_input("Study Hours/Week", 0, 168, 15)
    sleep_hours = st.sidebar.number_input("Sleep Hours/Night", 0, 24, 7)
    stress = st.sidebar.slider("Stress Level (1-10)", 1, 10, 5)

    st.sidebar.subheader("Exam Scores")
    midterm = st.sidebar.number_input("Midterm Score", 0, 100, 75)
    final = st.sidebar.number_input("Final Score", 0, 100, 75)
    assignments = st.sidebar.number_input("Assignments Avg", 0, 100, 80)
    projects = st.sidebar.number_input("Projects Score", 0, 100, 80)

    data = {
        "Gender": gender,
        "Department": dept,
        "Parent_Education_Level": parent_edu,
        "Family_Income_Level": income,
        "Extracurricular_Activities": extra,
        "Internet_Access_at_Home": internet,
        "Attendance (%)": attendance,
        "Study_Hours_per_Week": study_hours,
        "Sleep_Hours_per_Night": sleep_hours,
        "Stress_Level (1-10)": stress,
        "Midterm_Score": midterm,
        "Final_Score": final,
        "Assignments_Avg": assignments,
        "Projects_Score": projects
    }

    return pd.DataFrame([data])     # FIXED: Correct index


input_df = user_input_features()

st.subheader("Student Details")
st.dataframe(input_df)


# --- PART 3: PREDICTION ---
if st.button("Predict Grade"):

    # Maps
    input_df["Gender"] = input_df["Gender"].map({"Male": 0, "Female": 1})
    input_df["Extracurricular_Activities"] = input_df["Extracurricular_Activities"].map({"No": 0, "Yes": 1})
    input_df["Internet_Access_at_Home"] = input_df["Internet_Access_at_Home"].map({"No": 0, "Yes": 1})

    edu_map = {"High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}
    input_df["Parent_Education_Level"] = input_df["Parent_Education_Level"].map(edu_map)

    income_map = {"Low": 1, "Medium": 2, "High": 3}
    input_df["Family_Income_Level"] = input_df["Family_Income_Level"].map(income_map)

    # One-Hot Encode department
    input_df = pd.get_dummies(input_df, columns=["Department"], drop_first=True)

    # Align columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Scale
    input_df[scale_cols] = scaler.transform(input_df[scale_cols])

    # Predict
    pred_encoded = model.predict(input_df)
    grade = encoder.inverse_transform(pred_encoded)[0]

    # Output
    st.divider()
    if grade == "A":
        st.success(f"### Predicted Grade: {grade} 🌟")
        st.balloons()
    elif grade == "C":
        st.info(f"### Predicted Grade: {grade}")
    else:
        st.error(f"### Predicted Grade: {grade}")
        st.write("Consider recommending extra study hours or tutoring.")
