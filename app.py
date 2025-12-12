# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Student Performance Dashboard w/ SHAP", layout="wide")
st.title("🎓 Student Performance Prediction, Analysis & SHAP Explanations")
st.markdown("Predict grade, view interpretable charts, and see SHAP explanations (global + per-student).")

# -------------------------
# Part 1: Load & Train Model
# -------------------------
@st.cache_resource
def load_and_train_model(csv_path="masked_data.csv"):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return None, None, None, None, None, None

    # Drop identifiers and Total_Score used in notebook
    drop_cols = ["Student_ID", "First_Name", "Last_Name", "Email", "Total_Score"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Fill and encode
    if "Parent_Education_Level" in df.columns:
        df["Parent_Education_Level"] = df["Parent_Education_Level"].fillna("Bachelor's")

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1}).astype(float)
    if "Extracurricular_Activities" in df.columns:
        df["Extracurricular_Activities"] = df["Extracurricular_Activities"].map({"No": 0, "Yes": 1}).astype(float)
    if "Internet_Access_at_Home" in df.columns:
        df["Internet_Access_at_Home"] = df["Internet_Access_at_Home"].map({"No": 0, "Yes": 1}).astype(float)

    edu_map = {"High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}
    if "Parent_Education_Level" in df.columns:
        df["Parent_Education_Level"] = df["Parent_Education_Level"].map(edu_map).astype(float)

    income_map = {"Low": 1, "Medium": 2, "High": 3}
    if "Family_Income_Level" in df.columns:
        df["Family_Income_Level"] = df["Family_Income_Level"].map(income_map).astype(float)

    # One-hot department into Department_* columns
    if "Department" in df.columns:
        df = pd.get_dummies(df, columns=["Department"], prefix="Department", drop_first=False)

    # Ensure Grade exists
    if "Grade" not in df.columns:
        st.error("Dataset missing 'Grade' column.")
        return None, None, None, None, None, None

    # Numeric clipping
    numeric_cols = [c for c in df.select_dtypes(include=["int64", "float64"]).columns if c != "Grade"]
    for col in numeric_cols:
        if df[col].isnull().all():
            continue
        low, high = df[col].quantile([0.05, 0.95])
        df[col] = df[col].clip(lower=low, upper=high)

    # Target encoding
    encoder = LabelEncoder()
    df["Grade"] = encoder.fit_transform(df["Grade"].astype(str))

    # Decide scale columns (avoid department one-hot)
    dept_prefix = "Department_"
    scale_cols = [c for c in numeric_cols if not c.startswith(dept_prefix)]
    scaler = StandardScaler()
    if len(scale_cols) > 0:
        df[scale_cols] = scaler.fit_transform(df[scale_cols])

    # X, y
    X = df.drop(columns="Grade")
    y = df["Grade"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    return model, scaler, encoder, X.columns.tolist(), scale_cols, df

# Load and train
model, scaler, encoder, model_columns, scale_cols, df_full = load_and_train_model()
if model is None:
    st.error("Error: 'masked_data.csv' not found or invalid. Upload CSV with expected columns.")
    st.stop()

# -------------------------
# Part 2: Input form
# -------------------------
st.sidebar.header("Student Parameters")
def user_input_features():
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    age = st.sidebar.number_input("Age", min_value=15, max_value=40, value=18)
    dept = st.sidebar.selectbox("Department", ("CS", "Engineering", "Mathematics"))
    attendance = st.sidebar.slider("Attendance (%)", 0, 100, 80)
    midterm = st.sidebar.number_input("Midterm Score", 0, 100, 75)
    final = st.sidebar.number_input("Final Score", 0, 100, 75)
    assignments = st.sidebar.number_input("Assignments Avg", 0, 100, 80)
    quizzes = st.sidebar.number_input("Quizzes Avg", 0, 100, 70)
    participation = st.sidebar.number_input("Participation Score", 0, 100, 70)
    projects = st.sidebar.number_input("Projects Score", 0, 100, 80)
    study_hours = st.sidebar.number_input("Study Hours per Week", 0, 168, 15)
    extracurricular = st.sidebar.selectbox("Extracurricular Activities", ("Yes", "No"))
    internet = st.sidebar.selectbox("Internet Access at Home", ("Yes", "No"))
    parent_edu = st.sidebar.selectbox("Parent Education", ("High School", "Bachelor's", "Master's", "PhD"))
    income = st.sidebar.selectbox("Family Income", ("Low", "Medium", "High"))
    stress = st.sidebar.slider("Stress Level (1-10)", 1, 10, 5)
    sleep = st.sidebar.number_input("Sleep Hours per Night", 0, 24, 7)

    data = {
        "Gender": gender,
        "Age": age,
        "Attendance (%)": attendance,
        "Midterm_Score": midterm,
        "Final_Score": final,
        "Assignments_Avg": assignments,
        "Quizzes_Avg": quizzes,
        "Participation_Score": participation,
        "Projects_Score": projects,
        "Study_Hours_per_Week": study_hours,
        "Extracurricular_Activities": extracurricular,
        "Internet_Access_at_Home": internet,
        "Parent_Education_Level": parent_edu,
        "Family_Income_Level": income,
        "Stress_Level (1-10)": stress,
        "Sleep_Hours_per_Night": sleep,
        "Department": dept,
    }
    return pd.DataFrame([data])

input_df = user_input_features()
st.subheader("🧾 Student Input")
st.dataframe(input_df)

# -------------------------
# Part 3: Prepare & Predict
# -------------------------
def prepare_input(input_df, model_columns, scaler, scale_cols):
    df = input_df.copy()
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1}).astype(float)
    df["Extracurricular_Activities"] = df["Extracurricular_Activities"].map({"No": 0, "Yes": 1}).astype(float)
    df["Internet_Access_at_Home"] = df["Internet_Access_at_Home"].map({"No": 0, "Yes": 1}).astype(float)

    edu_map = {"High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}
    df["Parent_Education_Level"] = df["Parent_Education_Level"].map(edu_map).astype(float)
    income_map = {"Low": 1, "Medium": 2, "High": 3}
    df["Family_Income_Level"] = df["Family_Income_Level"].map(income_map).astype(float)

    # Department one-hot columns expected: Department_CS, Department_Engineering, Department_Mathematics
    for col in ["Department_CS", "Department_Engineering", "Department_Mathematics"]:
        df[col] = 0
    dept_map = {"CS": "Department_CS", "Engineering": "Department_Engineering", "Mathematics": "Department_Mathematics"}
    chosen = df["Department"].values[0]
    if chosen in dept_map:
        df.loc[:, dept_map[chosen]] = 1
    df.drop(columns="Department", inplace=True)

    # Reindex to model columns
    df = df.reindex(columns=model_columns, fill_value=0)

    # Scale numeric columns
    if len(scale_cols) > 0:
        try:
            df[scale_cols] = scaler.transform(df[scale_cols])
        except Exception as e:
            st.warning("Input scaling failed; proceeding without scaling for input. Error: " + str(e))

    return df

# -------------------------
# Part 4: SHAP helpers (safe)
# -------------------------
def try_import_shap():
    try:
        import shap
        return shap
    except Exception:
        return None

def compute_shap_global_and_local(shap_mod, model, X_train, model_input, pred_class_idx):
    """
    Returns:
      global_shap_vals (shap_values full for classes) or None,
      local_shap (array for the predicted class for the sample) or None,
      expected_value (float)
    """
    try:
        explainer = shap_mod.TreeExplainer(model, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_train)  # list (one per class) for classifiers
        expected_values = explainer.expected_value
        # shap_values[class_idx] shape: (n_samples, n_features)
        local_vals = None
        exp_val = None
        if isinstance(shap_values, list):
            # classification
            local_vals = shap_values[pred_class_idx][0]
            exp_val = expected_values[pred_class_idx] if hasattr(expected_values, "__len__") else expected_values
        else:
            # regression-like
            local_vals = shap_values[0]
            exp_val = explainer.expected_value
        return shap_values, local_vals, exp_val, explainer
    except Exception as e:
        st.warning("SHAP computation failed: " + str(e))
        return None, None, None, None

# -------------------------
# Part 5: Run prediction, charts, SHAP
# -------------------------
if st.button("Predict & Analyze"):
    model_input = prepare_input(input_df, model_columns, scaler, scale_cols)

    # Predict
    pred_enc = model.predict(model_input)
    pred_label = encoder.inverse_transform(pred_enc)[0]

    st.divider()
    st.subheader("🎯 Predicted Grade")
    if pred_label == "A":
        st.success(f"Predicted Grade: **{pred_label}** 🌟")
        st.balloons()
    elif pred_label == "C":
        st.info(f"Predicted Grade: **{pred_label}**")
    else:
        st.error(f"Predicted Grade: **{pred_label}**")

    # Feature importance (model-level)
    st.subheader("📊 Model Feature Importance (RandomForest)")
    feat_names = model_columns
    importances = model.feature_importances_
    fi_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=True)
    fig_fi, ax_fi = plt.subplots(figsize=(8, max(4, 0.25 * len(fi_df))))
    ax_fi.barh(fi_df["feature"], fi_df["importance"])
    ax_fi.set_xlabel("Importance")
    ax_fi.set_title("RandomForest Feature Importances")
    st.pyplot(fig_fi)

    # Score breakdown
    st.subheader("📈 Score Breakdown")
    labels = ["Midterm", "Final", "Assignments", "Quizzes", "Participation", "Projects"]
    scores = [
        input_df.loc[0, "Midterm_Score"],
        input_df.loc[0, "Final_Score"],
        input_df.loc[0, "Assignments_Avg"],
        input_df.loc[0, "Quizzes_Avg"],
        input_df.loc[0, "Participation_Score"],
        input_df.loc[0, "Projects_Score"],
    ]
    fig_sb, ax_sb = plt.subplots(figsize=(8, 4))
    ax_sb.bar(labels, scores)
    ax_sb.set_ylim(0, 100)
    ax_sb.set_ylabel("Score")
    ax_sb.set_title("Exam/Assessment Score Breakdown")
    plt.xticks(rotation=15)
    st.pyplot(fig_sb)

    # Attendance vs Final Score plot (dataset background + student marker)
    st.subheader("📉 Attendance vs Final Score (Context)")
    if "Attendance (%)" in df_full.columns and "Final_Score" in df_full.columns:
        fig_att, ax_att = plt.subplots(figsize=(8, 4))
        ax_att.scatter(df_full["Attendance (%)"], df_full["Final_Score"], alpha=0.4)
        ax_att.scatter(input_df.loc[0, "Attendance (%)"], input_df.loc[0, "Final_Score"], marker="x", s=120, color="k")
        ax_att.set_xlabel("Attendance (%)")
        ax_att.set_ylabel("Final Score")
        ax_att.set_title("Attendance vs Final Score (dataset)")
        st.pyplot(fig_att)
    else:
        st.info("Attendance vs Final Score chart requires those columns in dataset.")

    # SHAP integration (mix approach)
    st.subheader("🔬 SHAP Explainability (Global + Per-student)")

    shap_mod = try_import_shap()
    if shap_mod is None:
        st.warning("shap package not found. To enable SHAP explanations, install it: `pip install shap==0.42.1`")
        st.info("Falling back to textual feature-contribution explanation.")
        # Fallback: text-based top contributors using feature importances and input values
        # compute per-feature pseudo-contribution: (value - mean) * feature_importance
        try:
            mean_vals = df_full[model_columns].mean()
            input_series = model_input.iloc[0]
            contribs = {}
            for f in model_columns:
                fi = model.feature_importances_[list(model_columns).index(f)]
                contribs[f] = (input_series[f] - mean_vals[f]) * fi
            contribs_series = pd.Series(contribs).sort_values(ascending=False)
            st.markdown("**Top positive contributors (pseudo):**")
            for f, v in contribs_series.head(6).items():
                st.write(f"- {f}: {v:.3f}")
            st.markdown("**Top negative contributors (pseudo):**")
            for f, v in contribs_series.tail(6).sort_values().items():
                st.write(f"- {f}: {v:.3f}")
        except Exception as e:
            st.warning("Fallback explanation failed: " + str(e))
    else:
        # Compute SHAP values on training set (or subset for speed)
        # use up to 1000 rows to keep it fast
        X_train = df_full[model_columns].copy()
        if len(X_train) > 1000:
            X_train_sample = X_train.sample(1000, random_state=42)
        else:
            X_train_sample = X_train

        # Compute predicted class index
        try:
            pred_class_idx = int(pred_enc[0])
        except Exception:
            pred_class_idx = 0

        shap_vals_all, local_vals, exp_val, explainer = compute_shap_global_and_local(shap_mod, model, X_train_sample, model_input, pred_class_idx)
        if shap_vals_all is None:
            st.warning("SHAP calculation failed; showing textual fallback.")
        else:
            # Global summary (matplotlib-based summary plot)
            try:
                st.markdown("**Global SHAP summary (feature impact across dataset)**")
                fig_sum = plt.figure(figsize=(8, 6))
                # For classifier shap_values is list; pick the class with highest avg importance magnitude
                if isinstance(shap_vals_all, list):
                    # choose class with largest mean abs shap
                    mean_abs = [np.mean(np.abs(sv)) for sv in shap_vals_all]
                    class_idx_for_summary = int(np.argmax(mean_abs))
                    shap_mod.summary_plot(shap_vals_all[class_idx_for_summary], X_train_sample, show=False)
                else:
                    shap_mod.summary_plot(shap_vals_all, X_train_sample, show=False)
                st.pyplot(fig_sum)
                plt.close(fig_sum)
            except Exception as e:
                st.warning("Global SHAP summary plot failed: " + str(e))

            # Per-sample bar of top contributors (matplotlib)
            try:
                st.markdown("**Per-student SHAP contributions (top positive & negative)**")
                # local_vals is a 1d array of shap values for each feature for this instance
                # create DataFrame
                local_df = pd.DataFrame({
                    "feature": model_columns,
                    "shap_value": local_vals
                })
                local_df["abs_shap"] = local_df["shap_value"].abs()
                top_pos = local_df[local_df["shap_value"] > 0].sort_values("shap_value", ascending=False).head(6)
                top_neg = local_df[local_df["shap_value"] < 0].sort_values("shap_value").head(6)
                top_df = pd.concat([top_pos, top_neg]).sort_values("shap_value")

                fig_local, ax_local = plt.subplots(figsize=(8, max(3, 0.5 * len(top_df))))
                colors = ["#d9534f" if v < 0 else "#5cb85c" for v in top_df["shap_value"]]
                ax_local.barh(top_df["feature"], top_df["shap_value"], color=colors)
                ax_local.set_xlabel("SHAP value (impact on model output)")
                ax_local.set_title("Top SHAP contributors for this student")
                st.pyplot(fig_local)
                plt.close(fig_local)
            except Exception as e:
                st.warning("Per-student SHAP bar plot failed: " + str(e))
                # textual fallback
                top_text = local_df.sort_values("abs_shap", ascending=False).head(6)
                st.markdown("**Top contributors (fallback text):**")
                for _, r in top_text.iterrows():
                    sign = "↑" if r["shap_value"] > 0 else "↓"
                    st.write(f"- {r['feature']}: {sign} {r['shap_value']:.3f}")

    # -------------------------
    # Text summary & recommendations (as before)
    # -------------------------
    st.subheader("📝 Performance Summary & Recommendations")
    summary = []
    recs = []

    att = input_df.loc[0, "Attendance (%)"]
    sh = input_df.loc[0, "Study_Hours_per_Week"]
    stress_val = input_df.loc[0, "Stress_Level (1-10)"]
    sleep_h = input_df.loc[0, "Sleep_Hours_per_Night"]
    final_score = input_df.loc[0, "Final_Score"]
    midterm_score = input_df.loc[0, "Midterm_Score"]

    # Attendance
    if att >= 85:
        summary.append("✔ Excellent attendance — good predictor of consistent learning.")
    elif att >= 70:
        summary.append("▲ Attendance is decent but could be improved.")
        recs.append("Aim for at least 85% attendance for better continuity.")
    else:
        summary.append("❗ Low attendance — likely negatively impacting performance.")
        recs.append("Increase class attendance and review missed lectures.")

    # Study hours
    if sh >= 12:
        summary.append("✔ Adequate study hours per week.")
    else:
        summary.append("❗ Low weekly study time.")
        recs.append("Structure study: target 12–15 hours/week with active recall & spaced repetition.")

    # Stress & sleep
    if stress_val >= 7:
        summary.append("❗ High reported stress — may harm performance and retention.")
        recs.append("Adopt stress management: short breaks, exercise, or counseling.")
    else:
        summary.append("✔ Stress level within manageable range.")

    if sleep_h < 6:
        summary.append("❗ Short sleep duration — can impair memory consolidation.")
        recs.append("Improve sleep hygiene: target 7–8 hours nightly.")
    else:
        summary.append("✔ Healthy sleep duration.")

    # Scores
    if final_score >= 80 and midterm_score >= 75:
        summary.append("✔ Strong exam performance — keep refining weak areas.")
    elif final_score < 60:
        summary.append("❗ Final exam underperforming — targeted revision required.")
        recs.append("Focus on past exam papers and targeted problem practice for weak topics.")

    for s in summary:
        st.write(s)

    if len(recs) > 0:
        st.markdown("**Recommendations:**")
        for r in recs:
            st.write("- " + r)
    else:
        st.success("No urgent recommendations — keep up the good work!")

    st.caption("SHAP-based explanations provide feature-level reasons behind the model's prediction. Use them together with domain knowledge for interventions.")
    with st.expander("Dataset snapshot (for context)"):
        st.dataframe(df_full.head(200))
    st.success("Analysis complete ✔")
