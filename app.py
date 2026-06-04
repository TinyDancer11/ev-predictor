import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import shap

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EV Adoption Predictor",
    page_icon="⚡",
    layout="centered",
)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(__file__)

    with open(os.path.join(base, "catboost_ev_model.pkl"), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(base, "feature_columns.pkl"), "rb") as f:
        features = pickle.load(f)

    with open(os.path.join(base, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    explainer = shap.TreeExplainer(model)

    return model, features, scaler, explainer


model, feature_columns, scaler, explainer = load_models()

# ── Question labels ──────────────────────────────────────────────────────────
QUESTION_LABELS = {
    "q22_future_belief": "Keyakinan bahwa EV adalah kendaraan masa depan",
    "q25_infra_concern": "Kekhawatiran terhadap infrastruktur pengisian daya",
    "q21_tech_interest": "Ketertarikan terhadap teknologi kendaraan baru",
    "q24_range_anxiety": "Kecemasan terhadap jangkauan baterai (range anxiety)",
    "q23_ev_knowledge": "Pengetahuan tentang kendaraan listrik",
    "q16_fuel_cost": "Kepedulian terhadap biaya bahan bakar",
    "q9_frekuensi_pakai": "Frekuensi penggunaan kendaraan sehari-hari",
    "q19_interest_green": "Minat terhadap gaya hidup ramah lingkungan",
}

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("⚡ EV Adoption Predictor")

st.markdown(
    """
Aplikasi ini memprediksi kemungkinan seseorang mengadopsi kendaraan listrik (EV)
berdasarkan hasil survei dan menjelaskan alasan prediksi menggunakan Explainable AI (SHAP).
"""
)

st.divider()

st.subheader("🔍 Kuesioner")

st.caption(
    "1 = Sangat Tidak Setuju | 2 = Tidak Setuju | 3 = Netral | 4 = Setuju | 5 = Sangat Setuju"
)

responses = {}

cols = st.columns(2)

for i, col_name in enumerate(feature_columns):

    label = QUESTION_LABELS.get(col_name, col_name)

    with cols[i % 2]:

        value = st.slider(
            label,
            min_value=1,
            max_value=5,
            value=3,
            help=f"Feature: {col_name}",
        )

        responses[col_name] = value

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button(
    "🔮 Prediksi Sekarang",
    use_container_width=True,
    type="primary"
):

    input_df = pd.DataFrame(
        [[responses[col] for col in feature_columns]],
        columns=feature_columns
    )

    scaled = scaler.transform(input_df)

    prediction = model.predict(scaled)[0]

    proba = model.predict_proba(scaled)[0]

    prob_adopt = float(proba[1]) * 100
    prob_no = float(proba[0]) * 100

    st.divider()

    st.subheader("📊 Hasil Prediksi")

    if prediction == 1:
        st.success(
            f"✅ Kemungkinan AKAN mengadopsi EV ({prob_adopt:.1f}%)"
        )
    else:
        st.warning(
            f"❌ Kemungkinan TIDAK mengadopsi EV ({prob_no:.1f}%)"
        )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Probabilitas Adopsi EV",
            f"{prob_adopt:.1f}%"
        )

    with col2:
        st.metric(
            "Probabilitas Tidak Adopsi",
            f"{prob_no:.1f}%"
        )

    st.progress(
        int(prob_adopt),
        text=f"Confidence Adopsi EV: {prob_adopt:.1f}%"
    )

    # ──────────────────────────────────────────────────────────────────────
    # SHAP EXPLANATION
    # ──────────────────────────────────────────────────────────────────────

    st.divider()

    st.subheader("🧠 Penjelasan Prediksi (SHAP)")

    scaled_df = pd.DataFrame(
        scaled,
        columns=feature_columns
    )

    try:

        shap_values = explainer.shap_values(scaled_df)

        if isinstance(shap_values, list):
            shap_values_single = shap_values[1][0]
        else:
            shap_values_single = shap_values[0]

        shap_df = pd.DataFrame({
            "Feature": feature_columns,
            "Contribution": shap_values_single
        })

        shap_df["Question"] = shap_df["Feature"].map(
            lambda x: QUESTION_LABELS.get(x, x)
        )

        shap_df["AbsContribution"] = shap_df["Contribution"].abs()

        shap_df = shap_df.sort_values(
            by="AbsContribution",
            ascending=False
        )

        st.markdown("### 📈 Faktor yang Paling Berpengaruh")

        chart_df = shap_df[["Question", "Contribution"]].set_index("Question")

        st.bar_chart(chart_df)

        st.dataframe(
            shap_df[["Question", "Contribution"]],
            use_container_width=True
        )

    except Exception as e:

        st.error(f"Gagal membuat visualisasi SHAP: {str(e)}")

    # ──────────────────────────────────────────────────────────────
    # Detail Input
    # ──────────────────────────────────────────────────────────────

    with st.expander("📋 Detail Input yang Dikirim"):

        display_df = pd.DataFrame(
            [responses]
        ).rename(
            columns=QUESTION_LABELS
        ).T

        display_df.columns = ["Nilai"]

        st.dataframe(
            display_df,
            use_container_width=True
        )