import streamlit as st
import pickle
from preprocessing import preprocessing

# ======================================
# KONFIGURASI HALAMAN
# ======================================
st.set_page_config(
    page_title="Analisis Sentimen BPD Bali",
    page_icon="🏦",
    layout="centered"
)

# ======================================
# MEMANGGIL CSS
# ======================================
with open("style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

# ======================================
# LOAD MODEL
# ======================================
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

with open("chi_selector.pkl", "rb") as file:
    chi_selector = pickle.load(file)

with open("model_naive_bayes.pkl", "rb") as file:
    model = pickle.load(file)

# ======================================
# HEADER
# ======================================

col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.image("logo-bank-bpd-bali.png", width=130)

st.markdown(
    """
    <h1 style="text-align:center;">
    Analisis Sentimen
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p class="subtitle">
    Ulasan Aplikasi BPD Bali Mobile
    </p>
    """,
    unsafe_allow_html=True
)

# ======================================
# INPUT
# ======================================

st.subheader("Masukkan Ulasan")

ulasan = st.text_area(
    "",
    height=180,
    placeholder="Contoh: Aplikasi sangat membantu dan mudah digunakan."
)

# ======================================
# PREDIKSI
# ======================================

if st.button("🔍 Analisis Sentimen"):

    if ulasan.strip() == "":
        st.warning("Silakan masukkan ulasan terlebih dahulu.")

    else:

        # Preprocessing
        hasil_preprocessing = preprocessing(ulasan)

        # CountVectorizer
        vector = vectorizer.transform([hasil_preprocessing])

        # Chi-Square
        vector = chi_selector.transform(vector)

        # Prediksi
        prediksi = model.predict(vector)[0]

        # Confidence (jika model mendukung)
        try:
            probabilitas = model.predict_proba(vector)[0]
            confidence = max(probabilitas) * 100
        except:
            confidence = None

        st.divider()

        st.subheader("Hasil Analisis")

        if prediksi == 1:

            st.success("😊 Sentimen Positif")

        else:

            st.error("😞 Sentimen Negatif")

        if confidence is not None:

            st.info(f"Tingkat Keyakinan Model : **{confidence:.2f}%**")

        with st.expander("Hasil Preprocessing"):

            st.write(hasil_preprocessing)
