import streamlit as st
import pickle
from preprocessing import preprocessing

# ======================================
# KONFIGURASI
# ======================================
st.set_page_config(
    page_title="Analisis Sentimen BPD Bali",
    page_icon="🏦",
    layout="centered"
)

# ======================================
# CSS
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

col1, col2, col3 = st.columns([2,1,2])

with col2:
    st.image("logo-bank-bpd-bali.png", width=120)

st.markdown(
    "<h1 class='judul'>Analisis Sentimen</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle'>Ulasan Aplikasi BPD Bali Mobile</p>",
    unsafe_allow_html=True
)

st.write("")

# ======================================
# INPUT
# ======================================

st.markdown(
    "<p class='label'>Masukkan Ulasan</p>",
    unsafe_allow_html=True
)

ulasan = st.text_area(
    "",
    height=180,
    placeholder="Contoh : Aplikasi sangat membantu dan mudah digunakan."
)

# ======================================
# BUTTON
# ======================================

if st.button("🔍 Analisis Sentimen"):

    if ulasan.strip() == "":
        st.warning("Masukkan ulasan terlebih dahulu.")

    else:

        hasil = preprocessing(ulasan)

        vector = vectorizer.transform([hasil])

        vector = chi_selector.transform(vector)

        prediksi = model.predict(vector)[0]

        st.divider()

        st.subheader("Hasil Analisis")

        if prediksi == 1:
            st.success("😊 Sentimen Positif")
        else:
            st.error("😞 Sentimen Negatif")

        with st.expander("Hasil Preprocessing"):
            st.write(hasil)
