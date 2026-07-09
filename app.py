import streamlit as st
import pickle
from preprocessing import preprocessing

# ==========================
# LOAD MODEL
# ==========================
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

with open("chi_selector.pkl", "rb") as file:
    chi_selector = pickle.load(file)

with open("model_naive_bayes.pkl", "rb") as file:
    model = pickle.load(file)

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Analisis Sentimen BPD Bali",
    page_icon="🏦",
    layout="centered"
)

# ==========================
# MEMANGGIL CSS
# ==========================
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ==========================
# TAMPILAN
# ==========================

st.image("logo-bank-bpd-bali.png", width=90)

st.markdown(
    "<h1>Analisis Sentimen</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='sub'>Ulasan Aplikasi BPD Bali Mobile</p>",
    unsafe_allow_html=True
)

teks = st.text_area(
    "Masukkan Ulasan",
    placeholder="Contoh: Aplikasi sangat membantu dan mudah digunakan.",
    height=180
)

if st.button("Analisis Sentimen"):

    if teks.strip() == "":
        st.warning("Silakan masukkan ulasan terlebih dahulu.")

    else:
        # Preprocessing
        clean_text = preprocessing(teks)

        # Vectorisasi
        vector = vectorizer.transform([clean_text])

        # Seleksi fitur
        vector = chi_selector.transform(vector)

        # Prediksi
        prediksi = model.predict(vector)[0]

        st.markdown("---")

        if prediksi == 1:
            st.success("😊 Sentimen Positif")
        else:
            st.error("😞 Sentimen Negatif")

        with st.expander("Hasil Preprocessing"):
            st.write(clean_text)
