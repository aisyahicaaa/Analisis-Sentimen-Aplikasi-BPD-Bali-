import streamlit as st
import pickle
from preprocessing import preprocessing

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(
    page_title="Analisis Sentimen BPD Bali",
    page_icon="🏦",
    layout="centered"
)

# ==========================
# MEMANGGIL CSS
# ==========================
with open("style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

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
# CARD
# ==========================
st.markdown('<div class="card">', unsafe_allow_html=True)

# Logo
col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.image("logo-bank-bpd-bali.png", width=110)

# Judul
st.markdown(
    "<h1>Analisis Sentimen</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='sub'>Ulasan Aplikasi BPD Bali Mobile</p>",
    unsafe_allow_html=True
)

# Input
teks = st.text_area(
    "Masukkan Ulasan",
    placeholder="Contoh : Aplikasi sangat membantu dan mudah digunakan.",
    height=180
)

# Tombol
if st.button("Analisis Sentimen"):

    if teks.strip() == "":
        st.warning("Masukkan ulasan terlebih dahulu.")

    else:

        # Preprocessing
        clean_text = preprocessing(teks)

        # Vectorizer
        vector = vectorizer.transform([clean_text])

        # Chi Square
        vector = chi_selector.transform(vector)

        # Prediksi
        prediksi = model.predict(vector)[0]

        st.markdown("---")

        st.subheader("Hasil Analisis")

        if prediksi == 1:

            st.success("😊 Sentimen Positif")

        else:

            st.error("😞 Sentimen Negatif")

        with st.expander("Lihat Hasil Preprocessing"):

            st.write(clean_text)

st.markdown("</div>", unsafe_allow_html=True)
