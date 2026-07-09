import streamlit as st
import pickle

from preprocessing import preprocessing

# ==========================
# Load Model
# ==========================
with open("Model/vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

with open("Model/chi_selector.pkl", "rb") as file:
    chi_selector = pickle.load(file)

with open("Model/model_naive_bayes.pkl", "rb") as file:
    model = pickle.load(file)

# ==========================
# Tampilan Streamlit
# ==========================

st.set_page_config(
    page_title="Analisis Sentimen BPD Bali",
    page_icon="📱"
)

st.title("📱 Analisis Sentimen Aplikasi BPD Bali Mobile")
st.write("Masukkan ulasan pengguna untuk mengetahui hasil analisis sentimen.")

teks = st.text_area("Masukkan Ulasan")

if st.button("Analisis Sentimen"):

    if teks.strip() == "":
        st.warning("Masukkan ulasan terlebih dahulu.")
    else:

        # Preprocessing
        clean_text = preprocessing(teks)

        # Vectorisasi
        vector = vectorizer.transform([clean_text])

        # Seleksi fitur
        vector = chi_selector.transform(vector)

        # Prediksi
        prediksi = model.predict(vector)[0]

        if prediksi == 1:
            st.success("😊 Sentimen Positif")
        else:
            st.error("😞 Sentimen Negatif")
