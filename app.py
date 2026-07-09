import streamlit as st
import pickle

from preprocessing import preprocessing

# ==========================
# Load Model
# ==========================
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

with open("chi_selector.pkl", "rb") as file:
    chi_selector = pickle.load(file)

with open("model_naive_bayes.pkl", "rb") as file:
    model = pickle.load(file)

# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(
    page_title="Analisis Sentimen BPD Bali",
    page_icon="📱",
    layout="centered"
)

# ==========================
# Tampilan
# ==========================

st.title("📱 Analisis Sentimen Aplikasi BPD Bali Mobile")

st.write("""
Masukkan ulasan pengguna aplikasi **BPD Bali Mobile** untuk mengetahui
hasil analisis sentimen menggunakan metode **Naïve Bayes Classifier**
dengan seleksi fitur **Chi-Square**.
""")

# Input
teks = st.text_area(
    "Masukkan Ulasan",
    placeholder="Contoh: Aplikasi sangat membantu dan mudah digunakan."
)

# Tombol Prediksi
if st.button("Analisis Sentimen"):

    if teks.strip() == "":
        st.warning("Silakan masukkan ulasan terlebih dahulu.")

    else:

        # ==========================
        # Preprocessing
        # ==========================
        clean_text = preprocessing(teks)

        # ==========================
        # Vectorisasi
        # ==========================
        vector = vectorizer.transform([clean_text])

        # ==========================
        # Seleksi Fitur Chi-Square
        # ==========================
        vector = chi_selector.transform(vector)

        # ==========================
        # Prediksi
        # ==========================
        prediksi = model.predict(vector)[0]

        st.subheader("Hasil Analisis")

        if prediksi == 1:
            st.success("😊 Sentimen Positif")
        else:
            st.error("😞 Sentimen Negatif")

        # Menampilkan hasil preprocessing
        st.write("### Hasil Preprocessing")
        st.write(clean_text)
