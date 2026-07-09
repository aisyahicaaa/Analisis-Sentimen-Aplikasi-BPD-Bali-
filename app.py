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
# MEMANGGIL CSS EKSTERNAL
# ======================================
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Memuat file style.css dari folder static
local_css("static/style.css")

# ======================================
# LOAD MODEL
# ======================================
@st.cache_resource
def load_models():
    with open("vectorizer.pkl", "rb") as file:
        vec = pickle.load(file)
    with open("chi_selector.pkl", "rb") as file:
        sel = pickle.load(file)
    with open("model_naive_bayes.pkl", "rb") as file:
        mdl = pickle.load(file)
    return vec, sel, mdl

try:
    vectorizer, chi_selector, model = load_models()
except FileNotFoundError:
    st.error("Model (.pkl) tidak ditemukan, pastikan file berada di folder yang sama.")

# ======================================
# MEMANGGIL HTML EKSTERNAL (HEADER & JUDUL)
# ======================================
# Container kosong menjaga jarak atas browser agar logo tidak kepotong
st.container(height=45, border=False)

# Mengunci posisi logo di tengah menggunakan kolom Streamlit agar 100% muncul
kiri, tengah, kanan = st.columns([1.6, 1, 1.6]) 
with tengah:
    st.image("logo-bank-bpd-bali.png", use_container_width=True)

# Memuat teks judul dan sub-judul dari file index.html
with open("templates/index.html", "r") as f:
    html_content = f.read()
st.markdown(html_content, unsafe_allow_html=True)

# ======================================
# INPUT USER & PROSES PREDIKSI
# ======================================
ulasan = st.text_area(
    "",
    height=150,
    placeholder="Contoh: Aplikasi sangat membantu dan mudah digunakan.",
    label_visibility="collapsed"
)

st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

if st.button("🔍 Analisis Sentimen"):
    if ulasan.strip() == "":
        st.warning("Masukkan ulasan terlebih dahulu.")
    else:
        hasil = preprocessing(ulasan)
        vector = vectorizer.transform([hasil])
        vector = chi_selector.transform(vector)
        prediksi = model.predict(vector)[0]
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if prediksi == 1:
            st.success("😊 Sentimen Terdeteksi: POSITIF")
        else:
            st.error("😞 Sentimen Terdeteksi: NEGATIF")
            
        with st.expander("Lihat Hasil Preprocessing"):
            st.write(hasil)
