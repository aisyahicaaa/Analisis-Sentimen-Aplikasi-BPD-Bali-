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
# REFINEMENT CSS (SUPER RAPAT & PADAT)
# ======================================
st.markdown("""
<style>
/* Background Full Hijau BPD Bali */
.stApp {
    background-color: #0B6B3A;
}

/* Mengatur container utama */
.block-container {
    max-width: 680px;
    padding-top: 0px !important; 
    padding-bottom: 40px;
}

/* Memastikan gambar/logo di dalam kolom otomatis centering */
[data-testid="stImage"] {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 0 auto;
}

/* Merapikan Text Area bawaan */
.stTextArea textarea {
    background: white !important;
    color: black !important;
    font-size: 16px !important;
    border-radius: 10px !important;
    padding: 12px !important;
}

/* Merapikan Tombol Analisis */
.stButton > button {
    width: 100%;
    height: 50px;
    background: white !important;
    color: #0B6B3A !important;
    font-size: 18px !important;
    font-weight: bold !important;
    border: none !important;
    border-radius: 8px !important;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #F2F2F2 !important;
    color: #0B6B3A !important;
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)

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
# HEADER & LOGO (JARAK AMAN ATAS)
# ======================================

# Container kosong untuk menurunkan logo agar tidak kepotong layar atas browser
st.container(height=45, border=False)

# Mengunci posisi logo di tengah dengan ukuran mini/pas
kiri, tengah, kanan = st.columns([1.6, 1, 1.6]) 
with tengah:
    st.image("logo-bank-bpd-bali.png", use_container_width=True)

# Teks Judul Utama (Menggunakan margin-top minus agar naik memeluk bagian bawah logo)
st.markdown("""
<h1 style="
    text-align: center;
    color: white;
    font-size: 38px;
    font-weight: bold;
    margin-top: -20px; 
    margin-bottom: 0px;
    line-height: 1.0;
    letter-spacing: -0.5px;">
    Analisis Sentimen
</h1>
""", unsafe_allow_html=True)

# Teks Sub-judul (Dibuat menempel langsung di bawah h1)
st.markdown("""
<p style="
    text-align: center;
    color: rgba(255, 255, 255, 0.9);
    font-size: 18px;
    margin-top: 2px;
    margin-bottom: 20px;">
    Ulasan Aplikasi BPD Bali Mobile
</p>
""", unsafe_allow_html=True)

# ======================================
# INPUT USER (SANGAT DEKAT DENGAN SUBJUDUL)
# ======================================
st.markdown("""
<p style="
    font-size: 18px;
    font-weight: bold;
    color: white;
    margin-top: 0px;
    margin-bottom: 8px;">
    Masukkan Ulasan
</p>
""", unsafe_allow_html=True)

ulasan = st.text_area(
    "",
    height=150,
    placeholder="Contoh: Aplikasi sangat membantu dan mudah digunakan.",
    label_visibility="collapsed"
)

# Jarak tipis ke tombol
st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

# ======================================
# BUTTON & PROSES PREDIKSI
# ======================================
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
