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
# REFINEMENT CSS (PROPORSIONAL & SUPER RAPAT)
# ======================================
st.markdown("""
<style>
/* Background Full Hijau BPD Bali */
.stApp {
    background-color: #0B6B3A;
}

/* Mengatur container utama agar aman */
.block-container {
    max-width: 680px;
    padding-top: 20px; 
    padding-bottom: 40px;
}

/* CSS Khusus Kontainer Logo Baru (Menurunkan logo tanpa menjauhkan tulisan) */
.logo-container {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
    padding-top: 40px;    /* Ini yang mendorong LOGO TURUN agar tidak kepotong */
    margin-bottom: 5px;   /* Ini yang menjaga LOGO TETAP DEKAT dengan tulisan h1 */
}

.logo-container img {
    width: 100%;
    max-width: 140px;     /* Ukuran mini proporsional */
    height: auto;
    object-fit: contain;
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
# HEADER & LOGO (PERBAIKAN PERMANEN)
# ======================================

# Kita tidak pakai st.columns/st.image bawaan untuk logo agar tidak di-override Streamlit.
# Kita panggil langsung via HTML bertingkat yang sudah dikunci oleh CSS di atas.
st.markdown("""
<div class="logo-container">
    <img src="app/static/logo-bank-bpd-bali.png" alt="Logo BPD Bali">
</div>
""", unsafe_allow_html=True)

# Teks Judul Utama (Ukurannya diperkecil ke 38px & margin nempel)
st.markdown("""
<h1 style="
    text-align: center;
    color: white;
    font-size: 38px;
    font-weight: bold;
    margin-top: 0px; 
    margin-bottom: 0px;
    letter-spacing: -0.5px;">
    Analisis Sentimen
</h1>
""", unsafe_allow_html=True)

# Teks Sub-judul (Diperkecil ke 18px & jarak bawah disesuaikan)
st.markdown("""
<p style="
    text-align: center;
    color: rgba(255, 255, 255, 0.9);
    font-size: 18px;
    margin-top: 0px;
    margin-bottom: 30px;">
    Ulasan Aplikasi BPD Bali Mobile
</p>
""", unsafe_allow_html=True)

# ======================================
# INPUT USER
# ======================================
st.markdown("""
<p style="
    font-size: 18px;
    font-weight: bold;
    color: white;
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

st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

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
