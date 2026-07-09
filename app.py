import streamlit as st
import pickle
import base64
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
# REFINEMENT CSS (FIX LOGO & RESPONSIVE)
# ======================================
st.markdown("""
<style>
/* Background Full Hijau BPD Bali */
.stApp {
    background-color: #0B6B3A;
}

/* Mengatur container utama agar fleksibel */
.block-container {
    max-width: 680px;
    width: 100%;
    padding-top: 40px !important; /* Memberikan ruang atas agar logo tidak mentok */
    padding-bottom: 40px;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
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

/* Kerapatan Text untuk Judul HTML */
.brand-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    width: 100%;
    margin-bottom: 25px;
}

.brand-logo {
    width: 130px !important;
    height: auto !important;
    margin-bottom: 10px;
}

.brand-title {
    color: white !important;
    font-size: 38px !important;
    font-weight: bold !important;
    margin: 0 !important;
    line-height: 1.1 !important;
    letter-spacing: -0.5px !important;
}

.brand-subtitle {
    color: rgba(255, 255, 255, 0.9) !important;
    font-size: 18px !important;
    margin-top: 3px !important;
    margin-bottom: 0px !important;
}

/* CSS RESPONSIVE HP */
@media (max-width: 768px) {
    .brand-logo {
        width: 105px !important; /* Logo mengecil sedikit di HP */
    }
    .brand-title {
        font-size: 28px !important; /* Judul mengecil di HP agar pas */
    }
    .brand-subtitle {
        font-size: 15px !important;
    }
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
# FUNGSI ENKODE GAMBAR LOKAL KE BASE64
# ======================================
# Ini trik wajib agar HTML di Streamlit bisa membaca file gambar lokal laptop Anda
def get_image_base64(path):
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        return ""

img_base64 = get_image_base64("logo-bank-bpd-bali.png")

# ======================================
# HEADER & LOGO MURNI HTML (100% LURUS TENGAH)
# ======================================
st.html(f"""
<div class="brand-container">
    <img class="brand-logo" src="data:image/png;base64,{img_base64}" alt="Logo BPD Bali">
    <h1 class="brand-title">Analisis Sentimen</h1>
    <p class="brand-subtitle">Ulasan Aplikasi BPD Bali Mobile</p>
</div>
""")

# ======================================
# INPUT USER
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
