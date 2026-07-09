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
# REFINEMENT CSS (FULLY RESPONSIVE & PADAT)
# ======================================
st.markdown("""
<style>
/* Background Full Hijau BPD Bali */
.stApp {
    background-color: #0B6B3A;
}

/* Mengatur container utama agar fleksibel di berbagai layar */
.block-container {
    max-width: 680px;
    width: 100%;
    padding-top: 0px !important; 
    padding-bottom: 40px;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

/* Mengunci Gambar/Logo agar responsif di tengah */
[data-testid="stImage"] {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 0 auto;
    width: 100% !important;
}

[data-testid="stImage"] img {
    width: 140px !important; /* Ukuran default laptop */
    height: auto !important;
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

/* CSS RESPONSIF: Aturan khusus untuk layar HP / Mobile (Maksimal lebar 768px) */
@media (max-width: 768px) {
    [data-testid="stImage"] img {
        width: 110px !important; /* Logo sedikit mengecil di HP agar proporsional */
    }
    
    .stApp h1 {
        font-size: 28px !important; /* Judul mengecil di HP agar tidak patah berantakan */
        margin-top: -10px !important; /* Penyesuaian jarak rapat untuk mobile */
    }
    
    .stApp p {
        font-size: 15px !important; /* Sub-judul mengecil di HP */
    }
    
    .input-label {
        font-size: 16px !important;
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
# HEADER & LOGO
# ======================================

# Container kosong menjaga jarak atas browser agar logo tidak kepotong
st.container(height=45, border=False)

# Menggunakan satu kolom penuh agar logo tidak terhimpit di HP, centering diatur lewat CSS [data-testid="stImage"]
st.image("logo-bank-bpd-bali.png")

# Teks Judul Utama (Menempel rapat di bawah logo)
st.markdown("""
<h1 style="
    text-align: center;
    color: white;
    font-size: 38px;
    font-weight: bold;
    margin-top: -20px; 
    margin-bottom: 0px;
    line-height: 1.1;
    letter-spacing: -0.5px;">
    Analisis Sentimen
</h1>
""", unsafe_allow_html=True)

# Teks Sub-judul (Diberi margin-top minus agar menempel sangat dekat dengan judul atasnya)
st.markdown("""
<p style="
    text-align: center;
    color: rgba(255, 255, 255, 0.9);
    font-size: 18px;
    margin-top: -3px; 
    margin-bottom: 20px;">
    Ulasan Aplikasi BPD Bali Mobile
</p>
""", unsafe_allow_html=True)

# ======================================
# INPUT USER
# ======================================
st.markdown("""
<p class="input-label" style="
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
