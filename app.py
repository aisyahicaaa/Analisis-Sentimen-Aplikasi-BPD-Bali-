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
# REFINEMENT CSS (AMAN & RESPONSIF)
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
    padding-top: 0px !important; 
    padding-bottom: 40px;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

/* Menghilangkan margin bawaan kolom Streamlit khusus logo agar bisa mepet */
[data-testid="stHorizontalBlock"] {
    margin-bottom: 0px !important;
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

/* CSS RESPONSIF: Otomatis mengecilkan tulisan jika dibuka di HP */
@media (max-width: 768px) {
    h1 {
        font-size: 28px !important; /* Judul mengecil di HP agar pas */
        margin-top: -15px !important; /* Tetap rapat di HP */
    }
    
    p {
        font-size: 15px !important; /* Sub-judul mengecil di HP */
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
# HEADER & LOGO (DINKUNCI AMAN DI TENGAH)
# ======================================

# Spasi aman atas agar tidak menempel ke batas browser
st.container(height=45, border=False)

# Menggunakan kolom fleksibel: otomatis ke tengah di laptop, dan tetap aman di HP
kiri, tengah, kanan = st.columns([1, 1, 1])
with tengah:
    # Mengunci ukuran lebar logo tepat 140 pixel agar pas di laptop dan HP (bebas eror layout)
    st.image("logo-bank-bpd-bali.png", width=140)

# Teks Judul Utama (Menempel rapat di bawah logo)
st.markdown("""
<h1 style="
    text-align: center;
    color: white;
    font-size: 38px;
    font-weight: bold;
    margin-top: -25px; 
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
