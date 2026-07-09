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
# REFINEMENT CSS (FULL GREEN THEME)
# ======================================
st.markdown("""
<style>
/* Background Full Hijau BPD Bali */
.stApp {
    background-color: #0B6B3A;
}

/* Mengatur container utama agar pas di tengah layar */
.block-container {
    max-width: 720px;
    padding-top: 40px;
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
    font-size: 18px !important;
    border-radius: 12px !important;
    padding: 15px !important;
}

/* Merapikan Tombol Analisis */
.stButton > button {
    width: 100%;
    height: 55px;
    background: white !important;
    color: #0B6B3A !important;
    font-size: 20px !important;
    font-weight: bold !important;
    border: none !important;
    border-radius: 10px !important;
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
# HEADER & LOGO (CENTERED)
# ======================================
st.markdown("<br>", unsafe_allow_html=True)

# Membuat 3 kolom dengan proporsi seimbang untuk mengunci logo di tengah
kiri, tengah, kanan = st.columns([1, 2, 1])
with tengah:
    # Menggunakan use_container_width agar ukuran logo proporsional di tengah
    st.image("logo-bank-bpd-bali.png", use_container_width=True)

# Teks Judul Utama
st.markdown("""
<h1 style="
    text-align: center;
    color: white;
    font-size: 56px;
    font-weight: bold;
    margin-top: 20px;
    margin-bottom: 5px;">
    Analisis Sentimen
</h1>
""", unsafe_allow_html=True)

# Teks Sub-judul
st.markdown("""
<p style="
    text-align: center;
    color: white;
    font-size: 22px;
    margin-bottom: 45px;">
    Ulasan Aplikasi BPD Bali Mobile
</p>
""", unsafe_allow_html=True)

# ======================================
# INPUT USER
# ======================================
st.markdown("""
<p style="
    font-size: 22px;
    font-weight: bold;
    color: white;
    margin-bottom: 10px;">
    Masukkan Ulasan
</p>
""", unsafe_allow_html=True)

ulasan = st.text_area(
    "",
    height=170,
    placeholder="Contoh: Aplikasi sangat membantu dan mudah digunakan.",
    label_visibility="collapsed"
)

st.markdown("<br>", unsafe_allow_html=True)

# ======================================
# BUTTON & PROSES PREDIKSI
# ======================================
if st.button("🔍 Analisis Sentimen"):
    if ulasan.strip() == "":
        st.warning("Masukkan ulasan terlebih dahulu.")
    else:
        # Panggil fungsi preprocessing & model
        hasil = preprocessing(ulasan)
        vector = vectorizer.transform([hasil])
        vector = chi_selector.transform(vector)
        prediksi = model.predict(vector)[0]
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Tampilan hasil di atas background hijau (Menggunakan komponen alert bawaan)
        if prediksi == 1:
            st.success("😊 Sentimen Terdeteksi: POSITIF")
        else:
            st.error("😞 Sentimen Terdeteksi: NEGATIF")
            
        # Expander untuk melihat hasil teks bersih
        with st.expander("Lihat Hasil Preprocessing"):
            st.write(hasil) logo dan tulisan jangan terlalu berjarak 
