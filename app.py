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
# MODERN CSS UI/UX ENHANCEMENT
# ======================================
st.markdown("""
<style>
/* Latar belakang utama aplikasi yang bersih dan modern */
.stApp {
    background-color: #F8FAFC;
}

/* Menyembunyikan padding default Streamlit */
.block-container {
    max-width: 680px;
    padding-top: 40px;
    padding-bottom: 40px;
}

/* Pembungkus Utama berbentuk Kartu (Card Container) */
.main-card {
    background-color: #FFFFFF;
    padding: 40px;
    border-radius: 20px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
    border: 1px solid #E2E8F0;
    margin-bottom: 25px;
}

/* Kustomisasi Text Area agar terlihat minimalis */
.stTextArea textarea {
    background-color: #F8FAFC !important;
    color: #1E293B !important;
    font-size: 16px !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 12px !important;
    padding: 15px !important;
    transition: all 0.3s ease;
}

.stTextArea textarea:focus {
    border-color: #0B6B3A !important;
    box-shadow: 0 0 0 3px rgba(11, 107, 58, 0.15) !important;
}

/* Desain Tombol Utama (BPD Bali Green Theme) */
.stButton > button {
    width: 100%;
    height: 52px;
    background-color: #0B6B3A !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 12px !important;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 4px 6px -1px rgba(11, 107, 58, 0.2);
}

.stButton > button:hover {
    background-color: #09542D !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 12px -1px rgba(11, 107, 58, 0.3);
}

.stButton > button:active {
    transform: translateY(1px);
}

/* Menyesuaikan style Expander hasil preprocessing */
.streamlit-expanderHeader {
    background-color: #F1F5F9 !important;
    border-radius: 8px !important;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ======================================
# LOAD MODEL (Backend Tetap Sama)
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
    st.error("Gagal memuat file model (.pkl). Pastikan file model berada di direktori yang sama.")

# ======================================
# TAMPILAN ELEMEN KARTU UTAMA (UI)
# ======================================

# Membuka div kontainer kartu
st.markdown('<div class="main-card">', unsafe_allow_html=True)

# Logo di tengah secara presisi
kiri, tengah, kanan = st.columns([1, 1.8, 1])
with tengah:
    st.image("logo-bank-bpd-bali.png", use_container_width=True)

# Judul Utama dan Sub-judul dengan warna kontras yang ramah mata
st.markdown("""
<h1 style="
    text-align:center;
    color:#1E293B;
    font-size:38px;
    font-weight:800;
    margin-top:15px;
    margin-bottom:0px;
    letter-spacing: -0.5px;">
    Analisis Sentimen
</h1>
<p style="
    text-align:center;
    color:#64748B;
    font-size:16px;
    margin-top:5px;
    margin-bottom:35px;
    font-weight: 400;">
    Ulasan Aplikasi BPD Bali Mobile
</p>
""", unsafe_allow_html=True)

# Label Input Ulasan
st.markdown("""
<p style="
    font-size:15px;
    font-weight:600;
    color:#334155;
    margin-bottom:8px;">
    Masukkan Ulasan Pengguna
</p>
""", unsafe_allow_html=True)

ulasan = st.text_area(
    "",
    height=140,
    placeholder="Contoh: Aplikasi sangat membantu dan transfernya cepat sekali.",
    label_visibility="collapsed" # Menyembunyikan label bawaan streamlit yang kosong
)

st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

# Tombol Eksekusi
if st.button("🔍 Analisis Sentimen"):
    if ulasan.strip() == "":
        st.toast("⚠️ Masukkan ulasan terlebih dahulu!", icon="ℹ️")
    else:
        # Proses Analisis
        hasil = preprocessing(ulasan)
        vector = vectorizer.transform([hasil])
        vector = chi_selector.transform(vector)
        prediksi = model.predict(vector)[0]
        
        # Area Output Hasil Analisis
        st.markdown("<hr style='border: 0; border-top: 1px dashed #E2E8F0; margin: 25px 0;'>", unsafe_allow_html=True)
        
        if prediksi == 1:
            st.markdown("""
            <div style="background-color: #DCFCE7; border-left: 5px solid #16A34A; padding: 15px; border-radius: 8px;">
                <span style="color: #14532D; font-size: 18px; font-weight: bold;">😊 Sentimen Terdeteksi: POSITIF</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #FEE2E2; border-left: 5px solid #DC2626; padding: 15px; border-radius: 8px;">
                <span style="color: #7F1D1D; font-size: 18px; font-weight: bold;">😞 Sentimen Terdeteksi: NEGATIF</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("✨ Lihat Teks Hasil Preprocessing"):
            st.info(f"**Hasil bersih:** {hasil}")

# Menutup div kontainer kartu
st.markdown('</div>', unsafe_allow_html=True)

# Footer Hak Cipta Halus
st.markdown("""
<p style="text-align: center; color: #94A3B8; font-size: 12px; margin-top: 10px;">
    © 2026 Analisis Sentimen BPD Bali Mobile Classifier
</p>
""", unsafe_allow_html=True)
