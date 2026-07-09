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
# MODERN CSS - GREEN THEME REFINEMENT
# ======================================
st.markdown("""
<style>
/* Latar belakang utama Hijau BPD Bali */
.stApp {
    background-color: #0B6B3A;
}

/* Mengatur container agar presisi di tengah */
.block-container {
    max-width: 680px;
    padding-top: 50px;
    padding-bottom: 50px;
}

/* Kartu Putih Bersih di atas Latar Hijau (Membuat konten mudah dibaca) */
.main-card {
    background-color: #FFFFFF;
    padding: 40px;
    border-radius: 24px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 20px;
}

/* Kustomisasi Text Area Input */
.stTextArea textarea {
    background-color: #F8FAFC !important;
    color: #1E293B !important;
    font-size: 16px !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 14px !important;
    padding: 15px !important;
    transition: all 0.3s ease;
}

/* Efek saat kolom input diklik */
.stTextArea textarea:focus {
    border-color: #0B6B3A !important;
    box-shadow: 0 0 0 3px rgba(11, 107, 58, 0.15) !important;
}

/* Tombol Analisis Hijau Solid */
.stButton > button {
    width: 100%;
    height: 54px;
    background-color: #0B6B3A !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 14px !important;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 4px 12px rgba(11, 107, 58, 0.2);
}

/* Efek Hover tombol */
.stButton > button:hover {
    background-color: #09542D !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 15px rgba(11, 107, 58, 0.3);
}

.stButton > button:active {
    transform: translateY(1px);
}

/* Styling Expander Preprocessing */
.streamlit-expanderHeader {
    background-color: #F1F5F9 !important;
    border-radius: 10px !important;
    font-weight: 500;
    color: #334155 !important;
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
    st.error("File model (.pkl) tidak ditemukan. Pastikan posisinya sudah benar.")

# ======================================
# TAMPILAN UI (KARTU UTAMA)
# ======================================

# Membuka pembungkus kartu utama putih
st.markdown('<div class="main-card">', unsafe_allow_html=True)

# Logo Bank BPD Bali di Tengah secara simetris
kiri, tengah, kanan = st.columns([1, 1.8, 1])
with tengah:
    st.image("logo-bank-bpd-bali.png", use_container_width=True)

# Judul Utama & Sub-judul (Warna gelap agar kontras di atas kartu putih)
st.markdown("""
<h1 style="
    text-align:center;
    color:#1E293B;
    font-size:36px;
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
    margin-bottom:35px;">
    Ulasan Aplikasi BPD Bali Mobile
</p>
""", unsafe_allow_html=True)

# Label Input
st.markdown("""
<p style="
    font-size:15px;
    font-weight:600;
    color:#334155;
    margin-bottom:8px;">
    Masukkan Ulasan
</p>
""", unsafe_allow_html=True)

ulasan = st.text_area(
    "",
    height=140,
    placeholder="Contoh: Aplikasi sangat membantu dan mudah digunakan.",
    label_visibility="collapsed"
)

st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

# Eksekusi Tombol
if st.button("🔍 Analisis Sentimen"):
    if ulasan.strip() == "":
        st.toast("⚠️ Tulis ulasan Anda terlebih dahulu!", icon="ℹ️")
    else:
        # Proses Prediksi ML
        hasil = preprocessing(ulasan)
        vector = vectorizer.transform([hasil])
        vector = chi_selector.transform(vector)
        prediksi = model.predict(vector)[0]
        
        st.markdown("<hr style='border: 0; border-top: 1px dashed #E2E8F0; margin: 25px 0;'>", unsafe_allow_html=True)
        
        # Tampilan Hasil Sentimen yang Eye-Catching
        if prediksi == 1:
            st.markdown("""
            <div style="background-color: #DCFCE7; border-left: 5px solid #16A34A; padding: 16px; border-radius: 10px; margin-bottom: 20px;">
                <span style="color: #14532D; font-size: 17px; font-weight: 700;">😊 Sentimen Terdeteksi: POSITIF</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #FEE2E2; border-left: 5px solid #DC2626; padding: 16px; border-radius: 10px; margin-bottom: 20px;">
                <span style="color: #7F1D1D; font-size: 17px; font-weight: 700;">😞 Sentimen Terdeteksi: NEGATIF</span>
            </div>
            """, unsafe_allow_html=True)
        
        with st.expander("✨ Lihat Hasil Preprocessing Teks"):
            st.info(f"**Teks Bersih:** {hasil}")

# Menutup pembungkus kartu utama putih
st.markdown('</div>', unsafe_allow_html=True)

# Footer Hak Cipta di bawah kartu (Warna putih tipis agar estetik di atas background hijau)
st.markdown("""
<p style="text-align: center; color: rgba(255,255,255,0.7); font-size: 12px; margin-top: 15px;">
    © 2026 Analisis Sentimen BPD Bali Mobile Classifier
</p>
""", unsafe_allow_html=True)
