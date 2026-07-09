import streamlit as st
import pickle
from preprocessing import preprocessing

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(
    page_title="Analisis Sentimen BPD Bali",
    page_icon="🏦",
    layout="centered"
)

# ==========================
# CSS (DITULIS LANGSUNG DI APP.PY)
# ==========================
st.markdown("""
<style>
.stApp{
    background:#0B6B3A;
}

.block-container{
    max-width:720px;
    padding-top:25px;
}

.stImage{
    display:flex;
    justify-content:center;
}

.stTextArea textarea{
    background:white !important;
    color:black !important;
    border-radius:12px !important;
    font-size:16px;
}

.stButton>button{
    width:100%;
    height:50px;
    background:white;
    color:#0B6B3A;
    border:none;
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#ECECEC;
    color:#0B6B3A;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODEL
# ==========================
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("chi_selector.pkl", "rb") as f:
    chi_selector = pickle.load(f)

with open("model_naive_bayes.pkl", "rb") as f:
    model = pickle.load(f)

# ==========================
# HEADER
# ==========================
st.image("logo-bank-bpd-bali.png", width=120)

st.markdown(
    "<h1 style='text-align:center;color:white;'>Analisis Sentimen</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;color:white;font-size:20px;'>Ulasan Aplikasi BPD Bali Mobile</p>",
    unsafe_allow_html=True
)

st.markdown("### <span style='color:white'>Masukkan Ulasan</span>", unsafe_allow_html=True)

ulasan = st.text_area(
    "",
    height=120,
    placeholder="Contoh: Aplikasi sangat membantu dan mudah digunakan."
)

if st.button("🔍 Analisis Sentimen", use_container_width=True):

    if ulasan.strip() == "":
        st.warning("Masukkan ulasan terlebih dahulu.")

    else:
        hasil = preprocessing(ulasan)

        vector = vectorizer.transform([hasil])
        vector = chi_selector.transform(vector)

        prediksi = model.predict(vector)[0]

        st.divider()

        if prediksi == 1:
            st.success("😊 Sentimen Positif")
        else:
            st.error("😞 Sentimen Negatif")

        with st.expander("Hasil Preprocessing"):
            st.write(hasil)
