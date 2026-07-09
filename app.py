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
# CSS
# ======================================

st.markdown("""
<style>

.stApp{
    background-color:#0B6B3A;
}

.block-container{
    max-width:720px;
    padding-top:20px;
    padding-bottom:40px;
}

/* Text Area */

.stTextArea textarea{

    background:white !important;

    color:black !important;

    font-size:18px !important;

    border-radius:12px !important;

}

/* Tombol */

.stButton > button{

    width:100%;

    height:55px;

    background:white;

    color:#0B6B3A;

    font-size:20px;

    font-weight:bold;

    border:none;

    border-radius:10px;

}

.stButton > button:hover{

    background:#F2F2F2;

    color:#0B6B3A;

}

</style>
""", unsafe_allow_html=True)

# ======================================
# LOAD MODEL
# ======================================

with open("vectorizer.pkl","rb") as file:
    vectorizer = pickle.load(file)

with open("chi_selector.pkl","rb") as file:
    chi_selector = pickle.load(file)

with open("model_naive_bayes.pkl","rb") as file:
    model = pickle.load(file)

# ======================================
# HEADER
# ======================================

st.markdown("<br>", unsafe_allow_html=True)

# Logo di tengah
kiri, tengah, kanan = st.columns([1,2,1])

with tengah:
    st.image("logo-bank-bpd-bali.png", width=220)

st.markdown("""
<h1 style="
text-align:center;
color:white;
font-size:56px;
font-weight:bold;
margin-top:10px;
margin-bottom:5px;">
Analisis Sentimen
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style="
text-align:center;
color:white;
font-size:22px;
margin-bottom:45px;">
Ulasan Aplikasi BPD Bali Mobile
</p>
""", unsafe_allow_html=True)

# ======================================
# INPUT
# ======================================

st.markdown("""
<p style="
font-size:22px;
font-weight:bold;
color:white;
margin-bottom:10px;">
Masukkan Ulasan
</p>
""", unsafe_allow_html=True)

ulasan = st.text_area(
    "",
    height=170,
    placeholder="Contoh: Aplikasi sangat membantu dan mudah digunakan."
)

# ======================================
# BUTTON
# ======================================

if st.button("🔍 Analisis Sentimen", use_container_width=True):

    if ulasan.strip() == "":

        st.warning("Masukkan ulasan terlebih dahulu.")

    else:

        hasil = preprocessing(ulasan)

        vector = vectorizer.transform([hasil])

        vector = chi_selector.transform(vector)

        prediksi = model.predict(vector)[0]

        st.markdown("<br>", unsafe_allow_html=True)

        if prediksi == 1:

            st.success("😊 Sentimen Positif")

        else:

            st.error("😞 Sentimen Negatif")

        with st.expander("Lihat Hasil Preprocessing"):

            st.write(hasil)
