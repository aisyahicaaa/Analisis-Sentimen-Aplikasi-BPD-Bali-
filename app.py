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
    background:#0B6B3A;
}

.block-container{
    max-width:760px;
    padding-top:60px;
    padding-bottom:40px;
}

/* Text Area */
.stTextArea textarea{
    background:white !important;
    color:black !important;
    border-radius:12px !important;
    font-size:18px;
}

/* Tombol */
.stButton > button{
    width:100%;
    height:55px;
    background:white;
    color:#0B6B3A;
    font-size:20px;
    font-weight:bold;
    border-radius:10px;
    border:none;
}

.stButton > button:hover{
    background:#efefef;
    color:#0B6B3A;
}

</style>
""", unsafe_allow_html=True)

# ======================================
# LOAD MODEL
# ======================================

with open("vectorizer.pkl","rb") as f:
    vectorizer = pickle.load(f)

with open("chi_selector.pkl","rb") as f:
    chi_selector = pickle.load(f)

with open("model_naive_bayes.pkl","rb") as f:
    model = pickle.load(f)

# ======================================
# HEADER
# ======================================

# Jarak dari atas
st.markdown("<div style='height:25px'></div>", unsafe_allow_html=True)

# Logo di tengah
col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.image("logo-bank-bpd-bali.png", width=190)

# Jarak logo ke judul
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

st.markdown("""
<h1 style="
text-align:center;
color:white;
font-size:58px;
font-weight:bold;
margin-bottom:5px;">
Analisis Sentimen
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style="
text-align:center;
color:white;
font-size:22px;
margin-bottom:40px;">
Ulasan Aplikasi BPD Bali Mobile
</p>
""", unsafe_allow_html=True)

# ======================================
# INPUT
# ======================================

st.markdown("""
<p style="
color:white;
font-size:24px;
font-weight:bold;
margin-bottom:10px;">
Masukkan Ulasan
</p>
""", unsafe_allow_html=True)

ulasan = st.text_area(
    "",
    height=150,
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

        st.divider()

        if prediksi == 1:
            st.success("😊 Sentimen Positif")
        else:
            st.error("😞 Sentimen Negatif")

        with st.expander("Hasil Preprocessing"):
            st.write(hasil)
