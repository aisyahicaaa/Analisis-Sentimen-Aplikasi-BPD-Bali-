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
    max-width:720px;
    padding-top:40px;
}

/* Logo */
.logo{
    text-align:center;
    margin-top:20px;
    margin-bottom:10px;
}

/* Judul */
.judul{
    text-align:center;
    color:white;
    font-size:55px;
    font-weight:bold;
    margin-bottom:0px;
}

/* Subtitle */
.subjudul{
    text-align:center;
    color:white;
    font-size:22px;
    margin-top:5px;
    margin-bottom:35px;
}

/* Label */
.label{
    color:white;
    font-size:22px;
    font-weight:bold;
}

/* Text Area */
.stTextArea textarea{
    background:white !important;
    color:black !important;
    border-radius:12px;
    font-size:18px;
}

/* Tombol */
.stButton>button{

    width:100%;

    height:55px;

    background:white;

    color:#0B6B3A;

    font-size:20px;

    font-weight:bold;

    border-radius:10px;

    border:none;

}

.stButton>button:hover{

    background:#ECECEC;

    color:#0B6B3A;

}

</style>
""", unsafe_allow_html=True)

# ======================================
# LOAD MODEL
# ======================================

with open("vectorizer.pkl","rb") as file:
    vectorizer=pickle.load(file)

with open("chi_selector.pkl","rb") as file:
    chi_selector=pickle.load(file)

with open("model_naive_bayes.pkl","rb") as file:
    model=pickle.load(file)

# ======================================
# HEADER
# ======================================

col1,col2,col3=st.columns([1,2,1])

with col2:
    st.image("logo-bank-bpd-bali.png",width=130)

st.markdown(
"""
<h1 class='judul'>
Analisis Sentimen
</h1>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<p class='subjudul'>
Ulasan Aplikasi BPD Bali Mobile
</p>
""",
unsafe_allow_html=True
)

# ======================================
# INPUT
# ======================================

st.markdown(
"""
<p class='label'>
Masukkan Ulasan
</p>
""",
unsafe_allow_html=True
)

ulasan=st.text_area(
"",
height=150,
placeholder="Contoh: Aplikasi sangat membantu dan mudah digunakan."
)

# ======================================
# BUTTON
# ======================================

if st.button("🔍 Analisis Sentimen",use_container_width=True):

    if ulasan.strip()=="":

        st.warning("Masukkan ulasan terlebih dahulu.")

    else:

        hasil=preprocessing(ulasan)

        vector=vectorizer.transform([hasil])

        vector=chi_selector.transform(vector)

        prediksi=model.predict(vector)[0]

        st.divider()

        if prediksi==1:

            st.success("😊 Sentimen Positif")

        else:

            st.error("😞 Sentimen Negatif")

        with st.expander("Hasil Preprocessing"):

            st.write(hasil)
