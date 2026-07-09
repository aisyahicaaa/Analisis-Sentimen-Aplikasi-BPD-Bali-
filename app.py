import streamlit as st
import pickle
from preprocessing import preprocessing

# ==========================
# LOAD MODEL
# ==========================
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

with open("chi_selector.pkl", "rb") as file:
    chi_selector = pickle.load(file)

with open("model_naive_bayes.pkl", "rb") as file:
    model = pickle.load(file)

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(
    page_title="Analisis Sentimen BPD Bali Mobile",
    page_icon="🏦",
    layout="centered"
)

# ==========================
# CSS
# ==========================
st.markdown("""
<style>

.stApp{
    background-color:#f4f9f4;
}

/* Judul */
h1{
    color:#00843D;
    text-align:center;
    font-weight:bold;
}

/* Subjudul */
h3{
    color:#00843D;
}

/* Tombol */
.stButton>button{
    width:100%;
    background-color:#00843D;
    color:white;
    border-radius:10px;
    border:none;
    height:50px;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background-color:#006b31;
    color:white;
}

/* Text Area */
textarea{
    border-radius:10px !important;
    border:2px solid #00843D !important;
}

/* Kotak hasil */
.result-box{
    background-color:white;
    padding:20px;
    border-radius:10px;
    border-left:8px solid #00843D;
    box-shadow:0px 2px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ==========================
# JUDUL
# ==========================
st.title("🏦 Analisis Sentimen BPD Bali Mobile")

st.write(
"""
Aplikasi ini digunakan untuk menganalisis sentimen ulasan pengguna
**BPD Bali Mobile** .
"""
)

# ==========================
# INPUT
# ==========================
teks = st.text_area(
    "Masukkan Ulasan",
    height=180,
    placeholder="Contoh: Aplikasi sangat membantu dan mudah digunakan."
)

# ==========================
# TOMBOL ANALISIS
# ==========================
if st.button("Analisis Sentimen"):

    if teks.strip() == "":
        st.warning("Silakan masukkan ulasan terlebih dahulu.")

    else:

        # Preprocessing
        clean_text = preprocessing(teks)

        # CountVectorizer
        vector = vectorizer.transform([clean_text])

        # Chi-Square
        vector = chi_selector.transform(vector)

        # Prediksi
        prediksi = model.predict(vector)[0]

        st.markdown("---")

        st.subheader("Hasil Analisis Sentimen")

        if prediksi == 1:
            st.success("😊 Sentimen Positif")
        else:
            st.error("😞 Sentimen Negatif")

        st.markdown(
            f"""
            <div class="result-box">
            <b>Hasil Preprocessing :</b><br><br>
            {clean_text}
            </div>
            """,
            unsafe_allow_html=True
        )
