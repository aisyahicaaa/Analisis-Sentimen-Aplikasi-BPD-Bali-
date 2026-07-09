# ======================================
# HEADER
# ======================================

st.image("logo-bank-bpd-bali.png", width=110)

st.markdown(
"""
<h1 style='text-align:center;color:white;margin-bottom:0'>
Analisis Sentimen
</h1>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<p style='text-align:center;color:white;font-size:22px;margin-top:0'>
Ulasan Aplikasi BPD Bali Mobile
</p>
""",
unsafe_allow_html=True
)

st.markdown("---")

st.markdown(
"""
<p style='font-size:20px;
font-weight:bold;
color:white;'>
Masukkan Ulasan
</p>
""",
unsafe_allow_html=True
)

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

        st.markdown("---")

        if prediksi == 1:
            st.success("😊 Sentimen Positif")
        else:
            st.error("😞 Sentimen Negatif")

        with st.expander("Hasil Preprocessing"):
            st.write(hasil)
