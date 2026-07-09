from flask import Flask, render_template, request
import pickle
from preprocessing import preprocessing  # Memastikan file preprocessing.py Anda ada di folder yang sama

app = Flask(__name__)

# ======================================
# LOAD MODEL & SELEKSI FITUR (CHI-SQUARE)
# ======================================
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
    print("PENTING: File .pkl model Anda tidak ditemukan di direktori utama!")

# ======================================
# ROUTE UTAMA APLIKASI
# ======================================
@app.route("/", methods=["GET", "POST"])
def index():
    teks = ""
    hasil = ""
    warna = ""

    if request.method == "POST":
        # Ambil data input dari textarea HTML
        ulasan_user = request.form.get("ulasan")
        
        if ulasan_user and ulasan_user.strip() != "":
            teks = ulasan_user  # Agar teks tidak hilang dari textarea setelah submit
            
            # 1. Tahap Preprocessing Data Teks
            teks_bersih = preprocessing(ulasan_user)
            
            # 2. Transformasi TF-IDF Vectorizer
            vector = vectorizer.transform([teks_bersih])
            
            # 3. Reduksi Dimensi Fitur Menggunakan Chi-Square
            vector_chi = chi_selector.transform(vector)
            
            # 4. Prediksi Menggunakan Naïve Bayes Classifier
            prediksi = model.predict(vector_chi)[0]
            
            # 5. Penentuan Label Hasil Klasifikasi Akhir
            if prediksi == 1:
                hasil = "😊 Sentimen Terdeteksi: POSITIF"
                warna = "positif"  # Memanggil class .hasil.positif di CSS
            else:
                hasil = "😞 Sentimen Terdeteksi: NEGATIF"
                warna = "negatif"  # Memanggil class .hasil.negatif di CSS

    # Kirim data ke templates/index.html
    return render_template("index.html", teks=teks, hasil=hasil, warna=warna)

# ======================================
# MENJALANKAN SERVER FLASK
# ======================================
if __name__ == "__main__":
    app.run(debug=True)
