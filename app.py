from flask import Flask, render_template, request
import pickle

from preprocessing import preprocessing

# Membuat aplikasi Flask
app = Flask(__name__)

# ==========================
# Load Model
# ==========================
with open("Model/vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

with open("Model/chi_selector.pkl", "rb") as file:
    chi_selector = pickle.load(file)

with open("Model/model_naive_bayes.pkl", "rb") as file:
    model = pickle.load(file)

# ==========================
# Halaman Utama
# ==========================
@app.route("/", methods=["GET", "POST"])
def index():

    hasil = None
    warna = ""
    teks = ""

    if request.method == "POST":

        # Mengambil input dari form
        teks = request.form["ulasan"]

        # Preprocessing
        clean_text = preprocessing(teks)

        # Vectorisasi
        vector = vectorizer.transform([clean_text])

        # Seleksi fitur Chi-Square
        vector = chi_selector.transform(vector)

        # Prediksi
        prediksi = model.predict(vector)[0]

        # Menentukan hasil
        if prediksi == 1:
            hasil = "😊 Sentimen Positif"
            warna = "positif"
        else:
            hasil = "😞 Sentimen Negatif"
            warna = "negatif"

    return render_template(
        "index.html",
        hasil=hasil,
        warna=warna,
        teks=teks
    )

# ==========================
# Menjalankan Flask
# ==========================
if __name__ == "__main__":
    app.run(debug=True)