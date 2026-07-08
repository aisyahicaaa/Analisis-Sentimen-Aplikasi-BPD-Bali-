import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download stopwords (hanya pertama kali)
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')


# Kamus Normalisasi
lexicon_url = "https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv"

lexicon_df = pd.read_csv(lexicon_url)
norm_dict = dict(zip(lexicon_df["slang"], lexicon_df["formal"]))


# Stopword
STOPWORDS = set(stopwords.words("indonesian"))

custom_stopwords = {
    'yang','aja','nya','kok',
    'untuk','dan','di','ke',
    'ini','ya','nih','dong',
    'sih','deh','lah','e'
}

STOPWORDS.update(custom_stopwords)


# Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()


def preprocessing(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Cleaning
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Case Folding
    text = text.lower()

    # Tokenizing
    words = text.split()

    # Normalisasi
    words = [norm_dict.get(word, word) for word in words]

    # Stopword Removal
    words = [word for word in words if word not in STOPWORDS]

    # Stemming
    words = [stemmer.stem(word) for word in words]

    # Gabungkan kembali menjadi kalimat
    return " ".join(words)