import joblib
import streamlit as st

model = joblib.load("models/model_logistic_regression.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

st.title("Aplikasi Analisis Sentimen Komentar KEVIN ABOX")
st.write("Aplikasi ini digunakan untuk memprediksi apakah sebuah komentar bernada *positif* atau *negatif* menggunakan model Logistic Regression yang sudah dilatih sebelumnya.")

# Input dari user
komentar = st.text_input("Masukkan komentar yang ingin dianalisis")

# Tombol submit
if st.button("Submit"):
    if komentar.strip() == "":
        st.warning("Jangan Kosong Bro Komentarnya !!.")
    else:
        # Transform dan prediksi
        vector = tfidf.transform([komentar])
        prediksi = model.predict(vector)[0]

        label_map = {
            0: "Negatif",
            1: "Positif"
        }

        hasil = label_map.get(prediksi, "Tidak Dikenal")

        st.subheader("Hasil Analisis Sentimen")
        st.write(f"*Komentar:* {komentar}")
        st.write(f"*Prediksi Sentimen:* {hasil}")