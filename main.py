import streamlit as st
import time
import pandas as pd
import numpy as np
import string
import re
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

def text_preprocessing(ulasan):
    if isinstance(ulasan, list):
        ulasan = ' '.join(map(str, ulasan))

    ulasan = ulasan.lower()
    ulasan = re.sub(r'[^\w\s]', '', ulasan)
    ulasan = re.sub(r'\d+', '', ulasan)
    ulasan = ulasan.encode('ascii', 'ignore').decode('ascii')
    ulasan = normalize_slang(ulasan)
    ulasan = lemmatize_text(ulasan)
    ulasan = word_tokenize(ulasan)
    ulasan = remove_stopwords(ulasan)  # Menggunakan daftar kata langsung
    return ulasan

def normalize_slang(ulasan):
    slang_dict = pd.read_csv('slang_dict.csv', sep=';')
    slang_vocab = dict(zip(slang_dict['Informal Vocab'], slang_dict['Formal Vocab']))
    words = ulasan.split()
    normalized_words = [slang_vocab.get(word, word) for word in words]
    normalized_text = " ".join(normalized_words)
    return normalized_text

def lemmatize_text(ulasan):
    lemmatizer = WordNetLemmatizer()
    words = ulasan.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    lemmatized_text = " ".join(lemmatized_words)
    return lemmatized_text

def remove_stopwords(words):
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)  # Ubah kembali menjadi teks

def text_tfidf(ulasan, vectorizer):
    if isinstance(ulasan, list):
        ulasan = ' '.join(map(str, ulasan))
    train_vectors = vectorizer.transform([ulasan])
    feature_names = vectorizer.get_feature_names_out()
    dense = train_vectors.todense()
    denselist = dense.tolist()
    dftfidf = pd.DataFrame(denselist, columns=feature_names)
    return dftfidf

aspek_model = pickle.load(open('aspect_classifier.sav', 'rb'))
sentimen_model = pickle.load(open('sentiment_classifier.sav', 'rb'))
vectorizer = pickle.load(open('vectorizer.sav', 'rb'))

st.title("Sentiment Classification Web App")

# Input teks dari pengguna
ulasan_input = st.text_area("Enter your review:")

# Tombol untuk membuat prediksi
if st.button("Predict Sentiment"):
    # Membuat prediksi sentimen
    hasil_preprocessing = text_preprocessing(ulasan_input)
    ulasan_tfidf = text_tfidf(hasil_preprocessing, vectorizer)
    hasil_prediksi_aspek = aspek_model.predict(ulasan_tfidf).tolist()[0]
    hasil_prediksi_sentimen = sentimen_model.predict(ulasan_tfidf).tolist()[0]

    # Menampilkan hasil prediksi
    st.subheader("Prediction Result:")
    with st.spinner('Wait for it...'):
        time.sleep(3)
    result_text = f"The aspect is: {hasil_prediksi_aspek}\n\nThe sentiment is: {hasil_prediksi_sentimen}"    
    st.info(result_text)