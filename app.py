import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Load Model (Pastikan nama file .keras sama dengan yang ada di GitHub)
# Jika file Anda bernama 'model_klasifikasibuah.keras', biarkan seperti ini.
model = tf.keras.models.load_model('model_klasifikasibuah.keras')

# 2. Daftar Label (SESUAIKAN URUTANNYA dengan hasil training Anda)
labels = ['Apple', 'Banana', 'avocado', 'cherry', 'kiwi', 'mango', 'orange', 'pinenapple', 'strawberries', 'watermelon']

# Tampilan UI
st.set_page_config(page_title="Fruit Scanner AI", page_icon="🍎")
st.title("🍎 Identifikasi Nama Buah")
st.write("Unggah foto buah, dan AI akan mendeteksi jenisnya.")

# Upload File
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membuka dan menampilkan gambar
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Gambar yang diunggah', use_container_width=True)
    
    # Preprocessing Gambar
    # Pastikan ukuran (96, 96) sama dengan saat Anda melatih model di Colab
    img = image.resize((96, 96)) 
    img_array = tf.keras.utils.img_to_array(img)
    img_bat = np.expand_dims(img_array, axis=0) # Tambahkan dimensi batch (1, 96, 96, 3)

    # Melakukan Prediksi
    with st.spinner('Sedang menganalisis...'):
        predictions = model.predict(img_bat)
        score = tf.nn.softmax(predictions[0]) # Mengubah output menjadi probabilitas
        
    # Menampilkan Hasil
    hasil = labels[np.argmax(score)]
    presentase = 100 * np.max(score)
    
    st.success(f"Hasil Prediksi: **{hasil}**")
    st.info(f"Tingkat Keyakinan: **{presentase:.2f}%**")
