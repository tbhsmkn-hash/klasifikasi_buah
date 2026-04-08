import streamlit as st
import tensorflow as tf
from PIL import Image
import nuimport streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Load Model (Pastikan nama file sesuai di GitHub)
model = tf.keras.models.load_model('model_klasifikasibuah.keras')
# Sesuaikan label dengan hasil print(data_cat) saat training
labels = ['Apel', 'Jeruk', 'Pisang', 'Alpukat', 'Lainnya'] 

st.title("🍎 Ridwan AI")

file = st.file_uploader("Unggah foto buah...", type=["jpg", "png", "jpeg"])

if file is not None:
    image = Image.open(file).convert('RGB') # Pastikan format RGB
    st.image(image, caption="Gambar yang dipilih", use_container_width=True)
    
    # --- BAGIAN PERBAIKAN ---
    # 1. Resize gambar tepat ke 96x96 (sesuai training)
    img = image.resize((96, 96)) 
    
    # 2. Ubah ke array dan normalisasi jika diperlukan
    img_array = tf.keras.utils.img_to_array(img)
    
    # 3. Tambahkan dimensi batch (menjadi 1, 96, 96, 3)
    img_bat = tf.expand_dims(img_array, 0)
    
    # 4. Prediksi
    predictions = model.predict(img_bat)
    score = tf.nn.softmax(predictions[0]) # Gunakan softmax untuk probabilitas
    
    hasil = labels[np.argmax(score)]
    akurasi = 100 * np.max(score)
    
    st.success(f"Hasil: **{hasil}** ({akurasi:.2f}%)")mpy as np

st.title("Identifikasi Nama Buah")
st.write("Unggah foto buah, dan AI akan menebak namanya!")

# Load model yang sudah dilatih
model = tf.keras.models.load_model('model_klasifikasibuah.keras')
labels = ['Apple', 'Banana', 'avocado', 'cherry', 'kiwi', 'mango', 'orange', 'pinenapple', 'strawberries', 'watermelon'] # Sesuaikan dengan dataset Anda

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)

    # Pre-processing gambar agar sesuai input model
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    predictions = model.predict(img_array)
    result = labels[np.argmax(predictions)]

    st.success(f"Hasil Identifikasi: **{result}**")
