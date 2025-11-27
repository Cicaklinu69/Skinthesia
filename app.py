import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="SkinThesia YOLO", page_icon="🔍", layout="wide")

# ==========================================
# 2. LOAD MODEL (DENGAN PERBAIKAN)
# ==========================================
@st.cache_resource
def load_model():
    path = 'runs/detect/train/weights/best.pt'
    
    if os.path.exists(path):
        # 1. Muat model
        model_yolo = YOLO(path)
        
        # 2. Ubah nama label 'fore' menjadi 'acne' (Trik Ganti Label)
        # Pastikan ini dilakukan DI DALAM fungsi, setelah model dimuat
        if model_yolo.names and 0 in model_yolo.names:
            model_yolo.names[0] = 'acne'
            
        return model_yolo
    else:
        return None

# Panggil fungsi load_model untuk mendapatkan objek model
model = load_model()

# ==========================================
# 3. FUNGSI HELPER (Proses Deteksi)
# ==========================================
def process_and_display(image_input):
    """
    Fungsi ini menerima gambar (dari upload atau kamera),
    melakukan deteksi YOLO, dan menampilkan hasilnya.
    """
    # 1. Konversi ke Array
    img_array = np.array(image_input)
    
    # 2. Buat Kolom Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Foto Asli")
        st.image(image_input, use_container_width=True)

    # 3. Proses Deteksi Otomatis
    with st.spinner("AI sedang memindai wajah..."):
        # Prediksi YOLO
        # conf=0.125 (12.5%) agar lebih sensitif menangkap jerawat kecil
        results = model.predict(img_array, conf=0.125)
        
        # Gambar kotak hasil
        res_plotted = results[0].plot()
        
        # Hitung jumlah
        jumlah_jerawat = len(results[0].boxes)

    with col2:
        st.subheader("Hasil Analisis AI")
        st.image(res_plotted, caption="Lokasi Jerawat Terdeteksi", use_container_width=True)
        
        st.divider()
        
        # Tampilkan Metrik
        st.metric("Jumlah Titik Jerawat", f"{jumlah_jerawat}")

        # Logika Status
        if jumlah_jerawat == 0:
            st.success("✅ **Kondisi: BERSIH / NORMAL**")
            st.write("Tidak ditemukan tanda-tanda jerawat aktif.")
        elif jumlah_jerawat < 10:
            st.info("⚠️ **Kondisi: JERAWAT RINGAN**")
            st.write("Terdeteksi beberapa titik jerawat. Jaga kebersihan wajah.")
        elif jumlah_jerawat < 20:
            st.warning("🟠 **Kondisi: JERAWAT SEDANG**")
            st.write("Cukup banyak titik jerawat. Perhatikan pola makan dan skincare.")
        else:
            st.error("🔴 **Kondisi: JERAWAT PARAH**")
            st.write("Terdeteksi banyak titik jerawat. Disarankan berkonsultasi dengan dokter.")

# ==========================================
# 4. UI APLIKASI UTAMA
# ==========================================
st.title("🔍 Skinthesia")
st.caption("Pilih metode input di bawah ini: Upload File atau Ambil Foto Langsung.")

if model is None:
    st.error("❌ **Model 'best.pt' Belum Ditemukan!**")
    st.warning("Silakan jalankan `python train_yolo.py` terlebih dahulu untuk melatih AI Anda.")
else:
    # --- TAB UNTUK PILIHAN INPUT ---
    tab1, tab2 = st.tabs(["📂 **Upload Foto**", "📸 **Ambil Foto (Kamera)**"])

    # TAB 1: UPLOAD FILE
    with tab1:
        uploaded_file = st.file_uploader("Upload Foto Wajah (JPG/PNG)", type=['jpg', 'png', 'jpeg'])
        if uploaded_file:
            img_pil = Image.open(uploaded_file).convert('RGB')
            process_and_display(img_pil)

    # TAB 2: KAMERA (WEBCAM)
    with tab2:
        st.info("Pastikan wajah terlihat jelas dan pencahayaan cukup.")
        camera_file = st.camera_input("Ambil Foto Wajah")
        if camera_file:
            img_pil = Image.open(camera_file).convert('RGB')
            process_and_display(img_pil)

st.markdown("---")
st.caption("Powered by YOLOv8 Object Detection")
