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
# 2. LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    # Pastikan path ini sesuai dengan lokasi hasil training Anda
    path = 'runs/detect/train/weights/best.pt'
    if os.path.exists(path):
        return YOLO(path)
    else:
        return None

model = load_model()

# ==========================================
# 3. UI APLIKASI (UPLOAD ONLY)
# ==========================================
st.title("🔍 SKINTHESIA")
st.caption("Unggah foto wajah untuk mendeteksi dan menghitung jerawat secara otomatis.")

if model is None:
    st.error("❌ **Model 'best.pt' Belum Ditemukan!**")
    st.warning("Silakan jalankan `python train_yolo.py` terlebih dahulu untuk melatih AI Anda.")
else:
    # Container untuk File Uploader
    with st.container():
        uploaded_file = st.file_uploader("📂 Upload Foto Wajah (JPG/PNG)", type=['jpg', 'png', 'jpeg'])

    # --- LOGIKA UTAMA ---
    if uploaded_file:
        # 1. Buka Gambar
        img_pil = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(img_pil)

        # 2. Buat Kolom untuk Tampilan (Kiri: Asli, Kanan: Hasil)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Foto Asli")
            st.image(img_pil, use_container_width=True)

        # 3. Proses Deteksi Otomatis (Tanpa Tombol)
        with st.spinner("AI sedang memindai wajah..."):
            # conf=0.25: Hanya ambil yg yakin > 25%
            results = model.predict(img_array, conf=0.15)
            
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

            # Logika Status (Bisa Anda sesuaikan batas angkanya)
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

    else:
        # Tampilan awal jika belum ada file
        st.info("👆 Silakan upload foto wajah di atas untuk memulai.")

st.markdown("---")
st.caption("Powered by YOLOv8 Object Detection")
