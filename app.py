import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# --- 1. Konfigurasi Halaman ---
st.set_page_config(page_title="SkinThesia YOLO", page_icon="🔍", layout="wide")

# --- 2. Load Model (Otomatis) ---
@st.cache_resource
def load_model():
    # Mencari file model hasil training
    path = 'runs/detect/train/weights/best.pt'
    if os.path.exists(path):
        print("Model custom 'best.pt' ditemukan dan dimuat.")
        return YOLO(path)
    else:
        print("Model 'best.pt' tidak ditemukan.")
        return None

model = load_model()

# --- 3. Fungsi Helper (Penting!) ---
# Kita buat fungsi ini agar tidak ada duplikasi kode
def process_and_display_image(image_source):
    """
    Menerima gambar (dari Upload atau Webcam),
    menjalankan deteksi YOLO, dan menampilkan hasilnya.
    """
    try:
        # Buka gambar
        img_pil = Image.open(image_source).convert('RGB')
        img_array = np.array(img_pil)

        with st.spinner("AI sedang memindai..."):
            # Prediksi YOLO
            results = model.predict(img_array, conf=0.1)
            res_plotted = results[0].plot() # Gambar kotak
            jumlah = len(results[0].boxes)  # Hitung jumlah

        # Tampilkan hasil berdampingan
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_pil, caption="Foto Asli", use_container_width=True)
        with col2:
            st.image(res_plotted, caption=f"Hasil Deteksi", use_container_width=True)

        st.divider()
        st.subheader("📊 Hasil Diagnosa")
        
        # Tampilkan metrik
        st.metric("Jumlah Jerawat Terdeteksi", f"{jumlah} Titik")
        
        # Logika Penentuan Status
        if jumlah == 0:
            st.success("✅ **Kondisi: BERSIH / NORMAL**")
            st.write("AI tidak menemukan tanda-tanda jerawat aktif pada wajah.")
        
        elif jumlah < 10:
            st.info("⚠️ **Kondisi: JERAWAT RINGAN**")
            st.write(f"Terdeteksi {jumlah} titik jerawat. Jaga kebersihan wajah.")
            
        elif jumlah < 20:
            st.warning("🟠 **Kondisi: JERAWAT SEDANG**")
            st.write(f"Terdeteksi {jumlah} titik jerawat. Perhatikan pola makan dan skincare.")
            
        else:
            st.error("🔴 **Kondisi: JERAWAT PARAH**")
            st.write(f"Terdeteksi {jumlah} titik jerawat. Disarankan berkonsultasi dengan dokter kulit.")
    
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses gambar: {e}")


# --- 4. UI Aplikasi ---
st.title("🔍 Deteksi Jerawat")

if model is None:
    st.error("❌ **Model AI ('best.pt') Tidak Ditemukan!**")
    st.info("Silakan jalankan `python train_yolo.py` di terminal Anda untuk melatih model terlebih dahulu.")
else:
    # Buat dua tab untuk pilihan input
    tab1, tab2 = st.tabs(["📤 **Upload Foto**", "📷 **Live Webcam**"])

    # --- Logika Tab 1: Upload Foto ---
    with tab1:
        st.info("Upload foto wajah yang jelas dan terang untuk hasil terbaik.")
        uploaded_file = st.file_uploader("Upload foto wajah", type=['jpg', 'png', 'jpeg'])
        
        if uploaded_file:
            # Panggil fungsi helper
            process_and_display_image(uploaded_file)

    # --- Logika Tab 2: Live Webcam ---
    with tab2:
        st.info("Arahkan wajah Anda ke kamera dan klik tombol 'Take photo'.")
        # Widget Kamera Streamlit
        webcam_photo = st.camera_input("Ambil Foto dari Webcam")
        
        if webcam_photo:
            # Panggil fungsi helper
            process_and_display_image(webcam_photo)

st.markdown("---")
st.caption("Powered by YOLOv8 (Ultralytics) & Streamlit")