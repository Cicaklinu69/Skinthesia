import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import io
import datetime
from supabase import create_client, Client

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="acne detection", page_icon="🔍", layout="wide")

# ==========================================
# 2. LOAD MODEL (DENGAN PERBAIKAN)
# ==========================================
@st.cache_resource
def load_model():
    path = 'runs/detect/train/weights/best.pt'
    
    if os.path.exists(path):
        # 1. Muat model
        model_yolo = YOLO(path)
        
        # 2. Ganti semua nama label yang mungkin muncul (fore, levle, dll) menjadi 'acne'
        if model_yolo.names:
            for idx, name in model_yolo.names.items():
                if name.lower() in ['fore', 'levle', 'levle1', 'levle2', 'levle3', 'level1', 'level2', 'level3']:
                    model_yolo.names[idx] = 'acne'
            
        return model_yolo
    else:
        return None

# Panggil fungsi load_model untuk mendapatkan objek model
model = load_model()

# ==========================================
# 2.5. SUPABASE CONFIGURATION
# ==========================================
# GANTI DENGAN URL & KEY DARI DASHBOARD SUPABASE ANDA
SUPABASE_URL = "https://irrcjylraianrfrbntms.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlycmNqeWxyYWlhbnJmcmJudG1zIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzUyMTc1MTksImV4cCI6MjA5MDc5MzUxOX0.6t1fTVsDfvKyqTAzo4E7Kox2dwLut28AgGwNPrJPIXY"

@st.cache_resource
def get_supabase_client(url, key):
    # Cek jika URL masih berupa placeholder atau belum diisi
    if not url or "supabase.co" not in url:
        return None
    return create_client(url, key)

supabase = get_supabase_client(SUPABASE_URL, SUPABASE_KEY)

def upload_and_save(image_input, acne_count, status_text):
    """
    Mengupload gambar ke Storage dan mencatat ke Database
    """
    if supabase is None:
        st.error("Konfigurasi Supabase belum lengkap. Isi URL dan KEY di app.py.")
        return

    try:
        # 1. Siapkan Gambar (Konversi ke JPEG)
        buf = io.BytesIO()
        image_input.save(buf, format="JPEG")
        image_bytes = buf.getvalue()

        # 2. Upload ke Storage
        filename = f"detection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        
        # PROSES UPLOAD
        try:
            res = supabase.storage.from_("skinthesia-images").upload(
                path=filename,
                file=image_bytes,
                file_options={"content-type": "image/jpeg"}
            )
        except Exception as upload_err:
            st.error(f"❌ Gagal Upload Gambar: {str(upload_err)}")
            return # Berhenti jika upload gagal

        # 3. Ambil Public URL
        public_url = supabase.storage.from_("skinthesia-images").get_public_url(filename)

        # 4. Simpan ke Database (Tabel 'history')
        try:
            data = {
                "image_url": public_url,
                "acne_count": acne_count,
                "status": status_text,
                "created_at": datetime.datetime.now().isoformat()
            }
            supabase.table("history").insert(data).execute()
        except Exception as db_err:
            st.error()
            st.info()

    except Exception as general_err:
        st.error()

# ==========================================
# 3. FUNGSI HELPER (Proses Deteksi)
# ==========================================
def process_and_display(image_input, key_suffix=""):
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

    # --- STATUS & TOMBOL DI LUAR KOLOM (Agar lebih terlihat) ---
    st.divider()
    
    # Logika Status
    if jumlah_jerawat == 0:
        st.success("**Kondisi: BERSIH / NORMAL** - Tidak ditemukan tanda-tanda jerawat aktif.")
        s = "Normal"
    elif jumlah_jerawat < 10:
        st.info("**Kondisi: JERAWAT RINGAN** - Terdeteksi beberapa titik jerawat. Jaga kebersihan wajah.")
        s = "Ringan"
    elif jumlah_jerawat < 20:
        st.warning("**Kondisi: JERAWAT SEDANG** - Cukup banyak titik jerawat. Perhatikan pola makan dan skincare.")
        s = "Sedang"
    else:
        st.error("**Kondisi: JERAWAT PARAH** - Terdeteksi banyak titik jerawat. Disarankan berkonsultasi dengan dokter.")
        s = "Parah"

    # --- OTOMATIS SIMPAN KE SUPABASE ---
    # Gunakan session state supaya tidak upload berulang kali saat halaman refresh atau interaksi widget
    state_key = f"last_uploaded_{key_suffix}"
    
    # Gunakan ID objek sebagai penanda unik untuk gambar saat ini
    current_img_id = id(image_input)
    
    if state_key not in st.session_state or st.session_state[state_key] != current_img_id:
        with st.spinner("..........."):
            upload_and_save(image_input, jumlah_jerawat, s)
            st.session_state[state_key] = current_img_id

# ==========================================
# 4. UI APLIKASI UTAMA
# ==========================================
st.title("🔍 acne detection")
st.caption("Ambil foto wajah Anda untuk analisis kesehatan kulit otomatis.")

if model is None:
    st.error("❌ **Model 'best.pt' Belum Ditemukan!**")
    st.warning("Silakan jalankan `python train_yolo.py` terlebih dahulu untuk melatih AI Anda.")
else:
    # --- INPUT KAMERA UTAMA ---
    st.info("Pastikan wajah terlihat jelas dan pencahayaan cukup.")
    camera_file = st.camera_input("Ambil Foto Wajah")
    if camera_file:
        img_pil = Image.open(camera_file).convert('RGB')
        process_and_display(img_pil, key_suffix="camera")
    
    # --- FITUR UPLOAD (DINONAKTIFKAN SEBAGAI KOMENTAR) ---
    # uploaded_file = st.file_uploader("Upload Foto Wajah (JPG/PNG)", type=['jpg', 'png', 'jpeg'])
    # if uploaded_file:
    #     img_pil = Image.open(uploaded_file).convert('RGB')
    #     process_and_display(img_pil, key_suffix="upload")

st.markdown("---")
st.caption("Powered by YOLOv8 Object Detection")
