from ultralytics import YOLO
import os

def main():
    # 1. Load model dasar (YOLOv8 Nano - Paling ringan & Cepat)
    model = YOLO('yolov8n.pt')

    # 2. Cek path file data.yaml
    # Jika Anda menggunakan skrip download di atas, path-nya biasanya di sini:
    data_path = os.path.join(os.getcwd(), 'datasets', 'data.yaml')
    
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} tidak ditemukan. Jalankan download_data.py dulu.")
        return

    print(f"Mulai training menggunakan dataset di: {data_path}")

    # 3. Mulai Training
    results = model.train(
        data=data_path,   # Lokasi data
        epochs=50,        # Jumlah putaran belajar (bisa ubah jadi 30-50)
        imgsz=640,        # Ukuran gambar
        plots=True        # Simpan grafik hasil
    )
    
    print("Pelatihan Selesai!")
    print("Model terbaik tersimpan di folder 'runs/detect/train/weights/best.pt'")

if __name__ == '__main__':
    main()