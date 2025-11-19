from roboflow import Roboflow
import os

# --- KONFIGURASI API KEY ---
# Ganti dengan API Key asli Anda dari: https://app.roboflow.com/settings/api
API_KEY = "JuwgFPVNFIL8iuxkgq9A" 

def download_dataset():
    print("⏳ Sedang menghubungkan ke Roboflow...")
    
    try:
        rf = Roboflow(api_key=API_KEY)
        # Mengambil dataset 'acne04' (Dataset publik yang sudah ada kotak jerawatnya)
        project = rf.workspace("andrei-dore-5lz05").project("acne04")
        version = project.version(1)
        
        print("📥 Sedang mendownload dataset (Gambar + Label Kotak)...")
        dataset = version.download("yolov8")
        
        # Merapikan nama folder agar mudah dipanggil
        # Folder hasil download biasanya bernama 'acne04-1', kita ubah jadi 'datasets'
        if os.path.exists("acne04-1"):
            if os.path.exists("datasets"):
                print("⚠️ Folder 'datasets' sudah ada. Hapus dulu jika ingin download ulang.")
            else:
                os.rename("acne04-1", "datasets")
                print("✅ Sukses! Dataset tersimpan di folder 'datasets/'")
        
        # Perbaikan otomatis file data.yaml (agar path-nya relatif)
        yaml_path = "datasets/data.yaml"
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                lines = f.readlines()
            
            with open(yaml_path, 'w') as f:
                for line in lines:
                    # Hapus baris 'test:' jika ada (kita fokus train/val dulu)
                    if "test:" in line: 
                        continue 
                    # Pastikan path train/val benar
                    if "train:" in line: f.write("train: train/images\n")
                    elif "val:" in line: f.write("val: valid/images\n")
                    # Hapus path absolut yang bikin error
                    elif "path:" not in line: 
                        f.write(line)
            print("✅ File konfigurasi data.yaml berhasil diperbaiki.")

    except Exception as e:
        print(f"❌ Gagal: {e}")
        print("Tips: Pastikan API Key benar dan internet lancar.")

if __name__ == "__main__":
    download_dataset()