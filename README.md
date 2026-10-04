# Rouen Motion Experiment

## Panduan Instalasi

### 1. Dapatkan Proyek

**Opsi A (Ekstrak ZIP):**
Ekstrak file `.zip` lalu buka terminal di folder tersebut:
```bash
cd rouen-motion-experiment
```

**Opsi B (Clone Git):**
```bash
git clone https://github.com/jelitarahma/rouen-motion-experiment.git
cd rouen-motion-experiment
```

---

### 2. Setup Lingkungan & Dependencies
> **Requirement**: Python $\ge$ 3.9

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -e .
```

**Windows (Command Prompt - CMD):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -e .
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

*(Opsional) Jalankan unit test:*
```bash
pytest tests/ -v
```

---

## Cara Menjalankan

### 1. Run Full Pipeline (Semua Modul Sekaligus)
```bash
python -m experiments.run_pipeline
```

### 2. Run Per Modul (Beserta Komparasinya)
```bash
# 1. Segmentasi (MoG2 vs K-Means)
python -m experiments.run_segmentation --compare-kmeans

# 2. Tracking (Two-Frame vs Multi-Frame)
python -m experiments.run_tracking --compare-twoframe

# 3. Optical Flow (2-Frame vs Multi-Frame)
python -m experiments.run_optical_flow --compare-multiframe

# 4. Interpolasi Frame (Evaluasi vs Ground Truth)
python -m experiments.run_interpolation --alpha 0.5
```

### 3. Ekspor Video Hasil (Opsional)
```bash
# Ekspor video tracking
python -m experiments.export_video --type tracking --fps 15

# Ekspor video optical flow
python -m experiments.export_video --type flow --fps 15
```

---

## Output
Semua gambar visualisasi dan ringkasan metrik kuantitatif (JSON) otomatis tersimpan di folder `outputs/`:
- `outputs/segmentation/` & `outputs/segmentation_summary.json`
- `outputs/tracking/` & `outputs/tracking_summary.json`
- `outputs/optical_flow/` & `outputs/optical_flow_summary.json`
- `outputs/interpolation/` & `outputs/interpolation_summary.json`
