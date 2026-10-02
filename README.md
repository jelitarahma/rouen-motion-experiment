# Rouen Motion Experiment

## Setup

```bash
# buat venv
python -m venv venv
.\venv\Scripts\activate

# install dependencies
pip install -e .

# test
pytest tests -v
```

## Cara Run

```bash
# run full pipeline
python -m experiments.run_pipeline

# run per modul
python -m experiments.run_segmentation
python -m experiments.run_segmentation --compare-kmeans

python -m experiments.run_tracking
python -m experiments.run_tracking --compare-twoframe

python -m experiments.run_optical_flow
python -m experiments.run_optical_flow --compare-multiframe

python -m experiments.run_interpolation --alpha 0.5

# export video hasil tracking (opsional)
python -m experiments.export_video --type tracking --fps 15
```

Hasil output (gambar dan json) akan tersimpan di folder `outputs/`.
