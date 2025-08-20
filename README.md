# LLM Digit Audio Classifier

A lightweight pipeline for classifying spoken digits (0–9) from audio using MFCC features and a CNN (in later sprints). This repository is organized in sprints with clear tickets and tests.

## Dataset
- Free Spoken Digit Dataset (FSDD) via Hugging Face: https://huggingface.co/datasets/mteb/free-spoken-digit-dataset/viewer/default/train?views%5B%5D=train

## Quickstart

### 1) Create and activate a virtual environment
```
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```
pip install -r requirements.txt
# If loading from Hugging Face within code:
pip install datasets
```

### 3) Run tests
```
pytest -q
```

## Notable modules
- `src/data_processing.py`
  - `load_audio(file_path)` — loads WAV and resamples to 8 kHz
  - `extract_mfcc_features(audio, sr)` — extracts 40 MFCCs
  - `pad_features(mfcc_array, max_len)` — pads/truncates to fixed length (e.g., 200)
  - `normalize_features(features)` — StandardScaler normalization (~0 mean, ~1 std)
  - `preprocess_audio_to_features(audio, sr, max_len)` — MFCC → pad → normalize
  - `generate_dataset_from_filepaths(paths, labels, max_len)` — batch preprocessing
  - `load_fsdd_from_hf(split, max_len)` — load + preprocess FSDD from Hugging Face

## Notes
- On macOS with Python 3.13, we use `soundfile` for reading WAV and `librosa.resample` to avoid `audioread`/`aifc` issues.
- Tests use synthetic signals; MFCC warnings were resolved by slightly increasing test clip duration to 0.3s.

## Training
Run a brief training and save artifacts:
```
source .venv/bin/activate
pip install datasets huggingface_hub keras torch
python -m src.model  # saves models/model.keras and models/model.weights.h5
```
If HF audio decoding is unavailable, the training entrypoint will fall back to a synthetic dataset to ensure the pipeline runs.
