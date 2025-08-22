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

## Baseline (CV-optimized)
Add and run the strong, reproducible baseline using SVM with flattened MFCC+deltas and PCA, tuned via stratified CV:

```
source .venv/bin/activate
python -m src.baseline \
  --algo svm \
  --cv \
  --pool flat \
  --add_deltas \
  --pca 128 \
  --n_fft 512 \
  --hop_length 128 \
  --kfolds 5 \
  --C_grid 0.5 1 2 4 8 \
  --gamma_grid scale auto 0.01 0.05 0.1
```

Example outcome (will vary slightly):
```
Best CV score: ~0.969
Best params: {'clf__C': 2.0, 'clf__gamma': 'scale'}
Test accuracy: ~0.973
```

## Baseline artifact for Sprint 3
- Saved model: `models/baseline_svm.joblib`
- Trained with SVM on flattened MFCC+deltas with PCA (see Baseline section above for the exact command).
- Load example:
```
python - << 'PY'
import joblib
pipe = joblib.load('models/baseline_svm.joblib')
print('Loaded baseline:', type(pipe))
PY
```

## Main script
Quick entrypoints using the baseline model:
```
# Train baseline and save artifact
python -m src.main --train

# Evaluate saved baseline (prints metrics and confusion matrix)
python -m src.main --evaluate

# Predict a local file
python -m src.main --predict path/to/audio.wav

# Predict directly from FSDD (HF) test split
python -m src.main --hf_idx 0 --hf_split test
```

## Inference latency
Measure prediction latency (warm-up + repeated runs):
```
python -m src.inference --model models/baseline_svm.joblib \
  --hf_split test --hf_idx 0 --add_deltas --pool flat \
  --warmup 2 --runs 15
```

## Key results
- Baseline SVM (MFCC+deltas, flat, PCA=128): test accuracy ≈ 0.973 (FSDD split)
- Wav2Vec2 (facebook/wav2vec2-base) fine-tuned for digits: eval accuracy ≈ 0.97 after 1 epoch; robust live predictions

## Technical approach (high level)
- Data: FSDD via Hugging Face (train/test), resampled to 8 kHz (MFCC) or 16 kHz (Wav2Vec2)
- Features (baseline): 40 MFCCs (+Δ, ΔΔ), padding to max_len=200, StandardScaler; pooling for classical models
- Baseline models: SVM (RBF) and Logistic Regression with CV tuning; optional PCA
- Deep encoder: Wav2Vec2 encoder fine-tuned for 10-class classification with small classifier head
- Regularization: SpecAugment on MFCCs; waveform noise (audiomentations) for robustness; dataset augmentation options
- Inference: single-file, HF index, latency timing, and live microphone with VAD/segment gating

## Wav2Vec2 training
```
python -m src.wav2vec2_train --epochs 5 --batch_size 16 --lr 5e-5 \
  --freeze_feature_extractor --noise_prob 0.2 \
  --output_dir models/wav2vec2_digits
```

## Wav2Vec2 inference
Single file or HF sample:
```
python -m src.inference --arch wav2vec2 --w2v_dir models/wav2vec2_digits --audio path/to.wav
python -m src.inference --arch wav2vec2 --w2v_dir models/wav2vec2_digits --hf_idx 0
```

## Live microphone
- Baseline MFCC or Wav2Vec2 modes; VAD gating; optional toggle/PTT; one emission per speech segment by default
```
# Wav2Vec2 live (toggle)
python -m src.live_inference --arch wav2vec2 --w2v_dir models/wav2vec2_digits \
  --duration 1.0 --stride 0.2 --capture_sr 8000 \
  --vad --vad_thresh 0.02 --min_speech_ms 400 --hangover_ms 150 \
  --conf_thresh 0.8 --toggle --toggle_key t
```

## LLM collaboration (summary)
- Co-developed the pipeline, iteratively fixed library and environment issues (audio decoding, Keras backend, FFmpeg/torchcodec), and added tests.
- Implemented baseline CV search, latency logging, plots, and HF data loaders with noise augmentation.
- Built live microphone streaming with VAD, confidence gating, segment-level emission, and push-to-talk/toggle modes.
- Added a transformer encoder baseline (Wav2Vec2) for improved live robustness; integrated training and inference into the toolchain.

## Models: Training, Evaluation, Inference, and Live Mic

### Baseline (SVM)
- Split: HF `train` used for fitting; metrics reported on HF `test`. No internal val; use GridSearchCV (`src/baseline.py`) for CV.
- Stratification: yes (StratifiedKFold in grid search)
- Train (example):
```bash
python -m src.baseline --algo svm --pool flat --add_deltas \
  --max_len 200 --n_fft 512 --hop_length 128 \
  --cv --kfolds 5 --C_grid 0.5,1,2,4 --gamma_grid scale,auto \
  --save_path models/baseline_svm.joblib
```
- Evaluate + confusion matrix:
```bash
python -m src.evaluate_baseline --model models/baseline_svm.joblib --add_deltas --pool flat --save_cm
```
- Live mic (no toggle; PTT):
```bash
python -m src.live_inference --arch baseline --model models/baseline_svm.joblib \
  --add_deltas --max_len 200 --n_fft 512 --hop_length 128 \
  --duration 1.0 --vad --vad_thresh 0.02 --min_speech_ms 400 --hangover_ms 200 \
  --conf_thresh 0.8 --ptt --capture_sr 8000
```

### Keras (cs230_cnn)
- Split: stratified train/val (15% val) from HF `train`; metrics on HF `test`.
- CV/Grid: `--cv` for K-fold; `--grid_search` for hyperparam grid (both integrated with StratifiedKFold).
- Train:
```bash
python -m src.train --arch cs230_cnn --epochs 50 --batch_size 32 \
  --add_deltas --n_fft 512 --hop_length 128 --max_len 200 \
  --out models --cache --cache_dir data/cache
```
- 5-fold CV (120 epochs, EarlyStopping):
```bash
python -m src.train --arch cs230_cnn --cv --kfolds 5 --cv_epochs 120 --cv_batch_size 32 \
  --add_deltas --n_fft 512 --hop_length 128 --max_len 200 --out models --cache --cache_dir data/cache
```
- Evaluate + confusion matrix:
```bash
python -m src.evaluate_keras --model models/cs230_cnn.keras --add_deltas --save_cm
```
- Live mic (no toggle; PTT):
```bash
python -m src.live_inference --arch keras --keras_model models/cs230_cnn.keras \
  --add_deltas --max_len 200 --n_fft 512 --hop_length 128 \
  --duration 1.1 --vad --vad_thresh 0.03 --min_speech_ms 500 --hangover_ms 250 \
  --conf_thresh 0.85 --temp 0.7 --force_emit --ptt --capture_sr 8000
```

### Wav2Vec2
- Split: HF `train`/`test`; Trainer handles eval; our evaluator uses HF `test` explicitly.
- Train (example):
```bash
python -m src.wav2vec2_train --model_name facebook/wav2vec2-base \
  --output_dir models/wav2vec2_digits --epochs 10 --batch_size 16 --lr 5e-5
```
- Evaluate + confusion matrix:
```bash
python -m src.evaluate_wav2vec2 --w2v_dir models/wav2vec2_digits --save_cm
```
- Live mic (no toggle; PTT):
```bash
python -m src.live_inference --arch wav2vec2 --w2v_dir models/wav2vec2_digits \
  --duration 1.1 --vad --vad_thresh 0.03 --min_speech_ms 500 --hangover_ms 250 \
  --conf_thresh 0.6 --ptt --capture_sr 16000
```

### Common errors and fixes
- Keras TF backend import: set `KERAS_BACKEND=torch` (handled in code).
- Librosa n_fft warning: ensure clip length ≥ ~0.3s in tests; use n_fft=512, hop=128 at 8 kHz.
- HuggingFace audio decode: install `torchcodec`; or use fallback array extraction (implemented).
- MPS OOM: reduce batch size; or `export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`.
- Live low confidence: increase window/`min_speech_ms`, use `--temp <1` to sharpen, `--force_emit` for segment-level output, adjust `--vad_thresh`.

### Splits summary
- Baseline SVM: HF train → fit (with StratifiedKFold during grid CV); HF test → report.
- Keras: HF train stratified into train/val (15%); HF test → report; supports K-fold CV and grid search integrated.
- Wav2Vec2: HF splits via Trainer/evaluator; evaluator runs on HF test.

## CLI Reference

### Training (unified)
Command:
```bash
python -m src.train_all --arch {baseline|keras|wav2vec2} --mode {train|cv|grid|cvgrid} [options]
```
Common options:
- `--name <str>`: Base name for artifact (prevents overwrites when combined with `--avoid_overwrite`).
- `--avoid_overwrite`: If artifact exists, appends a timestamp.
- `--cleanup`: Deletes the created artifact(s) after verifying they were created.
- `--epochs <int>`: Training epochs (default 1 for smoke tests).
- `--batch_size <int>`: Batch size.

Feature options (MFCC-based models: baseline, keras):
- `--max_len <int>`: Frames after padding/truncation (default 200).
- `--n_fft <int>`: STFT FFT size (default 512 for 8 kHz audio).
- `--hop_length <int>`: STFT hop (default 128).
- `--add_deltas`: Append deltas and delta-deltas to MFCCs.

Cross-validation/grid options:
- `--cv_folds <int>`: K-folds for CV (keras).
- `--gs_folds <int>`: Folds used during grid search (baseline/keras).

Baseline (SVM) grid options:
- `--svm_C_grid <csv>`: Grid for C (e.g., `0.5,1.0,2.0,4.0`).
- `--svm_gamma_grid <csv>`: Grid for gamma (`scale,auto`).

Keras (cs230_cnn) grid options:
- `--keras_lr_grid <csv>`: Learning rate grid (e.g., `0.001,0.0005,0.0001`).

Wav2Vec2 options:
- `--lr <float>`: Learning rate for single run.
- `--w2v2_lr_grid <csv>`: LR grid for W2V2 (e.g., `5e-5,1e-4`).

Examples:
```bash
# Baseline grid + CV
python -m src.train_all --arch baseline --mode grid \
  --svm_C_grid 0.5,1.0,2.0,4.0 --svm_gamma_grid scale,auto \
  --name svm_grid_run --avoid_overwrite

# Keras CV+grid (search LR with K-fold validation)
python -m src.train_all --arch keras --mode cvgrid --epochs 3 --batch_size 32 \
  --keras_lr_grid 0.001,0.0005,0.0001 --cv_folds 5 --gs_folds 5 \
  --name cs230_cvgrid --avoid_overwrite

# Wav2Vec2 grid (select best LR; best copied to base dir)
python -m src.train_all --arch wav2vec2 --mode grid --epochs 1 --batch_size 8 \
  --w2v2_lr_grid 5e-5,1e-4 --name w2v2_grid --avoid_overwrite
```

### Evaluation (unified)
Command:
```bash
python -m src.evaluate --arch {baseline|keras|wav2vec2} --model <path_or_dir> [options]
```
Options:
- `--split <str>`: HF split (`test` default).
- MFCC params for baseline/keras: `--max_len`, `--n_fft`, `--hop_length`, `--add_deltas`.
- `--out <dir>`: Output directory for plots.
- `--save_cm`: Save confusion matrix image.

Examples:
```bash
# Baseline
python -m src.evaluate --arch baseline --model models/baseline_svm.joblib --add_deltas --save_cm
# Keras
python -m src.evaluate --arch keras --model models/custom_tensorflow_cnn.keras --add_deltas --save_cm
# Wav2Vec2
python -m src.evaluate --arch wav2vec2 --model models/wav2vec2_digits --save_cm
```

### Single-file inference
Command:
```bash
python -m src.inference --arch {baseline|keras|wav2vec2} [--audio <wav>|--hf_idx <int>] [options]
```
Key options:
- Baseline: `--model models/baseline_svm.joblib`, pooling with `--pool {flat|mean|max}`.
- Keras: `--keras_model models/custom_tensorflow_cnn.keras`.
- Wav2Vec2: `--w2v_dir models/wav2vec2_digits`.
- Latency: `--warmup <int>`, `--runs <int>` report avg/p95.

Examples:
```bash
python -m src.inference --arch keras --keras_model models/custom_tensorflow_cnn.keras --hf_idx 0 --add_deltas
python -m src.inference --arch baseline --model models/baseline_svm.joblib --audio path/to/file.wav --add_deltas --pool flat
python -m src.inference --arch wav2vec2 --w2v_dir models/wav2vec2_digits --hf_idx 0
```

### Live microphone inference
Command:
```bash
python -m src.live_inference --arch {baseline|keras|wav2vec2} [options]
```
Common options:
- `--ptt`: Push-to-talk (press Enter to capture one window).
- Capture and segmentation: `--capture_sr`, `--duration`, `--stride`.
- VAD/gating: `--vad`, `--vad_thresh`, `--min_speech_ms`, `--hangover_ms`, `--conf_thresh`.
- Temperature and fallback: `--temp <float>` (softmax sharpening; <1 increases confidence), `--force_emit` (emit top-1 even if below threshold at segment end).
- Baseline model path: `--model models/baseline_svm.joblib`.
- Keras model path: `--keras_model models/custom_tensorflow_cnn.keras`.
- Wav2Vec2 dir: `--w2v_dir models/wav2vec2_digits`.

Examples (PTT):
```bash
# Baseline
python -m src.live_inference --arch baseline --model models/baseline_svm.joblib \
  --add_deltas --max_len 200 --n_fft 512 --hop_length 128 \
  --duration 1.0 --vad --vad_thresh 0.02 --min_speech_ms 400 --hangover_ms 200 \
  --conf_thresh 0.8 --ptt --capture_sr 8000

# Keras
python -m src.live_inference --arch keras --keras_model models/custom_tensorflow_cnn.keras \
  --add_deltas --max_len 200 --n_fft 512 --hop_length 128 \
  --duration 1.1 --vad --vad_thresh 0.03 --min_speech_ms 500 --hangover_ms 250 \
  --conf_thresh 0.85 --temp 0.7 --force_emit --ptt --capture_sr 8000

# Wav2Vec2
python -m src.live_inference --arch wav2vec2 --w2v_dir models/wav2vec2_digits \
  --duration 1.1 --vad --vad_thresh 0.03 --min_speech_ms 500 --hangover_ms 250 \
  --conf_thresh 0.6 --ptt --capture_sr 16000
```

## Project Structure (top-level)

```
llm-digit-audio-classifier_coding_challenge/
  ├─ Build Documentation/
  ├─ Cursor Ticket Prompts/
  ├─ Documentation/
  ├─ models/
  ├─ src/
  ├─ tests/
  ├─ Tickets/
  ├─ README.md
  ├─ requirements.txt
```

### Directory guide
- Build Documentation/: Progress logs, troubleshooting notes, structure snapshots.
- Cursor Ticket Prompts/: Executable prompts by sprint/ticket used to drive work.
- Documentation/: Product/architecture docs and assets.
- models/: Trained artifacts, logs, and confusion matrices.
- src/: All source code (data processing, models, training/evaluation/inference/live scripts).
- tests/: Unit tests for data processing and models.
- Tickets/: Sprint ticket descriptions and overviews.

### src/ overview
- data_processing.py: MFCC extraction, padding/normalization, HF dataset loader, augmentation.
- model.py: Keras models (baseline CNN, small CNN, cs230-style network) and helpers.
- baseline.py: SVM/logreg pipeline builder and training entrypoints.
- train.py: Legacy Keras training script (kept for reference).
- wav2vec2_train.py: Wav2Vec2 fine-tuning with HF Trainer.
- train_all.py: Unified trainer for baseline/Keras/Wav2Vec2 with CV/grid.
- evaluate.py: Unified evaluator (accuracy, per-class metrics, confusion matrix).
- inference.py: Single-file inference with latency measurement.
- live_inference.py: Live mic inference with VAD, gating, temperature, force-emit.
