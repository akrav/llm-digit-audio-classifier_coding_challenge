
### Troubleshooting Notes (Sprint 1 & 2)

- PEP 668 blocked system-wide pip installs. Solution: use local venv `.venv` and install deps there.
- Audio I/O on Python 3.13: prefer `soundfile` + `librosa.resample` to avoid `audioread/aifc` issues.
- Librosa warning `n_fft=2048 is too large` on very short clips: increase test clip duration to 0.3s or set smaller n_fft/hop.
- Keras backend: using Keras 3 with Torch backend to avoid TensorFlow install on macOS. Ensure:
  - `KERAS_BACKEND=torch` is set (tests and `src/model.py` handle this)
  - Install `keras` and `torch` in the venv
- Hugging Face audio decoding:
  - Decoding audio arrays can require `torchcodec` + FFmpeg. If unavailable, training falls back to synthetic data to keep the pipeline runnable.
  - Alternatively, implement path-based WAV loading if dataset provides absolute paths. The `mteb/free-spoken-digit-dataset` entries may not expose direct file paths for all records.
- Keras weight save API requires `.weights.h5` suffix. Fixed by saving to `models/model.weights.h5`.


- Training entrypoint may fall back to a synthetic dataset if HF decoding is unavailable.
