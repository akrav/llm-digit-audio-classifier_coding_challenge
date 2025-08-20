### Troubleshooting Notes (Sprint 1)

- Python environment (PEP 668) prevented system-wide pip installs: created local venv `.venv` and installed deps there.
- librosa audioread/aifc issue on Python 3.13: switched loader to `soundfile` + `librosa.resample`; installed `soundfile`.
- Librosa warning `n_fft=2048 is too large` on very short clips: resolved by increasing test clip duration to 0.3s (or set smaller n_fft).
