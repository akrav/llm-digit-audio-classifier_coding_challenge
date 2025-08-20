import os
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf
from sklearn.preprocessing import StandardScaler

TARGET_SAMPLE_RATE = 8000


def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
	"""Load an audio file and resample it to TARGET_SAMPLE_RATE.

	Args:
		file_path: Path to a .wav audio file.

	Returns:
		A tuple of (audio, sample_rate), where audio is a 1D float32 numpy array in range [-1, 1]
		and sample_rate is TARGET_SAMPLE_RATE.

	Raises:
		FileNotFoundError: If file_path does not exist.
		ValueError: If loading fails or produces empty audio.
	"""
	if not os.path.isfile(file_path):
		raise FileNotFoundError(f"Audio file not found: {file_path}")

	# Prefer soundfile backend to avoid audioread issues on newer Python versions
	audio, orig_sr = sf.read(file_path, dtype='float32', always_2d=False)
	if audio is None or (isinstance(audio, np.ndarray) and audio.size == 0):
		raise ValueError(f"Loaded empty audio from: {file_path}")

	# If multi-channel, convert to mono by averaging channels
	if isinstance(audio, np.ndarray) and audio.ndim > 1:
		audio = np.mean(audio, axis=1).astype(np.float32, copy=False)

	# Resample if needed
	if orig_sr != TARGET_SAMPLE_RATE:
		audio = librosa.resample(y=audio, orig_sr=orig_sr, target_sr=TARGET_SAMPLE_RATE)

	# Ensure dtype and numeric bounds are sensible
	audio = audio.astype(np.float32, copy=False)
	max_abs = np.max(np.abs(audio))
	if not np.isfinite(max_abs):
		raise ValueError("Audio contains non-finite values after loading")

	return audio, TARGET_SAMPLE_RATE


def extract_mfcc_features(audio: np.ndarray, sr: int, n_mfcc: int = 40) -> np.ndarray:
	"""Extract MFCC features from 1D audio.

	Args:
		audio: Mono audio signal as float32 numpy array.
		sr: Sample rate of the audio.
		n_mfcc: Number of MFCC coefficients to compute.

	Returns:
		MFCC feature array of shape (n_mfcc, num_frames), dtype float32.
	"""
	if audio is None or (isinstance(audio, np.ndarray) and audio.size == 0):
		raise ValueError("Audio is empty; cannot extract MFCC features")
	if sr <= 0:
		raise ValueError("Sample rate must be positive")

	# Ensure 1D mono
	if isinstance(audio, np.ndarray) and audio.ndim > 1:
		audio = np.mean(audio, axis=1)
	audio = audio.astype(np.float32, copy=False)

	mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
	return mfcc.astype(np.float32, copy=False)


def pad_features(mfcc_array: np.ndarray, max_len: int) -> np.ndarray:
	"""Pad or truncate MFCC array to a fixed number of frames.

	Args:
		mfcc_array: MFCCs shaped (n_mfcc, num_frames).
		max_len: Desired number of frames after padding/truncation.

	Returns:
		Array shaped (n_mfcc, max_len), zero-padded at the end if needed.
	"""
	if mfcc_array is None or mfcc_array.size == 0:
		raise ValueError("MFCC array is empty; cannot pad")
	if max_len <= 0:
		raise ValueError("max_len must be positive")
	if mfcc_array.ndim != 2:
		raise ValueError("mfcc_array must have shape (n_mfcc, num_frames)")

	n_mfcc, num_frames = mfcc_array.shape
	if num_frames == max_len:
		return mfcc_array.astype(np.float32, copy=False)
	elif num_frames > max_len:
		return mfcc_array[:, :max_len].astype(np.float32, copy=False)
	else:
		pad_width = max_len - num_frames
		padded = np.pad(mfcc_array, ((0, 0), (0, pad_width)), mode='constant', constant_values=0.0)
		return padded.astype(np.float32, copy=False)


def normalize_features(features: np.ndarray) -> np.ndarray:
	"""Normalize features to ~0 mean and ~1 std per coefficient.

	Treats time frames as samples and MFCC coefficients as features.

	Args:
		features: Array shaped (n_mfcc, num_frames).

	Returns:
		Array of same shape, normalized per coefficient.
	"""
	if features is None or features.size == 0:
		raise ValueError("features array is empty; cannot normalize")
	if features.ndim != 2:
		raise ValueError("features must have shape (n_mfcc, num_frames)")

	# Transpose to (num_frames, n_mfcc) for sklearn
	X = features.T.astype(np.float32, copy=False)
	scaler = StandardScaler()
	Xn = scaler.fit_transform(X)
	return Xn.T.astype(np.float32, copy=False) 