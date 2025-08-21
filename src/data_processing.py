import os
from typing import Tuple, List, Sequence

import librosa
import numpy as np
import soundfile as sf
from sklearn.preprocessing import StandardScaler

TARGET_SAMPLE_RATE = 8000

# Dataset repo id for HF hub downloads
HF_FSDD_REPO = "mteb/free-spoken-digit-dataset"


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


def extract_mfcc_features(
	audio: np.ndarray,
	sr: int,
	n_mfcc: int = 40,
	n_fft: int = 512,
	hop_length: int = 128,
	add_deltas: bool = False,
) -> np.ndarray:
	"""Extract MFCC (optionally with deltas) from 1D audio using params suited for 8 kHz.

	Returns shape (n_mfcc[, * (1 or 3)], num_frames) as float32.
	"""
	if audio is None or (isinstance(audio, np.ndarray) and audio.size == 0):
		raise ValueError("Audio is empty; cannot extract MFCC features")
	if sr <= 0:
		raise ValueError("Sample rate must be positive")

	# Ensure 1D mono
	if isinstance(audio, np.ndarray) and audio.ndim > 1:
		audio = np.mean(audio, axis=1)
	audio = audio.astype(np.float32, copy=False)

	mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
	if add_deltas:
		d1 = librosa.feature.delta(mfcc)
		d2 = librosa.feature.delta(mfcc, order=2)
		mfcc = np.concatenate([mfcc, d1, d2], axis=0)
	return mfcc.astype(np.float32, copy=False)


def pad_features(mfcc_array: np.ndarray, max_len: int) -> np.ndarray:
	"""Pad or truncate MFCC array to a fixed number of frames.

	Args:
		mfcc_array: MFCCs shaped (n_feats, num_frames).
		max_len: Desired number of frames after padding/truncation.

	Returns:
		Array shaped (n_feats, max_len), zero-padded at the end if needed.
	"""
	if mfcc_array is None or mfcc_array.size == 0:
		raise ValueError("MFCC array is empty; cannot pad")
	if max_len <= 0:
		raise ValueError("max_len must be positive")
	if mfcc_array.ndim != 2:
		raise ValueError("mfcc_array must have shape (n_feats, num_frames)")

	n_feats, num_frames = mfcc_array.shape
	if num_frames == max_len:
		return mfcc_array.astype(np.float32, copy=False)
	elif num_frames > max_len:
		return mfcc_array[:, :max_len].astype(np.float32, copy=False)
	else:
		pad_width = max_len - num_frames
		padded = np.pad(mfcc_array, ((0, 0), (0, pad_width)), mode='constant', constant_values=0.0)
		return padded.astype(np.float32, copy=False)


def normalize_features(features: np.ndarray) -> np.ndarray:
	"""Normalize features to ~0 mean and ~1 std per coefficient (over time frames)."""
	if features is None or features.size == 0:
		raise ValueError("features array is empty; cannot normalize")
	if features.ndim != 2:
		raise ValueError("features must have shape (n_feats, num_frames)")

	# Transpose to (num_frames, n_feats) for sklearn
	X = features.T.astype(np.float32, copy=False)
	scaler = StandardScaler()
	Xn = scaler.fit_transform(X)
	return Xn.T.astype(np.float32, copy=False)


def apply_specaugment(mfcc: np.ndarray, num_time_masks: int = 1, num_freq_masks: int = 1, time_mask_width: int = 20, freq_mask_width: int = 4) -> np.ndarray:
	"""Apply simple SpecAugment-style time and frequency masking on MFCCs.

	mfcc: (n_feats, T)
	"""
	aug = mfcc.copy()
	n_feats, T = aug.shape
	if num_time_masks > 0 and T > 0:
		for _ in range(num_time_masks):
			w = min(time_mask_width, T)
			start = np.random.randint(0, max(1, T - w + 1))
			aug[:, start:start + w] = 0.0
	if num_freq_masks > 0 and n_feats > 0:
		for _ in range(num_freq_masks):
			w = min(freq_mask_width, n_feats)
			start = np.random.randint(0, max(1, n_feats - w + 1))
			aug[start:start + w, :] = 0.0
	return aug


def preprocess_audio_to_features(
	audio: np.ndarray,
	sr: int,
	max_len: int = 200,
	n_fft: int = 512,
	hop_length: int = 128,
	add_deltas: bool = False,
	apply_augmentation: bool = False,
) -> np.ndarray:
	"""Full preprocessing for one audio clip: MFCC -> pad -> (optional SpecAugment) -> normalize.

	Returns array of shape (n_feats, max_len).
	"""
	mfcc = extract_mfcc_features(audio, sr, n_mfcc=40, n_fft=n_fft, hop_length=hop_length, add_deltas=add_deltas)
	mfcc_padded = pad_features(mfcc, max_len=max_len)
	if apply_augmentation:
		mfcc_padded = apply_specaugment(mfcc_padded)
	mfcc_norm = normalize_features(mfcc_padded)
	return mfcc_norm


def generate_dataset_from_filepaths(filepaths: Sequence[str], labels: Sequence[int], max_len: int = 200, n_fft: int = 512, hop_length: int = 128, add_deltas: bool = False, apply_augmentation: bool = False) -> Tuple[np.ndarray, np.ndarray]:
	"""Process a list of local filepaths into features and labels arrays.

	Returns X shape (N, n_feats, max_len), y shape (N,).
	"""
	if len(filepaths) != len(labels):
		raise ValueError("filepaths and labels must have the same length")

	features_list: List[np.ndarray] = []
	for p in filepaths:
		audio, sr = load_audio(p)
		features_list.append(preprocess_audio_to_features(audio, sr, max_len=max_len, n_fft=n_fft, hop_length=hop_length, add_deltas=add_deltas, apply_augmentation=apply_augmentation))

	X = np.stack(features_list, axis=0).astype(np.float32)
	y = np.asarray(labels, dtype=np.int64)
	return X, y


def _hf_download_audio(filename: str) -> str:
	"""Download an audio file from the HF dataset repo and return local path."""
	from huggingface_hub import hf_hub_download
	return hf_hub_download(repo_id=HF_FSDD_REPO, filename=filename, repo_type="dataset")


def load_fsdd_from_hf(split: str = "train", max_len: int = 200, n_fft: int = 512, hop_length: int = 128, add_deltas: bool = False, apply_augmentation: bool = False) -> Tuple[np.ndarray, np.ndarray]:
	"""Load FSDD from Hugging Face and preprocess into arrays.

	Dataset: mteb/free-spoken-digit-dataset
	Split: 'train' or 'test'
	"""
	try:
		from datasets import load_dataset
		from datasets.features import Audio
	except Exception as e:
		raise RuntimeError("Please install the 'datasets' package to load from Hugging Face.") from e

	ds = load_dataset("mteb/free-spoken-digit-dataset", split=split)
	# Prefer decoding to arrays now that torchcodec is available; keep path fallback as secondary
	if "audio" in ds.features and isinstance(ds.features["audio"], Audio.__mro__[0]):
		ds = ds.cast_column("audio", Audio(decode=True))

	features_list: List[np.ndarray] = []
	labels_list: List[int] = []

	for ex in ds:
		# Label
		label = None
		for key in ("label", "digit", "class", "target"):
			if key in ex:
				label = int(ex[key])
				break
		if label is None:
			raise KeyError("Could not find label field in dataset example")

		# Prefer decoded samples via torchcodec
		if "audio" in ex and hasattr(ex["audio"], "get_all_samples"):
			samples = ex["audio"].get_all_samples()
			# samples.data is a torch.Tensor shaped (channels, num_samples)
			try:
				import torch  # noqa: F401
				np_audio = samples.data.numpy()
			except Exception:
				np_audio = samples.data.detach().cpu().numpy()
			# Convert to mono if multi-channel
			if np_audio.ndim == 2 and np_audio.shape[0] > 1:
				np_audio = np.mean(np_audio, axis=0)
			else:
				np_audio = np_audio.squeeze(0)
			sr = int(samples.sample_rate)
			if sr != TARGET_SAMPLE_RATE:
				np_audio = librosa.resample(y=np_audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
			feat = preprocess_audio_to_features(np_audio.astype(np.float32, copy=False), TARGET_SAMPLE_RATE, max_len=max_len, n_fft=n_fft, hop_length=hop_length, add_deltas=add_deltas, apply_augmentation=apply_augmentation and split=="train")
		else:
			# Load via file path; if relative, download via HF hub
			path = None
			if "audio" in ex and isinstance(ex["audio"], dict) and ex["audio"].get("path"):
				path = ex["audio"]["path"]
			elif "path" in ex:
				path = ex["path"]
			if not path:
				raise KeyError("Example missing audio path")
			if not os.path.isabs(path) or not os.path.isfile(path):
				path = _hf_download_audio(os.path.basename(path))
			audio, _ = load_audio(path)
			feat = preprocess_audio_to_features(audio, TARGET_SAMPLE_RATE, max_len=max_len, n_fft=n_fft, hop_length=hop_length, add_deltas=add_deltas, apply_augmentation=apply_augmentation and split=="train")

		features_list.append(feat)
		labels_list.append(label)

	X = np.stack(features_list, axis=0).astype(np.float32)
	y = np.asarray(labels_list, dtype=np.int64)
	return X, y 