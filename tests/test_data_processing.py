import os
import tempfile
import wave
import struct

import numpy as np
import pytest

from src.data_processing import load_audio, TARGET_SAMPLE_RATE, extract_mfcc_features, pad_features, normalize_features, preprocess_audio_to_features, generate_dataset_from_filepaths


def _write_wav(path: str, data: np.ndarray, sr: int) -> None:
	# Write mono PCM16 WAV using built-in wave module
	data = np.clip(data, -1.0, 1.0).astype(np.float32, copy=False)
	pcm = (data * 32767.0).astype(np.int16, copy=False)
	with wave.open(path, 'wb') as wf:
		wf.setnchannels(1)
		wf.setsampwidth(2)  # 16-bit
		wf.setframerate(sr)
		wf.writeframes(pcm.tobytes())


def test_load_audio_resamples_to_target_rate():
	with tempfile.TemporaryDirectory() as tmp:
		wav_path = os.path.join(tmp, "tone.wav")
		orig_sr = 16000
		duration = 0.25
		t = np.linspace(0, duration, int(orig_sr * duration), endpoint=False)
		# 440 Hz tone
		signal = 0.1 * np.sin(2 * np.pi * 440 * t)
		_write_wav(wav_path, signal, orig_sr)

		audio, sr = load_audio(wav_path)
		assert sr == TARGET_SAMPLE_RATE
		assert audio.ndim == 1
		assert audio.dtype == np.float32
		assert audio.size > 0


def test_load_audio_raises_on_missing_file():
	with pytest.raises(FileNotFoundError):
		load_audio("/nonexistent/path/file.wav")


def test_extract_mfcc_features_shape():
	# Generate a short sine wave at target SR
	duration = 0.3
	t = np.linspace(0, duration, int(TARGET_SAMPLE_RATE * duration), endpoint=False)
	signal = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
	mfcc = extract_mfcc_features(signal, TARGET_SAMPLE_RATE)
	assert mfcc.shape[0] == 40
	assert mfcc.shape[1] > 0
	assert mfcc.dtype == np.float32


def test_extract_mfcc_features_raises_on_empty_audio():
	with pytest.raises(ValueError):
		extract_mfcc_features(np.array([], dtype=np.float32), TARGET_SAMPLE_RATE)


def test_pad_features_shapes():
	arr = np.random.randn(40, 150).astype(np.float32)
	padded = pad_features(arr, 200)
	assert padded.shape == (40, 200)
	# Truncate case
	truncated = pad_features(arr, 100)
	assert truncated.shape == (40, 100)


def test_normalize_features_stats():
	# Create synthetic MFCCs with known mean/std per coefficient
	rng = np.random.default_rng(0)
	arr = rng.normal(loc=5.0, scale=2.0, size=(40, 120)).astype(np.float32)
	norm = normalize_features(arr)
	# Means close to 0 and std close to 1 per coefficient
	means = norm.mean(axis=1)
	stds = norm.std(axis=1)
	assert np.all(np.isfinite(means)) and np.all(np.isfinite(stds))
	assert np.all(np.abs(means) < 1e-3)
	assert np.all(np.abs(stds - 1.0) < 1e-3)


def test_generate_dataset_from_filepaths_shapes():
	with tempfile.TemporaryDirectory() as tmp:
		paths = []
		labels = []
		for i, freq in enumerate([300, 600, 900]):
			wav_path = os.path.join(tmp, f"sig_{i}.wav")
			duration = 0.3
			t = np.linspace(0, duration, int(16000 * duration), endpoint=False)
			sig = 0.1 * np.sin(2 * np.pi * freq * t)
			_write_wav(wav_path, sig, 16000)
			paths.append(wav_path)
			labels.append(i % 10)
		X, y = generate_dataset_from_filepaths(paths, labels, max_len=200)
		assert X.shape == (3, 40, 200)
		assert y.shape == (3,)
		assert X.dtype == np.float32
		assert y.dtype == np.int64 