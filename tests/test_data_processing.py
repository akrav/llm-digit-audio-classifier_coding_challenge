import os
import tempfile
import wave
import struct

import numpy as np
import pytest

from src.data_processing import load_audio, TARGET_SAMPLE_RATE, extract_mfcc_features


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
	duration = 0.25
	t = np.linspace(0, duration, int(TARGET_SAMPLE_RATE * duration), endpoint=False)
	signal = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
	mfcc = extract_mfcc_features(signal, TARGET_SAMPLE_RATE)
	assert mfcc.shape[0] == 40
	assert mfcc.shape[1] > 0
	assert mfcc.dtype == np.float32


def test_extract_mfcc_features_raises_on_empty_audio():
	with pytest.raises(ValueError):
		extract_mfcc_features(np.array([], dtype=np.float32), TARGET_SAMPLE_RATE) 