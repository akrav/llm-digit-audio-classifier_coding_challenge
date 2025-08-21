import os
import argparse
import numpy as np
import joblib
import librosa
import time

from src.data_processing import load_audio, preprocess_audio_to_features, TARGET_SAMPLE_RATE


def _pool_feat(feat: np.ndarray, pool: str) -> np.ndarray:
	if pool == "mean":
		return feat.mean(axis=1, keepdims=False)[np.newaxis, :]
	elif pool == "max":
		return feat.max(axis=1, keepdims=False)[np.newaxis, :]
	elif pool == "flat":
		return feat.reshape(1, -1)
	else:
		raise ValueError("Unknown pooling mode")


def _load_hf_audio(split: str, index: int) -> tuple[np.ndarray, int]:
	from datasets import load_dataset
	from datasets.features import Audio as HF_Audio
	# Load and decode
	ds = load_dataset("mteb/free-spoken-digit-dataset", split=split)
	try:
		ds = ds.cast_column("audio", HF_Audio(decode=True))
	except Exception:
		pass
	ex = ds[int(index)]
	a = ex.get("audio")
	if a is None:
		raise KeyError("Example missing audio")
	# torchcodec path
	if hasattr(a, "get_all_samples"):
		samples = a.get_all_samples()
		try:
			import torch  # noqa: F401
			np_audio = samples.data.numpy()
		except Exception:
			np_audio = samples.data.detach().cpu().numpy()
		if np_audio.ndim == 2 and np_audio.shape[0] > 1:
			np_audio = np.mean(np_audio, axis=0)
		else:
			np_audio = np_audio.squeeze(0)
		sr = int(samples.sample_rate)
		audio = np.asarray(np_audio, dtype=np.float32)
	# dict decode path
	elif isinstance(a, dict) and ("array" in a and "sampling_rate" in a):
		audio = np.asarray(a["array"], dtype=np.float32)
		sr = int(a["sampling_rate"])
	# file path path
	elif isinstance(a, dict) and a.get("path") and os.path.isfile(a["path"]):
		return load_audio(a["path"])  # already resampled
	else:
		raise ValueError("Unsupported audio format from dataset")
	# Resample to target
	if sr != TARGET_SAMPLE_RATE:
		audio = librosa.resample(y=audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE).astype(np.float32)
		sr = TARGET_SAMPLE_RATE
	return audio, sr


def predict_single_audio(audio_path: str, model_path: str, max_len: int = 200, n_fft: int = 512, hop_length: int = 128, add_deltas: bool = True, pool: str = "flat") -> int:
	"""Load baseline pipeline and predict a single digit from an audio file.

	Args:
		audio_path: Path to a .wav file.
		model_path: Path to a saved sklearn Pipeline (joblib).

	Returns:
		Predicted digit (0-9).
	"""
	if not os.path.isfile(audio_path):
		raise FileNotFoundError(f"Audio file not found: {audio_path}")
	if not os.path.isfile(model_path):
		raise FileNotFoundError(f"Model file not found: {model_path}")

	# Load and preprocess
	audio, sr = load_audio(audio_path)
	feat = preprocess_audio_to_features(audio, TARGET_SAMPLE_RATE, max_len=max_len, n_fft=n_fft, hop_length=hop_length, add_deltas=add_deltas, apply_augmentation=False)
	x = _pool_feat(feat, pool)

	pipe = joblib.load(model_path)
	pred = pipe.predict(x)
	return int(pred[0])


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--audio", type=str)
	parser.add_argument("--model", type=str, default="models/baseline_svm.joblib")
	parser.add_argument("--max_len", type=int, default=200)
	parser.add_argument("--n_fft", type=int, default=512)
	parser.add_argument("--hop_length", type=int, default=128)
	parser.add_argument("--add_deltas", action="store_true")
	parser.add_argument("--pool", choices=["mean", "max", "flat"], default="flat")
	parser.add_argument("--hf_idx", type=int, help="Index of HF sample to use instead of --audio")
	parser.add_argument("--hf_split", type=str, default="test")
	parser.add_argument("--warmup", type=int, default=1, help="Number of warm-up runs")
	parser.add_argument("--runs", type=int, default=10, help="Number of timed runs")
	args = parser.parse_args()

	try:
		# Resolve input audio
		if args.audio:
			audio, sr = load_audio(args.audio)
		elif args.hf_idx is not None:
			audio, sr = _load_hf_audio(args.hf_split, args.hf_idx)
		else:
			raise ValueError("Provide --audio or --hf_idx")

		# Precompute features once for fair inference timing
		feat = preprocess_audio_to_features(audio, TARGET_SAMPLE_RATE, max_len=args.max_len, n_fft=args.n_fft, hop_length=args.hop_length, add_deltas=args.add_deltas, apply_augmentation=False)
		x = _pool_feat(feat, args.pool)
		pipe = joblib.load(args.model)

		# Warm-up
		for _ in range(max(0, args.warmup)):
			_ = pipe.predict(x)

		# Timed runs
		times = []
		for _ in range(max(1, args.runs)):
			start = time.perf_counter()
			pred = int(pipe.predict(x)[0])
			end = time.perf_counter()
			times.append(end - start)

		avg_ms = 1000.0 * float(np.mean(times))
		p95_ms = 1000.0 * float(np.percentile(times, 95))
		print(f"Prediction: {pred}")
		print(f"Latency over {len(times)} runs (warmup={args.warmup}): avg={avg_ms:.2f} ms, p95={p95_ms:.2f} ms")
		print(f"All runs (ms): {[round(t*1000.0,2) for t in times]}")
	except Exception as e:
		print(f"Error: {e}")
		raise


if __name__ == "__main__":
	main() 