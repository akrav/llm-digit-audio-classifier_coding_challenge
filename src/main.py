import os
import argparse

from src.baseline import main as baseline_cli
from src.baseline import evaluate_saved
from src.inference import predict_single_audio, _load_hf_audio
from src.data_processing import TARGET_SAMPLE_RATE


def train_baseline_and_save():
	# Delegate to baseline CLI defaults for a quick training+save if needed
	# Users can still run src.baseline directly for full control
	from src.baseline import main as baseline_main
	import sys
	args = [
		"--algo", "svm",
		"--cv",
		"--pool", "flat",
		"--add_deltas",
		"--pca", "128",
		"--n_fft", "512",
		"--hop_length", "128",
		"--kfolds", "5",
		"--C_grid", "2.0",
		"--gamma_grid", "scale",
		"--save_path", "models/baseline_svm.joblib",
	]
	sys.argv = ["baseline"] + args
	baseline_main()


def evaluate_baseline():
	evaluate_saved(
		model_path="models/baseline_svm.joblib",
		max_len=200,
		n_fft=512,
		hop_length=128,
		add_deltas=True,
		pool="flat",
	)


def predict(audio: str | None, hf_idx: int | None, hf_split: str, model: str) -> None:
	if audio:
		pred = predict_single_audio(audio, model_path=model, add_deltas=True, pool="flat")
		print(f"Prediction: {pred}")
		return
	if hf_idx is not None:
		audio_arr, sr = _load_hf_audio(hf_split, hf_idx)
		# predict_single_audio expects a path; reuse inference flow via features if needed
		from src.data_processing import preprocess_audio_to_features
		import joblib
		import numpy as np
		feat = preprocess_audio_to_features(audio_arr, TARGET_SAMPLE_RATE, max_len=200, n_fft=512, hop_length=128, add_deltas=True, apply_augmentation=False)
		x = feat.reshape(1, -1)
		pipe = joblib.load(model)
		pred = int(pipe.predict(x)[0])
		print(f"Prediction: {pred}")
		return
	raise ValueError("Provide --predict <path> or --hf_idx")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--train", action="store_true")
	parser.add_argument("--evaluate", action="store_true")
	parser.add_argument("--predict", type=str)
	parser.add_argument("--hf_idx", type=int)
	parser.add_argument("--hf_split", type=str, default="test")
	parser.add_argument("--model", type=str, default="models/baseline_svm.joblib")
	args = parser.parse_args()

	if args.train:
		train_baseline_and_save()
		return
	if args.evaluate:
		evaluate_baseline()
		return
	if args.predict or args.hf_idx is not None:
		predict(audio=args.predict, hf_idx=args.hf_idx, hf_split=args.hf_split, model=args.model)
		return
	parser.print_help()


if __name__ == "__main__":
	main() 