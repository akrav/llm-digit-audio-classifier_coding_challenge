import os
import argparse
import numpy as np
from typing import Literal, List

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.decomposition import PCA
import joblib

from src.data_processing import load_fsdd_from_hf


def make_baseline_model(algo: Literal["logreg", "svm"] = "logreg", C: float = 2.0, gamma: str | float = "scale") -> Pipeline:
	if algo == "logreg":
		clf = LogisticRegression(max_iter=4000, C=C, solver="lbfgs", multi_class="auto")
	elif algo == "svm":
		clf = SVC(C=C, kernel="rbf", gamma=gamma, probability=True)
	else:
		raise ValueError(f"Unknown algo: {algo}")
	return Pipeline([
		("scaler", StandardScaler()),
		("clf", clf),
	])


def make_pipeline_with_pca(algo: str, C: float, gamma: str | float, pca_components: int | None) -> Pipeline:
	steps: List[tuple] = [("scaler", StandardScaler())]
	if pca_components and pca_components > 0:
		steps.append(("pca", PCA(n_components=pca_components, random_state=42)))
	else:
		steps.append(("pca", "passthrough"))
	if algo == "logreg":
		clf = LogisticRegression(max_iter=4000, C=C, solver="lbfgs", multi_class="auto")
	elif algo == "svm":
		clf = SVC(C=C, kernel="rbf", gamma=gamma, probability=True)
	else:
		raise ValueError(f"Unknown algo: {algo}")
	steps.append(("clf", clf))
	return Pipeline(steps)


def pool_features(X: np.ndarray, mode: str = "mean") -> np.ndarray:
	"""Pool time dimension of MFCCs: X shape (N, F, T) -> (N, F)."""
	if mode == "mean":
		return X.mean(axis=2)
	elif mode == "max":
		return X.max(axis=2)
	elif mode == "flat":
		N, F, T = X.shape
		return X.reshape(N, F * T)
	else:
		raise ValueError("Unknown pooling mode")


def evaluate_saved(model_path: str, max_len: int, n_fft: int, hop_length: int, add_deltas: bool, pool: str, noisy: bool = False, noise_prob: float = 0.5, noise_bg_dir: str | None = None) -> None:
	# Load test set (optionally noisy)
	X_test, y_test = load_fsdd_from_hf(split="test", max_len=max_len, n_fft=n_fft, hop_length=hop_length, add_deltas=add_deltas, apply_waveform_noise=noisy, noise_prob=noise_prob, noise_bg_dir=noise_bg_dir)
	Xte = pool_features(X_test, mode=pool)
	pipe = joblib.load(model_path)
	pred = pipe.predict(Xte)
	acc = accuracy_score(y_test, pred)
	print(f"Loaded: {model_path}")
	print(f"Evaluate-only test accuracy ({'noisy' if noisy else 'clean'}): {acc:.4f}")
	print("Classification report:\n", classification_report(y_test, pred))
	print("Confusion matrix:\n", confusion_matrix(y_test, pred))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--algo", choices=["logreg", "svm"], default="logreg")
	parser.add_argument("--C", type=float, default=2.0)
	parser.add_argument("--gamma", type=str, default="scale", help="svm gamma; use 'scale', 'auto' or float")
	parser.add_argument("--n_fft", type=int, default=512)
	parser.add_argument("--hop_length", type=int, default=128)
	parser.add_argument("--add_deltas", action="store_true")
	parser.add_argument("--max_len", type=int, default=200)
	parser.add_argument("--pool", choices=["mean", "max", "flat"], default="mean")
	parser.add_argument("--pca", type=int, default=0, help="PCA components (0 disables)")
	parser.add_argument("--cv", action="store_true", help="Enable CV grid search on train set")
	parser.add_argument("--kfolds", type=int, default=5)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--C_grid", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 8.0])
	parser.add_argument("--gamma_grid", type=str, nargs="+", default=["scale", "auto", "0.01", "0.05", "0.1"])
	parser.add_argument("--save_path", type=str, default="", help="Optional path to save fitted model (joblib)")
	parser.add_argument("--evaluate", action="store_true", help="Load a saved model and evaluate on HF test set")
	parser.add_argument("--eval_model_path", type=str, default="models/baseline_svm.joblib")
	parser.add_argument("--eval_noisy", action="store_true", help="Evaluate on noisy test set")
	parser.add_argument("--waveform_noise", action="store_true", help="Apply waveform noise augmentation on training set")
	parser.add_argument("--noise_prob", type=float, default=0.5)
	parser.add_argument("--noise_bg_dir", type=str, default="")
	args = parser.parse_args()

	# Evaluation-only mode
	if args.evaluate:
		evaluate_saved(model_path=args.eval_model_path, max_len=args.max_len, n_fft=args.n_fft, hop_length=args.hop_length, add_deltas=args.add_deltas, pool=args.pool, noisy=args.eval_noisy, noise_prob=args.noise_prob, noise_bg_dir=args.noise_bg_dir or None)
		return

	# Load features (no augmentation for baselines, optional waveform noise for training)
	X_train, y_train = load_fsdd_from_hf(split="train", max_len=args.max_len, n_fft=args.n_fft, hop_length=args.hop_length, add_deltas=args.add_deltas, apply_waveform_noise=args.waveform_noise, noise_prob=args.noise_prob, noise_bg_dir=args.noise_bg_dir or None)
	X_test, y_test = load_fsdd_from_hf(split="test", max_len=args.max_len, n_fft=args.n_fft, hop_length=args.hop_length, add_deltas=args.add_deltas)

	Xtr = pool_features(X_train, mode=args.pool)
	Xte = pool_features(X_test, mode=args.pool)

	# If CV enabled, run grid search on training set and refit best
	if args.cv:
		# Parse gamma possibly numeric in provided grid
		gamma_grid: List[float | str] = []
		for g in args.gamma_grid:
			try:
				gamma_grid.append(float(g))
			except Exception:
				gamma_grid.append(g)

		pipe = make_pipeline_with_pca(args.algo, args.C, args.gamma, args.pca if args.pca > 0 else None)
		param_grid = {}
		if args.algo == "svm":
			param_grid.update({
				"clf__C": args.C_grid,
				"clf__gamma": gamma_grid,
			})
		else:
			param_grid.update({
				"clf__C": args.C_grid,
			})
		cv = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=args.seed)
		gs = GridSearchCV(pipe, param_grid=param_grid, cv=cv, n_jobs=-1, scoring="accuracy", refit=True, verbose=1)
		gs.fit(Xtr, y_train)
		print(f"Best CV score: {gs.best_score_:.4f}")
		print(f"Best params: {gs.best_params_}")
		best = gs.best_estimator_
		pred = best.predict(Xte)
		acc = accuracy_score(y_test, pred)
		print(f"Baseline-CV ({args.algo}, pool={args.pool}, deltas={args.add_deltas}, pca={args.pca}) test accuracy: {acc:.4f}")
		print("Classification report:\n", classification_report(y_test, pred))
		print("Confusion matrix:\n", confusion_matrix(y_test, pred))
		if args.save_path:
			os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
			joblib.dump(best, args.save_path)
			print(f"Saved baseline model to: {args.save_path}")
		return

	# No CV: simple fit/eval
	gamma_val: float | str
	try:
		gamma_val = float(args.gamma)
	except Exception:
		gamma_val = args.gamma
	pipe = make_pipeline_with_pca(args.algo, args.C, gamma_val, args.pca if args.pca > 0 else None)
	pipe.fit(Xtr, y_train)
	pred = pipe.predict(Xte)
	acc = accuracy_score(y_test, pred)
	print(f"Baseline ({args.algo}, C={args.C}, gamma={args.gamma}, pool={args.pool}, deltas={args.add_deltas}, pca={args.pca}) test accuracy: {acc:.4f}")
	print("Classification report:\n", classification_report(y_test, pred))
	print("Confusion matrix:\n", confusion_matrix(y_test, pred))
	if args.save_path:
		os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
		joblib.dump(pipe, args.save_path)
		print(f"Saved baseline model to: {args.save_path}")


if __name__ == "__main__":
	main() 