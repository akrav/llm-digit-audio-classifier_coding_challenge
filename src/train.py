import os
import math
import argparse
import numpy as np
from typing import Tuple, Optional

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from keras import callbacks
from keras.utils import Sequence
from sklearn.model_selection import train_test_split, StratifiedKFold

from src.model import build_cnn_model, build_small_cnn
from src.model import build_cs230_cnn
from src.data_processing import load_fsdd_from_hf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
import random
import csv


def _set_seeds(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	try:
		import torch
		torch.manual_seed(seed)
	except Exception:
		pass


def build_callbacks(out_dir: str, append_csv: bool = False) -> list:
	logs_dir = os.path.join(out_dir, "logs")
	os.makedirs(out_dir, exist_ok=True)
	os.makedirs(logs_dir, exist_ok=True)
	return [
		callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=8, restore_best_weights=True),
		callbacks.ReduceLROnPlateau(monitor="val_accuracy", mode="max", factor=0.5, patience=4, min_lr=1e-5),
		callbacks.ModelCheckpoint(os.path.join(out_dir, "best_model.keras"), monitor="val_accuracy", mode="max", save_best_only=True),
		callbacks.ModelCheckpoint(os.path.join(out_dir, "last_model.keras"), monitor="val_accuracy", mode="max", save_best_only=False),
		callbacks.CSVLogger(os.path.join(logs_dir, "training_log.csv"), append=append_csv),
	]


def _save_training_curves(history: keras.callbacks.History, out_dir: str, title_suffix: str = "") -> None:
	logs_dir = os.path.join(out_dir, "logs")
	os.makedirs(logs_dir, exist_ok=True)
	h = history.history or {}
	acc = h.get("accuracy", [])
	val_acc = h.get("val_accuracy", [])
	loss = h.get("loss", [])
	val_loss = h.get("val_loss", [])
	ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	acc_path = os.path.join(logs_dir, f"accuracy_curve_{ts}.png")
	loss_path = os.path.join(logs_dir, f"loss_curve_{ts}.png")

	plt.figure(figsize=(8, 5))
	plt.plot(acc, label="train accuracy")
	if val_acc:
		plt.plot(val_acc, label="val accuracy")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	title = "Accuracy vs Epochs"
	if title_suffix:
		title += f" ({title_suffix})"
	plt.title(title)
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(acc_path, dpi=150)
	plt.close()

	plt.figure(figsize=(8, 5))
	plt.plot(loss, label="train loss")
	if val_loss:
		plt.plot(val_loss, label="val loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	title = "Loss vs Epochs"
	if title_suffix:
		title += f" ({title_suffix})"
	plt.title(title)
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(loss_path, dpi=150)
	plt.close()

	print(f"Saved training curves:\n  {acc_path}\n  {loss_path}")


class FSDDSequence(Sequence):
	def __init__(self, indices: np.ndarray, batch_size: int, max_len: int, n_fft: int, hop_length: int, add_deltas: bool, split: str, use_cache: bool, cache_dir: str):
		self.indices = np.asarray(indices)
		self.batch_size = batch_size
		self.max_len = max_len
		self.n_fft = n_fft
		self.hop_length = hop_length
		self.add_deltas = add_deltas
		self.split = split
		self.use_cache = use_cache
		self.cache_dir = cache_dir
		# Preload all data once using cache; subselection per batch keeps memory moderate
		self.X_all, self.y_all = load_fsdd_from_hf(split=split, max_len=max_len, n_fft=n_fft, hop_length=hop_length, add_deltas=add_deltas, apply_augmentation=False, use_cache=use_cache, cache_dir=cache_dir)

	def __len__(self) -> int:
		return int(np.ceil(len(self.indices) / self.batch_size))

	def __getitem__(self, idx: int):
		start = idx * self.batch_size
		end = min(start + self.batch_size, len(self.indices))
		sel = self.indices[start:end]
		X = self.X_all[sel][..., np.newaxis]
		y = self.y_all[sel]
		return X, y


def _build_model_for_arch(arch: str, input_shape: Tuple[int, int, int], args) -> keras.Model:
	if arch == "cnn":
		return build_cnn_model(
			input_shape=input_shape,
			num_classes=10,
			use_batchnorm=args.batchnorm,
			dropout_rate=args.dropout,
			l2_weight=args.l2,
			add_third_block=args.third_block,
			use_gap=args.gap,
		)
	if arch == "cs230_cnn":
		return build_cs230_cnn(
			input_shape=input_shape,
			num_classes=10,
			use_pool=True,
			learning_rate=1e-6,
		)
	else:
		return build_small_cnn(
			input_shape=input_shape,
			num_classes=10,
			depth_multiplier=1.0,
			use_batchnorm=True,
			use_se=True,
			se_ratio=0.25,
			dropout_rate=args.dropout,
			l2_weight=args.l2,
			activation_name="relu",
			learning_rate=1e-3,
		)


def run_cv_small_cnn(args) -> None:
	# Load entire dataset once (can use cache)
	X, y = load_fsdd_from_hf(split="train", max_len=args.max_len, n_fft=args.n_fft, hop_length=args.hop_length, add_deltas=args.add_deltas, apply_augmentation=False, use_cache=args.cache, cache_dir=args.cache_dir)
	X = X[..., np.newaxis]
	input_shape = (X.shape[1], X.shape[2], 1)
	kf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=args.seed)
	fold_accs = []
	logs_root = os.path.join(args.out, "logs")
	os.makedirs(logs_root, exist_ok=True)
	best_acc = -1.0
	best_model_path = os.path.join(args.out, "small_cnn_cv.keras")

	for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
		print(f"[CV] Fold {fold_idx}/{args.kfolds}: train={len(train_idx)}, val={len(val_idx)}")
		X_tr, y_tr = X[train_idx], y[train_idx]
		X_val, y_val = X[val_idx], y[val_idx]
		model = _build_model_for_arch("small_cnn", input_shape, args)
		fold_out = os.path.join(args.out, f"cv_fold_{fold_idx}")
		os.makedirs(fold_out, exist_ok=True)
		cbs = [
			callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True),
			callbacks.ReduceLROnPlateau(monitor="val_accuracy", mode="max", factor=0.5, patience=3, min_lr=1e-5),
			callbacks.ModelCheckpoint(os.path.join(fold_out, "best_model.keras"), monitor="val_accuracy", mode="max", save_best_only=True),
			callbacks.ModelCheckpoint(os.path.join(fold_out, "last_model.keras"), monitor="val_accuracy", mode="max", save_best_only=False),
			callbacks.CSVLogger(os.path.join(fold_out, "fold_log.csv"), append=False),
		]
		h = model.fit(
			X_tr, y_tr,
			validation_data=(X_val, y_val),
			epochs=args.cv_epochs,
			batch_size=args.cv_batch_size,
			callbacks=cbs,
			verbose=1,
		)
		loss, acc = model.evaluate(X_val, y_val, verbose=0)
		fold_accs.append(float(acc))
		if acc > best_acc:
			best_acc = float(acc)
			try:
				model.save(best_model_path)
			except Exception:
				pass

	# Save summary CSV
	summary_path = os.path.join(logs_root, "cv_summary.csv")
	with open(summary_path, "w", newline="") as f:
		w = csv.writer(f)
		w.writerow(["fold", "val_accuracy"])
		for i, a in enumerate(fold_accs, start=1):
			w.writerow([i, a])
		w.writerow([])
		w.writerow(["mean", float(np.mean(fold_accs))])
		w.writerow(["std", float(np.std(fold_accs))])
	print(f"[CV] Accuracies: {fold_accs}; mean={np.mean(fold_accs):.4f} std={np.std(fold_accs):.4f}")
	print(f"Saved best CV model to: {best_model_path}")
	print(f"Saved CV summary to: {summary_path}")


def run_grid_search_small_cnn(args) -> None:
	# Prepare data once
	X, y = load_fsdd_from_hf(split="train", max_len=args.max_len, n_fft=args.n_fft, hop_length=args.hop_length, add_deltas=args.add_deltas, apply_augmentation=False, use_cache=args.cache, cache_dir=args.cache_dir)
	X = X[..., np.newaxis]
	input_shape = (X.shape[1], X.shape[2], 1)
	kf = StratifiedKFold(n_splits=args.gs_folds, shuffle=True, random_state=args.seed)
	logs_root = os.path.join(args.out, "logs")
	os.makedirs(logs_root, exist_ok=True)
	csv_path = os.path.join(logs_root, "grid_search.csv")

	# Focused grid (≤20 combos)
	grid = [
		{"dropout_rate": 0.3, "l2_weight": 0.0, "activation_name": "relu", "use_se": True, "learning_rate": 1e-3},
		{"dropout_rate": 0.5, "l2_weight": 0.0, "activation_name": "relu", "use_se": True, "learning_rate": 1e-3},
		{"dropout_rate": 0.3, "l2_weight": 1e-4, "activation_name": "relu", "use_se": True, "learning_rate": 1e-3},
		{"dropout_rate": 0.5, "l2_weight": 1e-4, "activation_name": "relu", "use_se": True, "learning_rate": 1e-3},
		{"dropout_rate": 0.3, "l2_weight": 0.0, "activation_name": "elu",  "use_se": True, "learning_rate": 1e-3},
		{"dropout_rate": 0.5, "l2_weight": 0.0, "activation_name": "elu",  "use_se": True, "learning_rate": 1e-3},
		{"dropout_rate": 0.3, "l2_weight": 1e-4, "activation_name": "elu",  "use_se": True, "learning_rate": 1e-3},
		{"dropout_rate": 0.5, "l2_weight": 1e-4, "activation_name": "elu",  "use_se": True, "learning_rate": 1e-3},
		{"dropout_rate": 0.3, "l2_weight": 0.0, "activation_name": "relu", "use_se": True, "learning_rate": 5e-4},
		{"dropout_rate": 0.5, "l2_weight": 0.0, "activation_name": "relu", "use_se": True, "learning_rate": 5e-4},
		{"dropout_rate": 0.3, "l2_weight": 1e-4, "activation_name": "relu", "use_se": True, "learning_rate": 5e-4},
		{"dropout_rate": 0.5, "l2_weight": 1e-4, "activation_name": "relu", "use_se": True, "learning_rate": 5e-4},
		{"dropout_rate": 0.3, "l2_weight": 0.0, "activation_name": "elu",  "use_se": True, "learning_rate": 5e-4},
		{"dropout_rate": 0.5, "l2_weight": 0.0, "activation_name": "elu",  "use_se": True, "learning_rate": 5e-4},
		{"dropout_rate": 0.3, "l2_weight": 1e-4, "activation_name": "elu",  "use_se": True, "learning_rate": 5e-4},
		{"dropout_rate": 0.5, "l2_weight": 1e-4, "activation_name": "elu",  "use_se": True, "learning_rate": 5e-4},
	]
	if args.gs_max_combos and args.gs_max_combos > 0:
		grid = grid[: int(args.gs_max_combos)]

	# Resume: skip combos whose idx already logged
	completed = set()
	if args.gs_resume and os.path.exists(csv_path):
		try:
			with open(csv_path, "r") as f:
				r = csv.reader(f)
				next(r, None)
				for row in r:
					if row and row[0].isdigit():
						completed.add(int(row[0]))
		except Exception:
			pass

	with open(csv_path, "a", newline="") as f:
		if os.path.getsize(csv_path) == 0:
			w = csv.writer(f)
			w.writerow(["idx", "dropout_rate", "l2_weight", "activation", "use_se", "learning_rate", "mean_acc", "std_acc"])

	best_mean = -1.0
	best_cfg = None
	best_model_path = os.path.join(args.out, "small_cnn_gs.keras")

	for idx, cfg in enumerate(grid, start=1):
		if idx in completed:
			print(f"[GS] Skipping combo {idx} (already completed)")
			continue
		print(f"[GS] Combo {idx}/{len(grid)}: {cfg}")
		scores: list[float] = []
		for fold_i, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
			print(f"  - Fold {fold_i}/{args.gs_folds}: training...")
			X_tr, y_tr = X[train_idx], y[train_idx]
			X_val, y_val = X[val_idx], y[val_idx]
			model = build_small_cnn(
				input_shape=input_shape,
				num_classes=10,
				depth_multiplier=1.0,
				use_batchnorm=True,
				use_se=bool(cfg["use_se"]),
				se_ratio=0.25,
				dropout_rate=float(cfg["dropout_rate"]),
				l2_weight=float(cfg["l2_weight"]),
				activation_name=str(cfg["activation_name"]),
				learning_rate=float(cfg["learning_rate"]),
			)
			h = model.fit(
				X_tr, y_tr,
				validation_data=(X_val, y_val),
				epochs=args.gs_epochs,
				batch_size=args.gs_batch_size,
				callbacks=[
					callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=3, restore_best_weights=True),
					callbacks.ReduceLROnPlateau(monitor="val_accuracy", mode="max", factor=0.5, patience=2, min_lr=1e-5),
				],
				verbose=1,
			)
			val_acc = float(max(h.history.get("val_accuracy", [0.0])))
			scores.append(val_acc)
		mean_acc = float(np.mean(scores))
		std_acc = float(np.std(scores))
		print(f"[GS] -> mean_acc={mean_acc:.4f} std={std_acc:.4f}")

		with open(csv_path, "a", newline="") as f:
			w = csv.writer(f)
			w.writerow([idx, cfg["dropout_rate"], cfg["l2_weight"], cfg["activation_name"], cfg["use_se"], cfg["learning_rate"], mean_acc, std_acc])

		if mean_acc > best_mean:
			best_mean = mean_acc
			best_cfg = cfg
			# Refit best config on full data (quick epochs)
			best_model = build_small_cnn(
				input_shape=input_shape,
				num_classes=10,
				depth_multiplier=1.0,
				use_batchnorm=True,
				use_se=bool(cfg["use_se"]),
				se_ratio=0.25,
				dropout_rate=float(cfg["dropout_rate"]),
				l2_weight=float(cfg["l2_weight"]),
				activation_name=str(cfg["activation_name"]),
				learning_rate=float(cfg["learning_rate"]),
			)
			best_model.fit(X, y, epochs=max(1, min(5, args.gs_epochs)), batch_size=args.gs_batch_size, verbose=0)
			try:
				best_model.save(best_model_path)
			except Exception:
				pass

	print(f"[GS] Best mean accuracy: {best_mean:.4f}")
	print(f"[GS] Best config: {best_cfg}")
	print(f"Saved grid search CSV to: {csv_path}")
	print(f"Saved best model to: {best_model_path}")


def run_cv_cs230_cnn(args) -> None:
	# Load dataset once
	X, y = load_fsdd_from_hf(split="train", max_len=args.max_len, n_fft=args.n_fft, hop_length=args.hop_length, add_deltas=args.add_deltas, apply_augmentation=False, use_cache=args.cache, cache_dir=args.cache_dir)
	X = X[..., np.newaxis]
	input_shape = (X.shape[1], X.shape[2], 1)
	kf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=args.seed)
	logs_root = os.path.join(args.out, "logs")
	os.makedirs(logs_root, exist_ok=True)
	best_acc = -1.0
	best_model_path = os.path.join(args.out, "cs230_cnn_cv.keras")
	fold_accs: list[float] = []

	for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
		print(f"[CV][cs230] Fold {fold_idx}/{args.kfolds}: train={len(train_idx)}, val={len(val_idx)}")
		X_tr, y_tr = X[train_idx], y[train_idx]
		X_val, y_val = X[val_idx], y[val_idx]
		model = build_cs230_cnn(input_shape=input_shape, num_classes=10, use_pool=True, learning_rate=1e-6)
		fold_out = os.path.join(args.out, f"cs230_cv_fold_{fold_idx}")
		os.makedirs(fold_out, exist_ok=True)
		cbs = [
			callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True),
			callbacks.ReduceLROnPlateau(monitor="val_accuracy", mode="max", factor=0.5, patience=3, min_lr=1e-6),
			callbacks.ModelCheckpoint(os.path.join(fold_out, "best_model.keras"), monitor="val_accuracy", mode="max", save_best_only=True),
			callbacks.ModelCheckpoint(os.path.join(fold_out, "last_model.keras"), monitor="val_accuracy", mode="max", save_best_only=False),
			callbacks.CSVLogger(os.path.join(fold_out, "fold_log.csv"), append=False),
		]
		_ = model.fit(
			X_tr, y_tr,
			validation_data=(X_val, y_val),
			epochs=args.cv_epochs,
			batch_size=args.cv_batch_size,
			callbacks=cbs,
			verbose=1,
		)
		loss, acc = model.evaluate(X_val, y_val, verbose=0)
		fold_accs.append(float(acc))
		if acc > best_acc:
			best_acc = float(acc)
			try:
				model.save(best_model_path)
			except Exception:
				pass

	summary_path = os.path.join(logs_root, "cs230_cv_summary.csv")
	with open(summary_path, "w", newline="") as f:
		w = csv.writer(f)
		w.writerow(["fold", "val_accuracy"])
		for i, a in enumerate(fold_accs, start=1):
			w.writerow([i, a])
		w.writerow([])
		w.writerow(["mean", float(np.mean(fold_accs))])
		w.writerow(["std", float(np.std(fold_accs))])
	print(f"[CV][cs230] Accuracies: {fold_accs}; mean={np.mean(fold_accs):.4f} std={np.std(fold_accs):.4f}")
	print(f"Saved best CV model to: {best_model_path}")
	print(f"Saved CV summary to: {summary_path}")


def run_grid_search_cs230_cnn(args) -> None:
	# Data
	X, y = load_fsdd_from_hf(split="train", max_len=args.max_len, n_fft=args.n_fft, hop_length=args.hop_length, add_deltas=args.add_deltas, apply_augmentation=False, use_cache=args.cache, cache_dir=args.cache_dir)
	X = X[..., np.newaxis]
	input_shape = (X.shape[1], X.shape[2], 1)
	kf = StratifiedKFold(n_splits=args.gs_folds, shuffle=True, random_state=args.seed)
	logs_root = os.path.join(args.out, "logs")
	os.makedirs(logs_root, exist_ok=True)
	csv_path = os.path.join(logs_root, "cs230_grid_search.csv")

	# Small grid: learning rate × pooling
	lr_list = [1e-6, 5e-6, 1e-5]
	pool_list = [True, False]
	grid = [{"learning_rate": lr, "use_pool": up} for lr in lr_list for up in pool_list]
	if args.gs_max_combos and args.gs_max_combos > 0:
		grid = grid[: int(args.gs_max_combos)]

	# Resume
	completed = set()
	if args.gs_resume and os.path.exists(csv_path):
		try:
			with open(csv_path, "r") as f:
				r = csv.reader(f)
				next(r, None)
				for row in r:
					if row and row[0].isdigit():
						completed.add(int(row[0]))
		except Exception:
			pass

	with open(csv_path, "a", newline="") as f:
		if os.path.getsize(csv_path) == 0:
			w = csv.writer(f)
			w.writerow(["idx", "learning_rate", "use_pool", "mean_acc", "std_acc"])

	best_mean = -1.0
	best_cfg = None
	best_model_path = os.path.join(args.out, "cs230_cnn_gs.keras")

	for idx, cfg in enumerate(grid, start=1):
		if idx in completed:
			print(f"[GS][cs230] Skipping combo {idx} (already completed)")
			continue
		print(f"[GS][cs230] Combo {idx}/{len(grid)}: {cfg}")
		scores: list[float] = []
		for fold_i, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
			print(f"  - Fold {fold_i}/{args.gs_folds}: training...")
			X_tr, y_tr = X[train_idx], y[train_idx]
			X_val, y_val = X[val_idx], y[val_idx]
			model = build_cs230_cnn(input_shape=input_shape, num_classes=10, use_pool=bool(cfg["use_pool"]), learning_rate=float(cfg["learning_rate"]))
			_ = model.fit(
				X_tr, y_tr,
				validation_data=(X_val, y_val),
				epochs=args.gs_epochs,
				batch_size=args.gs_batch_size,
				callbacks=[
					callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=3, restore_best_weights=True),
					callbacks.ReduceLROnPlateau(monitor="val_accuracy", mode="max", factor=0.5, patience=2, min_lr=1e-6),
				],
				verbose=1,
			)
			val_acc = float(model.evaluate(X_val, y_val, verbose=0)[1])
			scores.append(val_acc)
		mean_acc = float(np.mean(scores))
		std_acc = float(np.std(scores))
		print(f"[GS][cs230] -> mean_acc={mean_acc:.4f} std={std_acc:.4f}")
		with open(csv_path, "a", newline="") as f:
			w = csv.writer(f)
			w.writerow([idx, cfg["learning_rate"], cfg["use_pool"], mean_acc, std_acc])
		if mean_acc > best_mean:
			best_mean = mean_acc
			best_cfg = cfg
			best_model = build_cs230_cnn(input_shape=input_shape, num_classes=10, use_pool=bool(cfg["use_pool"]), learning_rate=float(cfg["learning_rate"]))
			best_model.fit(X, y, epochs=max(1, min(5, args.gs_epochs)), batch_size=args.gs_batch_size, verbose=0)
			try:
				best_model.save(best_model_path)
			except Exception:
				pass

	print(f"[GS][cs230] Best mean accuracy: {best_mean:.4f}")
	print(f"[GS][cs230] Best config: {best_cfg}")
	print(f"Saved grid search CSV to: {csv_path}")
	print(f"Saved best model to: {best_model_path}")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--arch", choices=["cnn", "small_cnn", "cs230_cnn"], default="small_cnn")
	parser.add_argument("--max_len", type=int, default=200)
	parser.add_argument("--epochs", type=int, default=30)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--dropout", type=float, default=0.3)
	parser.add_argument("--l2", type=float, default=1e-4)
	parser.add_argument("--batchnorm", action="store_true")
	parser.add_argument("--third_block", action="store_true")
	parser.add_argument("--gap", action="store_true")
	parser.add_argument("--n_fft", type=int, default=512)
	parser.add_argument("--hop_length", type=int, default=128)
	parser.add_argument("--add_deltas", action="store_true")
	parser.add_argument("--out", type=str, default="models")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--cache", action="store_true")
	parser.add_argument("--cache_dir", type=str, default="data/cache")
	parser.add_argument("--generator", action="store_true")
	parser.add_argument("--resume_model", type=str, default="", help="Path to a saved .keras model to resume from")
	# CV flags
	parser.add_argument("--cv", action="store_true")
	parser.add_argument("--kfolds", type=int, default=5)
	parser.add_argument("--cv_epochs", type=int, default=10)
	parser.add_argument("--cv_batch_size", type=int, default=32)
	# Grid search flags
	parser.add_argument("--grid_search", action="store_true")
	parser.add_argument("--gs_folds", type=int, default=3)
	parser.add_argument("--gs_epochs", type=int, default=5)
	parser.add_argument("--gs_batch_size", type=int, default=32)
	parser.add_argument("--gs_max_combos", type=int, default=0)
	parser.add_argument("--gs_resume", action="store_true")
	args = parser.parse_args()

	_set_seeds(args.seed)

	if args.grid_search and args.arch == "small_cnn":
		run_grid_search_small_cnn(args)
		return

	if args.grid_search and args.arch == "cs230_cnn":
		run_grid_search_cs230_cnn(args)
		return

	if args.cv and args.arch == "small_cnn":
		run_cv_small_cnn(args)
		return

	if args.cv and args.arch == "cs230_cnn":
		run_cv_cs230_cnn(args)
		return

	if not args.generator:
		X_all, y_all = load_fsdd_from_hf(split="train", max_len=args.max_len, n_fft=args.n_fft, hop_length=args.hop_length, add_deltas=args.add_deltas, apply_augmentation=False, use_cache=args.cache, cache_dir=args.cache_dir)
		X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_all, test_size=0.15, random_state=args.seed, stratify=y_all)
		X_tr = X_tr[..., np.newaxis]
		X_val = X_val[..., np.newaxis]
	else:
		X_all, y_all = load_fsdd_from_hf(split="train", max_len=args.max_len, n_fft=args.n_fft, hop_length=args.hop_length, add_deltas=args.add_deltas, apply_augmentation=False, use_cache=args.cache, cache_dir=args.cache_dir)
		idx = np.arange(len(y_all))
		X_idx_tr, X_idx_val, y_tr, y_val = train_test_split(idx, y_all, test_size=0.15, random_state=args.seed, stratify=y_all)
		train_gen = FSDDSequence(X_idx_tr, args.batch_size, args.max_len, args.n_fft, args.hop_length, args.add_deltas, split="train", use_cache=args.cache, cache_dir=args.cache_dir)
		val_gen = FSDDSequence(X_idx_val, args.batch_size, args.max_len, args.n_fft, args.hop_length, args.add_deltas, split="train", use_cache=args.cache, cache_dir=args.cache_dir)

	X_test, y_test = load_fsdd_from_hf(split="test", max_len=args.max_len, n_fft=args.n_fft, hop_length=args.hop_length, add_deltas=args.add_deltas, apply_augmentation=False, use_cache=args.cache, cache_dir=args.cache_dir)
	X_test_exp = X_test[..., np.newaxis]

	input_shape = (X_test_exp.shape[1], X_test_exp.shape[2], X_test_exp.shape[3])
	if args.resume_model and os.path.isfile(args.resume_model):
		print(f"[Resume] Loading model from {args.resume_model}")
		model = keras.saving.load_model(args.resume_model)
		append_csv = True
	else:
		model = _build_model_for_arch(args.arch, input_shape, args)
		append_csv = False

	cbs = build_callbacks(args.out, append_csv=append_csv)

	# Determine initial_epoch if resuming
	initial_epoch = 0
	if append_csv:
		log_path = os.path.join(args.out, "logs", "training_log.csv")
		if os.path.isfile(log_path):
			try:
				with open(log_path, "r") as f:
					r = csv.reader(f)
					next(r, None)
					initial_epoch = sum(1 for _ in r)
				print(f"[Resume] initial_epoch={initial_epoch}")
			except Exception:
				pass

	if not args.generator:
		history = model.fit(
			X_tr,
			y_tr,
			validation_data=(X_val, y_val),
			epochs=args.epochs,
			initial_epoch=initial_epoch,
			batch_size=args.batch_size,
			callbacks=cbs,
			verbose=1,
		)
	else:
		history = model.fit(
			train_gen,
			validation_data=val_gen,
			epochs=args.epochs,
			initial_epoch=initial_epoch,
			callbacks=cbs,
			verbose=1,
		)

	title_suffix = f"arch={args.arch}, do={args.dropout}, l2={args.l2}, nfft={args.n_fft}, hop={args.hop_length}, deltas={args.add_deltas}, gen={args.generator}, cache={args.cache}"
	_save_training_curves(history, args.out, title_suffix=title_suffix)

	test_loss, test_acc = model.evaluate(X_test_exp, y_test, verbose=0)
	print(f"Final test accuracy: {test_acc:.4f}")

	os.makedirs(args.out, exist_ok=True)
	if args.arch == "cnn":
		final_name = "final_model.keras"
	elif args.arch == "cs230_cnn":
		final_name = "cs230_cnn.keras"
	else:
		final_name = "small_cnn.keras"
	final_path = os.path.join(args.out, final_name)
	model.save(final_path)
	print(f"Saved final model to: {final_path}")


if __name__ == "__main__":
	main() 