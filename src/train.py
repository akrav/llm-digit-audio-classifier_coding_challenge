import os
import math
import argparse
import numpy as np
from typing import Tuple

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from keras import callbacks
from sklearn.model_selection import train_test_split

from src.model import build_cnn_model
from src.data_processing import load_fsdd_from_hf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime


def build_callbacks(out_dir: str) -> list:
	os.makedirs(out_dir, exist_ok=True)
	return [
		callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=8, restore_best_weights=True),
		callbacks.ReduceLROnPlateau(monitor="val_accuracy", mode="max", factor=0.5, patience=4, min_lr=1e-5),
		callbacks.ModelCheckpoint(os.path.join(out_dir, "best_model.keras"), monitor="val_accuracy", mode="max", save_best_only=True),
		callbacks.CSVLogger(os.path.join(out_dir, "training_log.csv")),
	]


def _save_training_curves(history: keras.callbacks.History, out_dir: str, title_suffix: str = "") -> None:
	os.makedirs(out_dir, exist_ok=True)
	h = history.history or {}
	acc = h.get("accuracy", [])
	val_acc = h.get("val_accuracy", [])
	loss = h.get("loss", [])
	val_loss = h.get("val_loss", [])
	ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	acc_path = os.path.join(out_dir, f"accuracy_curve_{ts}.png")
	loss_path = os.path.join(out_dir, f"loss_curve_{ts}.png")

	# Accuracy plot
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

	# Loss plot
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


def main():
	parser = argparse.ArgumentParser()
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
	args = parser.parse_args()

	# Load data with MFCC params
	X_all, y_all = load_fsdd_from_hf(split="train", max_len=args.max_len, n_fft=args.n_fft, hop_length=args.hop_length, add_deltas=args.add_deltas, apply_augmentation=False)
	# Stratified split train/val
	X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_all, test_size=0.15, random_state=42, stratify=y_all)
	# Add channel dim
	X_tr = X_tr[..., np.newaxis]
	X_val = X_val[..., np.newaxis]
	X_test, y_test = load_fsdd_from_hf(split="test", max_len=args.max_len, n_fft=args.n_fft, hop_length=args.hop_length, add_deltas=args.add_deltas, apply_augmentation=False)
	X_test = X_test[..., np.newaxis]

	# Build model with regularization options
	input_shape = (X_tr.shape[1], X_tr.shape[2], X_tr.shape[3])
	model = build_cnn_model(
		input_shape=input_shape,
		num_classes=10,
		use_batchnorm=args.batchnorm,
		dropout_rate=args.dropout,
		l2_weight=args.l2,
		add_third_block=args.third_block,
		use_gap=args.gap,
	)

	cbs = build_callbacks(args.out)
	history = model.fit(
		X_tr,
		y_tr,
		validation_data=(X_val, y_val),
		epochs=args.epochs,
		batch_size=args.batch_size,
		callbacks=cbs,
		verbose=1,
	)

	# Save training curves
	title_suffix = f"bn={args.batchnorm}, gap={args.gap}, do={args.dropout}, l2={args.l2}, nfft={args.n_fft}, hop={args.hop_length}, deltas={args.add_deltas}"
	_save_training_curves(history, args.out, title_suffix=title_suffix)

	# Evaluate
	test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
	print(f"Final test accuracy: {test_acc:.4f}")

	# Save final
	os.makedirs(args.out, exist_ok=True)
	final_path = os.path.join(args.out, "final_model.keras")
	model.save(final_path)
	print(f"Saved final model to: {final_path}")


if __name__ == "__main__":
	main() 