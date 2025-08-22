import os
import csv
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime


def _read_csv_metrics(csv_path: str) -> Tuple[List[float], List[float]]:
	acc, val_acc = [], []
	loss, val_loss = [], []
	if not os.path.isfile(csv_path):
		raise FileNotFoundError(csv_path)
	with open(csv_path, "r") as f:
		reader = csv.DictReader(f)
		for row in reader:
			if "accuracy" in row:
				try:
					acc.append(float(row["accuracy"]))
					val_acc.append(float(row.get("val_accuracy", "nan")))
					loss.append(float(row.get("loss", "nan")))
					val_loss.append(float(row.get("val_loss", "nan")))
				except Exception:
					continue
	return acc, val_acc, loss, val_loss


def plot_run(csv_path: str, out_dir: str, title: str) -> Tuple[str, str]:
	os.makedirs(out_dir, exist_ok=True)
	acc, val_acc, loss, val_loss = _read_csv_metrics(csv_path)
	ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	acc_png = os.path.join(out_dir, f"{title}_acc_{ts}.png")
	loss_png = os.path.join(out_dir, f"{title}_loss_{ts}.png")

	plt.figure(figsize=(8, 5))
	plt.plot(acc, label="train acc")
	if any(v == v for v in val_acc):
		plt.plot(val_acc, label="val acc")
	plt.title(f"Accuracy - {title}")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(acc_png, dpi=150)
	plt.close()

	plt.figure(figsize=(8, 5))
	plt.plot(loss, label="train loss")
	if any(v == v for v in val_loss):
		plt.plot(val_loss, label="val loss")
	plt.title(f"Loss - {title}")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(loss_png, dpi=150)
	plt.close()

	return acc_png, loss_png


def plot_cv_folds(fold_dirs: List[str], out_dir: str, title: str) -> Tuple[str, str]:
	os.makedirs(out_dir, exist_ok=True)
	fold_accs = []
	fold_val_accs = []
	fold_losses = []
	fold_val_losses = []
	for d in fold_dirs:
		csv_path = os.path.join(d, "fold_log.csv")
		acc, val_acc, loss, val_loss = _read_csv_metrics(csv_path)
		if acc:
			fold_accs.append(acc)
			fold_val_accs.append(val_acc)
			fold_losses.append(loss)
			fold_val_losses.append(val_loss)

	def _mean_curve(curves: List[List[float]]) -> List[float]:
		if not curves:
			return []
		m = max(len(c) for c in curves)
		means = []
		for i in range(m):
			vals = [c[i] for c in curves if i < len(c)]
			means.append(sum(vals) / max(1, len(vals)))
		return means

	acc_m = _mean_curve(fold_accs)
	val_acc_m = _mean_curve(fold_val_accs)
	loss_m = _mean_curve(fold_losses)
	val_loss_m = _mean_curve(fold_val_losses)

	ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	acc_png = os.path.join(out_dir, f"{title}_cv_acc_{ts}.png")
	loss_png = os.path.join(out_dir, f"{title}_cv_loss_{ts}.png")

	plt.figure(figsize=(8, 5))
	for i, acc in enumerate(fold_accs):
		plt.plot(acc, alpha=0.3, label=f"fold{i+1}")
	plt.plot(acc_m, color="k", linewidth=2, label="mean")
	plt.title(f"CV Accuracy - {title}")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(acc_png, dpi=150)
	plt.close()

	plt.figure(figsize=(8, 5))
	for i, ls in enumerate(fold_losses):
		plt.plot(ls, alpha=0.3, label=f"fold{i+1}")
	plt.plot(loss_m, color="k", linewidth=2, label="mean")
	plt.title(f"CV Loss - {title}")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(loss_png, dpi=150)
	plt.close()

	return acc_png, loss_png


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--single_csv", type=str, help="Path to a CSVLogger file (training_log.csv)")
	parser.add_argument("--cv_root", type=str, help="Directory containing cv_fold_*/fold_log.csv")
	parser.add_argument("--out", type=str, default="models/logs/plots")
	args = parser.parse_args()

	paths = []
	if args.single_csv and os.path.isfile(args.single_csv):
		paths.extend(list(plot_run(args.single_csv, args.out, title="single_run")))
	if args.cv_root and os.path.isdir(args.cv_root):
		fold_dirs = [os.path.join(args.cv_root, d) for d in os.listdir(args.cv_root) if d.startswith("cv_fold_")]
		paths.extend(list(plot_cv_folds(fold_dirs, args.out, title="cv")))
	print("Saved plots:")
	for p in paths:
		print(" ", p) 