import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import argparse
import numpy as np
import subprocess
import csv

from typing import List, Tuple

from src.data_processing import load_fsdd_from_hf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
import shutil


def _save_curves(history, out_dir: str, title: str) -> None:
	os.makedirs(out_dir, exist_ok=True)
	h = getattr(history, 'history', {}) or {}
	acc = h.get('accuracy', [])
	val_acc = h.get('val_accuracy', [])
	loss = h.get('loss', [])
	val_loss = h.get('val_loss', [])
	plt.figure(figsize=(6,4))
	plt.plot(acc, label='acc')
	if val_acc: plt.plot(val_acc, label='val')
	plt.title(f'Accuracy - {title}')
	plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
	plt.savefig(os.path.join(out_dir, f'{title}_acc.png'), dpi=120)
	plt.close()
	plt.figure(figsize=(6,4))
	plt.plot(loss, label='loss')
	if val_loss: plt.plot(val_loss, label='val')
	plt.title(f'Loss - {title}')
	plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
	plt.savefig(os.path.join(out_dir, f'{title}_loss.png'), dpi=120)
	plt.close()


def _parse_float_grid(s: str, default: List[float]) -> List[float]:
	if not s:
		return default
	parts = [p.strip() for p in s.split(',') if p.strip()]
	vals: List[float] = []
	for p in parts:
		try:
			vals.append(float(p))
		except Exception:
			pass
	return vals or default


def _parse_str_grid(s: str, default: List[str]) -> List[str]:
	if not s:
		return default
	parts = [p.strip() for p in s.split(',') if p.strip()]
	return parts or default


def train_baseline(save_path: str, max_len: int, n_fft: int, hop: int, add_deltas: bool, mode: str, gs_folds: int, C_grid: List[float], gamma_grid: List[str]) -> str:
	from src.baseline import make_baseline_model
	from sklearn.model_selection import StratifiedKFold, GridSearchCV
	X, y = load_fsdd_from_hf(split='train', max_len=max_len, n_fft=n_fft, hop_length=hop, add_deltas=add_deltas, apply_augmentation=False, use_cache=True, cache_dir='data/cache')
	X_flat = X.reshape(X.shape[0], -1)
	if mode in ('grid','cvgrid'):
		pipe = make_baseline_model(algo='svm', C=2.0, gamma='scale')
		cv = StratifiedKFold(n_splits=max(2, gs_folds), shuffle=True, random_state=42)
		params = {'clf__C': C_grid, 'clf__gamma': gamma_grid}
		gs = GridSearchCV(pipe, params, cv=cv, n_jobs=1, verbose=3)
		print(f"[baseline][grid] params={params}, folds={gs_folds}")
		gs.fit(X_flat, y)
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		import joblib
		joblib.dump(gs.best_estimator_, save_path)
	else:
		cv = StratifiedKFold(n_splits=max(2, gs_folds), shuffle=True, random_state=42)
		accs: List[float] = []
		from sklearn.metrics import accuracy_score
		for i, (tr, va) in enumerate(cv.split(X_flat, y), start=1):
			print(f"[baseline][cv] fold {i}/{gs_folds} training...")
			pipe = make_baseline_model(algo='svm', C=2.0, gamma='scale')
			pipe.fit(X_flat[tr], y[tr])
			pred = pipe.predict(X_flat[va])
			accs.append(float(accuracy_score(y[va], pred)))
			print(f"[baseline][cv] fold {i} acc={accs[-1]:.4f}")
		print(f'[baseline][cv] accs={accs}, mean={np.mean(accs):.4f}')
		pipe = make_baseline_model(algo='svm', C=2.0, gamma='scale')
		pipe.fit(X_flat, y)
		import joblib
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		joblib.dump(pipe, save_path)
	return save_path


def train_keras(save_path: str, max_len: int, n_fft: int, hop: int, add_deltas: bool, epochs: int, batch: int, mode: str, cv_folds: int, gs_folds: int, lr_grid: List[float]) -> str:
	import keras
	from keras import callbacks
	from sklearn.model_selection import StratifiedKFold
	from src.model import build_cs230_cnn
	X, y = load_fsdd_from_hf(split='train', max_len=max_len, n_fft=n_fft, hop_length=hop, add_deltas=add_deltas, apply_augmentation=False, use_cache=True, cache_dir='data/cache')
	X_in = X[..., np.newaxis]
	input_shape = (X_in.shape[1], X_in.shape[2], X_in.shape[3])
	logs = os.path.join(os.path.dirname(save_path), 'logs')
	os.makedirs(logs, exist_ok=True)
	if mode == 'cv':
		kf = StratifiedKFold(n_splits=max(2, cv_folds), shuffle=True, random_state=42)
		accs: List[float] = []
		for i, (tr, va) in enumerate(kf.split(X_in, y), start=1):
			model = build_cs230_cnn(input_shape=input_shape, num_classes=10, use_pool=True, learning_rate=1e-6)
			cbs = [
				callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=5, restore_best_weights=True),
				callbacks.ReduceLROnPlateau(monitor='val_accuracy', mode='max', factor=0.5, patience=2, min_lr=1e-6),
			]
			h = model.fit(X_in[tr], y[tr], validation_data=(X_in[va], y[va]), epochs=epochs, batch_size=batch, verbose=1, callbacks=cbs)
			_save_curves(h, logs, f'keras_cv_fold{i}')
			accs.append(float(model.evaluate(X_in[va], y[va], verbose=0)[1]))
		print(f'[keras][cv] accs={accs}, mean={np.mean(accs):.4f}')
		model = build_cs230_cnn(input_shape=input_shape, num_classes=10, use_pool=True, learning_rate=1e-6)
		h = model.fit(X_in, y, validation_split=0.15, epochs=epochs, batch_size=batch, verbose=1)
		_save_curves(h, logs, 'keras_cv_final')
		model.save(save_path)
	elif mode == 'grid':
		grid = [{'learning_rate': lr} for lr in lr_grid]
		best_acc = -1.0
		best_lr = None
		for j, cfg in enumerate(grid, start=1):
			print(f"[keras][grid] combo {j}/{len(grid)}: {cfg}")
			model = build_cs230_cnn(input_shape=input_shape, num_classes=10, use_pool=True, learning_rate=float(cfg['learning_rate']))
			h = model.fit(X_in, y, validation_split=0.15, epochs=epochs, batch_size=batch, verbose=1)
			acc = float(max(h.history.get('val_accuracy', [0.0])))
			_save_curves(h, logs, f'keras_grid_lr{cfg["learning_rate"]}')
			if acc > best_acc:
				best_acc = acc; best_lr = cfg['learning_rate']
		print(f"[keras][grid] best lr={best_lr} val_acc={best_acc:.4f}")
		model = build_cs230_cnn(input_shape=input_shape, num_classes=10, use_pool=True, learning_rate=float(best_lr or 1e-3))
		h = model.fit(X_in, y, validation_split=0.15, epochs=epochs, batch_size=batch, verbose=1)
		_save_curves(h, logs, 'keras_grid_final')
		model.save(save_path)
	elif mode == 'cvgrid':
		grid = [{'learning_rate': lr} for lr in lr_grid]
		kf = StratifiedKFold(n_splits=max(2, gs_folds), shuffle=True, random_state=42)
		best_mean = -1.0
		best_cfg = None
		for j, cfg in enumerate(grid, start=1):
			print(f"[keras][cvgrid] combo {j}/{len(grid)}: {cfg}")
			scores: List[float] = []
			for i, (tr, va) in enumerate(kf.split(X_in, y), start=1):
				model = build_cs230_cnn(input_shape=input_shape, num_classes=10, use_pool=True, learning_rate=float(cfg['learning_rate']))
				cbs = [
					callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=3, restore_best_weights=True),
					callbacks.ReduceLROnPlateau(monitor='val_accuracy', mode='max', factor=0.5, patience=2, min_lr=1e-6),
				]
				h = model.fit(X_in[tr], y[tr], validation_data=(X_in[va], y[va]), epochs=epochs, batch_size=batch, verbose=1, callbacks=cbs)
				acc = float(max(h.history.get('val_accuracy', [0.0])))
				_save_curves(h, logs, f'keras_cvgrid_lr{cfg["learning_rate"]}_fold{i}')
				scores.append(acc)
			m = float(np.mean(scores))
			print(f"  -> mean_val_acc={m:.4f}")
			if m > best_mean:
				best_mean = m; best_cfg = cfg
		print(f"[keras][cvgrid] best cfg={best_cfg} mean_val_acc={best_mean:.4f}")
		model = build_cs230_cnn(input_shape=input_shape, num_classes=10, use_pool=True, learning_rate=float((best_cfg or {'learning_rate':1e-3})['learning_rate']))
		h = model.fit(X_in, y, validation_split=0.15, epochs=epochs, batch_size=batch, verbose=1)
		_save_curves(h, logs, 'keras_cvgrid_final')
		model.save(save_path)
	else:
		model = build_cs230_cnn(input_shape=input_shape, num_classes=10, use_pool=True, learning_rate=1e-6)
		h = model.fit(X_in, y, validation_split=0.15, epochs=epochs, batch_size=batch, verbose=1)
		_save_curves(h, logs, 'keras_train')
		model.save(save_path)
	return save_path


def _evaluate_w2v_acc(dir_path: str) -> float:
	try:
		out = subprocess.check_output(f"python -m src.evaluate --arch wav2vec2 --model {dir_path}", shell=True, text=True)
		for line in out.splitlines():
			if line.strip().startswith("Overall accuracy:"):
				return float(line.strip().split()[-1])
	except Exception:
		return -1.0
	return -1.0


def train_wav2vec2(save_dir: str, epochs: int, batch: int, lr: float, mode: str, name: str, lr_grid: List[float]) -> str:
	# Base directory contains subfolders per LR
	os.makedirs(save_dir, exist_ok=True)
	best_dir = None
	best_acc = -1.0
	if mode in ('grid','cvgrid'):
		for l in lr_grid:
			subdir = os.path.join(save_dir, f"lr_{l}")
			cmd = f"python -m src.wav2vec2_train --output_dir {subdir} --epochs {epochs} --batch_size {batch} --lr {l} --freeze_feature_extractor"
			print('[w2v2][grid] running:', cmd)
			subprocess.run(cmd, shell=True, check=False)
			acc = _evaluate_w2v_acc(subdir)
			print(f"[w2v2][grid] lr={l} acc={acc:.4f}")
			if acc > best_acc:
				best_acc = acc; best_dir = subdir
		# Copy best model files into base dir
		if best_dir:
			for item in os.listdir(save_dir):
				# preserve lr_* subdirs; we'll copy best contents to root
				pass
			for fn in os.listdir(best_dir):
				src = os.path.join(best_dir, fn)
				dst = os.path.join(save_dir, fn)
				if os.path.isdir(src):
					if os.path.exists(dst):
						shutil.rmtree(dst)
					shutil.copytree(src, dst)
				else:
					shutil.copy2(src, dst)
			print(f"[w2v2][grid] best={best_dir} copied into {save_dir}")
	else:
		cmd = f"python -m src.wav2vec2_train --output_dir {save_dir} --epochs {epochs} --batch_size {batch} --lr {lr} --freeze_feature_extractor"
		print('[w2v2][train] running:', cmd)
		subprocess.run(cmd, shell=True, check=False)
	return save_dir


def _timestamped(path: str) -> str:
	base, ext = os.path.splitext(path)
	ts = datetime.now().strftime('%Y%m%d_%H%M%S')
	return f"{base}_{ts}{ext}"


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--arch', choices=['baseline', 'keras', 'wav2vec2'], required=True)
	parser.add_argument('--out', type=str, default='models')
	parser.add_argument('--name', type=str, default='')
	parser.add_argument('--avoid_overwrite', action='store_true')
	parser.add_argument('--cleanup', action='store_true')
	parser.add_argument('--mode', choices=['train', 'cv', 'grid', 'cvgrid'], default='train')
	parser.add_argument('--max_len', type=int, default=200)
	parser.add_argument('--n_fft', type=int, default=512)
	parser.add_argument('--hop_length', type=int, default=128)
	parser.add_argument('--add_deltas', action='store_true')
	parser.add_argument('--epochs', type=int, default=1)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--lr', type=float, default=5e-5)
	parser.add_argument('--cv_folds', type=int, default=3)
	parser.add_argument('--gs_folds', type=int, default=3)
	parser.add_argument('--svm_C_grid', type=str, default='1.0,2.0')
	parser.add_argument('--svm_gamma_grid', type=str, default='scale,auto')
	parser.add_argument('--keras_lr_grid', type=str, default='0.001,0.0005,0.0001')
	parser.add_argument('--w2v2_lr_grid', type=str, default='5e-5,1e-4')
	args = parser.parse_args()

	# Decide artifact name
	ext = '.joblib' if args.arch=='baseline' else ('' if args.arch=='wav2vec2' else '.keras')
	default_base = f"{('baseline_svm' if args.arch=='baseline' else ('cs230_cnn' if args.arch=='keras' else 'wav2vec2_digits'))}_{args.mode}_test"
	base_name = args.name if args.name else default_base
	artifact = os.path.join(args.out, base_name + ('' if args.arch=='wav2vec2' else ext))
	if args.avoid_overwrite and os.path.exists(artifact):
		artifact = _timestamped(artifact)

	if args.arch == 'baseline':
		C_grid = _parse_float_grid(args.svm_C_grid, [1.0, 2.0])
		gamma_grid = _parse_str_grid(args.svm_gamma_grid, ['scale','auto'])
		train_baseline(artifact, args.max_len, args.n_fft, args.hop_length, args.add_deltas, args.mode, args.gs_folds, C_grid, gamma_grid)
	elif args.arch == 'keras':
		lr_grid = _parse_float_grid(args.keras_lr_grid, [1e-3, 5e-4, 1e-4])
		train_keras(artifact, args.max_len, args.n_fft, args.hop_length, args.add_deltas, args.epochs, args.batch_size, args.mode, args.cv_folds, args.gs_folds, lr_grid)
	else:
		lr_grid = _parse_float_grid(args.w2v2_lr_grid, [5e-5, 1e-4])
		train_wav2vec2(artifact, args.epochs, args.batch_size, args.lr, args.mode, base_name, lr_grid)

	exists = os.path.exists(artifact) or args.arch=='wav2vec2'
	if exists:
		print(f"Created artifact: {artifact}")
		if args.cleanup:
			if os.path.isdir(artifact):
				shutil.rmtree(artifact)
			else:
				try:
					os.remove(artifact)
				except Exception:
					pass
			print("Cleanup: removed artifact.")
	else:
		print("No artifact created or path missing.")


if __name__ == '__main__':
	main() 