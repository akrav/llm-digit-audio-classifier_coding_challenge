import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import argparse
import numpy as np
from typing import List
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_processing import load_fsdd_from_hf


def _plot_confusion_matrix(cm: np.ndarray, classes: List[str], out_path: str) -> None:
	plt.figure(figsize=(6, 5))
	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title('Confusion matrix')
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes)
	plt.yticks(tick_marks, classes)
	thresh = cm.max() / 2.0 if cm.size else 0.0
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			plt.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.tight_layout()
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	plt.savefig(out_path, dpi=150)
	plt.close()


def _eval_baseline(model_path: str, split: str, max_len: int, n_fft: int, hop: int, add_deltas: bool, out_dir: str, save_cm: bool):
	import joblib
	from src.baseline import pool_features
	X, y = load_fsdd_from_hf(split=split, max_len=max_len, n_fft=n_fft, hop_length=hop, add_deltas=add_deltas, apply_augmentation=False, use_cache=True, cache_dir='data/cache')
	X_in = X.reshape(X.shape[0], -1)
	pipe = joblib.load(model_path)
	y_pred = pipe.predict(X_in)
	acc = accuracy_score(y, y_pred)
	print(f"Overall accuracy: {acc:.4f}")
	labels = list(range(10))
	print(classification_report(y, y_pred, labels=labels, digits=4))
	cm = confusion_matrix(y, y_pred, labels=labels)
	print(cm)
	if save_cm:
		_plot_confusion_matrix(cm, [str(i) for i in labels], os.path.join(out_dir, 'confusion_matrix_baseline.png'))


def _eval_keras(model_path: str, split: str, max_len: int, n_fft: int, hop: int, add_deltas: bool, out_dir: str, save_cm: bool):
	import keras
	X, y = load_fsdd_from_hf(split=split, max_len=max_len, n_fft=n_fft, hop_length=hop, add_deltas=add_deltas, apply_augmentation=False, use_cache=True, cache_dir='data/cache')
	X_in = X[..., np.newaxis]
	model = keras.saving.load_model(model_path)
	probs = model.predict(X_in, verbose=0)
	y_pred = np.argmax(probs, axis=1)
	acc = accuracy_score(y, y_pred)
	print(f"Overall accuracy: {acc:.4f}")
	labels = list(range(10))
	print(classification_report(y, y_pred, labels=labels, digits=4))
	cm = confusion_matrix(y, y_pred, labels=labels)
	print(cm)
	if save_cm:
		_plot_confusion_matrix(cm, [str(i) for i in labels], os.path.join(out_dir, 'confusion_matrix_keras.png'))


def _get_label(ex: dict) -> int:
	for key in ("label", "digit", "class", "target"):
		if key in ex:
			return int(ex[key])
	name = ex.get('filename') or (ex.get('path') if isinstance(ex.get('audio'), dict) else None)
	if name:
		try:
			return int(os.path.basename(name)[0])
		except Exception:
			pass
	raise KeyError("Could not find label in example")


def _eval_w2v(dir_path: str, split: str, out_dir: str, save_cm: bool):
	import torch
	from datasets import load_dataset
	from datasets.features import Audio as HF_Audio
	from transformers import AutoProcessor, Wav2Vec2ForSequenceClassification
	processor = AutoProcessor.from_pretrained(dir_path)
	model = Wav2Vec2ForSequenceClassification.from_pretrained(dir_path)
	model.eval()
	sr = int(getattr(getattr(processor, 'feature_extractor', {}), 'sampling_rate', 16000))
	ds = load_dataset('mteb/free-spoken-digit-dataset', split=split)
	ds = ds.cast_column('audio', HF_Audio(decode=True, sampling_rate=sr))
	y_true, y_pred = [], []
	for ex in ds:
		arr = np.asarray(ex['audio']['array'], dtype=np.float32)
		inputs = processor(arr, sampling_rate=sr, return_tensors='pt', padding=True)
		with torch.no_grad():
			out = model(**inputs)
			pred = int(out.logits.argmax(dim=-1).cpu().numpy()[0])
		y_true.append(_get_label(ex))
		y_pred.append(pred)
	acc = accuracy_score(y_true, y_pred)
	print(f"Overall accuracy: {acc:.4f}")
	labels = list(range(10))
	print(classification_report(y_true, y_pred, labels=labels, digits=4))
	cm = confusion_matrix(y_true, y_pred, labels=labels)
	print(cm)
	if save_cm:
		_plot_confusion_matrix(cm, [str(i) for i in labels], os.path.join(out_dir, 'confusion_matrix_wav2vec2.png'))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--arch', choices=['baseline', 'keras', 'wav2vec2'], required=True)
	parser.add_argument('--model', type=str, help='Path to model or directory for wav2vec2')
	parser.add_argument('--split', type=str, default='test')
	parser.add_argument('--max_len', type=int, default=200)
	parser.add_argument('--n_fft', type=int, default=512)
	parser.add_argument('--hop_length', type=int, default=128)
	parser.add_argument('--add_deltas', action='store_true')
	parser.add_argument('--out', type=str, default='models/logs')
	parser.add_argument('--save_cm', action='store_true')
	args = parser.parse_args()

	if args.arch == 'baseline':
		if not args.model: raise ValueError('--model required for baseline')
		_eval_baseline(args.model, args.split, args.max_len, args.n_fft, args.hop_length, args.add_deltas, args.out, args.save_cm)
	elif args.arch == 'keras':
		if not args.model: raise ValueError('--model required for keras')
		_eval_keras(args.model, args.split, args.max_len, args.n_fft, args.hop_length, args.add_deltas, args.out, args.save_cm)
	else:
		if not args.model: raise ValueError('--model (dir) required for wav2vec2')
		_eval_w2v(args.model, args.split, args.out, args.save_cm)


if __name__ == '__main__':
	main() 