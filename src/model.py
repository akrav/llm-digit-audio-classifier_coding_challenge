from typing import Tuple, Optional, Dict, Any, Iterable

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from keras import layers, models, optimizers, regularizers, losses, callbacks
import numpy as np
import sys

from src.data_processing import load_fsdd_from_hf
from sklearn.model_selection import StratifiedKFold


def _get_activation_layer(name: str) -> layers.Layer:
	name = name.lower()
	if name == "relu":
		return layers.Activation("relu")
	if name == "elu":
		return layers.ELU(alpha=1.0)
	if name in ("leaky_relu", "lrelu"):
		return layers.LeakyReLU(alpha=0.1)
	raise ValueError(f"Unsupported activation: {name}")


def _apply_activation(x, name: str):
	name = name.lower()
	if name == "relu":
		return layers.Activation("relu")(x)
	if name == "elu":
		return layers.ELU(alpha=1.0)(x)
	if name in ("leaky_relu", "lrelu"):
		return layers.LeakyReLU(alpha=0.1)(x)
	raise ValueError(f"Unsupported activation: {name}")


def _make_optimizer(name: str, learning_rate: float) -> optimizers.Optimizer:
	name = name.lower()
	if name == "adam":
		return optimizers.Adam(learning_rate=learning_rate)
	if name == "sgd":
		return optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
	raise ValueError(f"Unsupported optimizer: {name}")


def _make_loss(name: str, label_smoothing: float = 0.0) -> losses.Loss:
	name = name.lower()
	if name in ("sparse_categorical_crossentropy", "sparse_ce", "sce"):
		# Some backends don't support label_smoothing for sparse losses
		return losses.SparseCategoricalCrossentropy()
	if name in ("categorical_crossentropy", "ce"):
		return losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
	raise ValueError(f"Unsupported loss: {name}")


def build_cnn_model(
	input_shape: Tuple[int, int, int],
	num_classes: int,
	use_batchnorm: bool = False,
	dropout_rate: float = 0.0,
	l2_weight: float = 0.0,
	add_third_block: bool = False,
	use_gap: bool = False,
	activation_name: str = "relu",
	optimizer_name: str = "adam",
	learning_rate: float = 1e-3,
	loss_name: str = "sparse_categorical_crossentropy",
	label_smoothing: float = 0.0,
) -> keras.Model:
	"""Build and compile a lightweight 2D CNN for digit classification with optional regularization.

	Args:
		input_shape: (height, width, channels), e.g., (40, 200, 1).
		num_classes: e.g., 10.
		use_batchnorm: If True, add BatchNorm after convolutions and dense layers where sensible.
		dropout_rate: If > 0, apply Dropout before the final classification head.
		l2_weight: If > 0, apply L2 weight decay to conv and dense kernels.
		add_third_block: If True, add a third Conv-MaxPool block for extra capacity.
		use_gap: If True, use GlobalAveragePooling2D instead of Flatten (reduces params/overfitting).
		activation_name: Activation to use in conv and dense blocks (relu, elu, leaky_relu).
		optimizer_name: Optimizer name (adam, sgd).
		learning_rate: Optimizer learning rate.
		loss_name: Loss name (sparse_categorical_crossentropy or categorical_crossentropy).
		label_smoothing: Optional label smoothing factor for classification loss.
	"""
	if len(input_shape) != 3:
		raise ValueError("input_shape must be a 3-tuple (height, width, channels)")
	if num_classes <= 0:
		raise ValueError("num_classes must be positive")

	kernel_reg = regularizers.l2(l2_weight) if l2_weight > 0 else None

	inputs = layers.Input(shape=input_shape)
	# Block 1
	x = layers.Conv2D(16, (3, 3), padding="same", activation=None, kernel_regularizer=kernel_reg, kernel_initializer="he_normal")(inputs)
	if use_batchnorm:
		x = layers.BatchNormalization()(x)
	x = _apply_activation(x, activation_name)
	x = layers.MaxPooling2D((2, 2))(x)
	# Block 2
	x = layers.Conv2D(32, (3, 3), padding="same", activation=None, kernel_regularizer=kernel_reg, kernel_initializer="he_normal")(x)
	if use_batchnorm:
		x = layers.BatchNormalization()(x)
	x = _apply_activation(x, activation_name)
	x = layers.MaxPooling2D((2, 2))(x)
	# Optional Block 3
	if add_third_block:
		x = layers.Conv2D(64, (3, 3), padding="same", activation=None, kernel_regularizer=kernel_reg, kernel_initializer="he_normal")(x)
		if use_batchnorm:
			x = layers.BatchNormalization()(x)
		x = _apply_activation(x, activation_name)
		x = layers.MaxPooling2D((2, 2))(x)

	# Head
	if use_gap:
		x = layers.GlobalAveragePooling2D()(x)
	else:
		x = layers.Flatten()(x)
	# Dense head
	x = layers.Dense(64, activation=None, kernel_regularizer=kernel_reg, kernel_initializer="he_normal")(x)
	if use_batchnorm:
		x = layers.BatchNormalization()(x)
	x = _apply_activation(x, activation_name)
	if dropout_rate and dropout_rate > 0:
		x = layers.Dropout(dropout_rate)(x)
	outputs = layers.Dense(num_classes, activation="softmax", kernel_regularizer=kernel_reg)(x)

	model = models.Model(inputs=inputs, outputs=outputs, name="digit_cnn")
	opt = _make_optimizer(optimizer_name, learning_rate)
	loss_obj = _make_loss(loss_name, label_smoothing=label_smoothing)
	model.compile(optimizer=opt, loss=loss_obj, metrics=["accuracy"])
	return model


def _load_local_npy() -> tuple:
	"""Try loading pre-generated numpy arrays from data/ directory."""
	X_train_path = os.path.join("data", "X_train.npy")
	y_train_path = os.path.join("data", "y_train.npy")
	X_test_path = os.path.join("data", "X_test.npy")
	y_test_path = os.path.join("data", "y_test.npy")
	if all(os.path.isfile(p) for p in [X_train_path, y_train_path, X_test_path, y_test_path]):
		X_train = np.load(X_train_path)
		y_train = np.load(y_train_path)
		X_test = np.load(X_test_path)
		y_test = np.load(y_test_path)
		return X_train, y_train, X_test, y_test
	raise FileNotFoundError("Local numpy datasets not found")


def _make_synthetic(n_train: int = 64, n_test: int = 16) -> tuple:
	"""Create a small synthetic dataset for a smoke test."""
	rng = np.random.default_rng(0)
	X_train = rng.normal(size=(n_train, 40, 200, 1)).astype(np.float32)
	y_train = rng.integers(low=0, high=10, size=(n_train,), dtype=np.int64)
	X_test = rng.normal(size=(n_test, 40, 200, 1)).astype(np.float32)
	y_test = rng.integers(low=0, high=10, size=(n_test,), dtype=np.int64)
	return X_train, y_train, X_test, y_test


def _make_callbacks(out_dir: Optional[str] = None) -> list:
	cbs = [
		callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=8, restore_best_weights=True),
		callbacks.ReduceLROnPlateau(monitor="val_accuracy", mode="max", factor=0.5, patience=4, min_lr=1e-5),
	]
	if out_dir:
		os.makedirs(out_dir, exist_ok=True)
		cbs.append(callbacks.CSVLogger(os.path.join(out_dir, "training_log.csv")))
	return cbs


def train_and_save_model(max_len: int = 200, epochs: int = 1, batch_size: int = 32) -> str:
	"""Train the CNN on available data and save the model.

	Order of preference:
	1) Local pre-generated numpy arrays in data/
	2) Hugging Face dataset via pipeline
	3) Synthetic fallback (smoke test)
	"""
	# Try local npy
	try:
		X_train, y_train, X_test, y_test = _load_local_npy()
		loaded_from = "local_npy"
	except Exception:
		# Try HF
		try:
			X_train, y_train = load_fsdd_from_hf(split="train", max_len=max_len)
			X_test, y_test = load_fsdd_from_hf(split="test", max_len=max_len)
			# Add channel dimension
			X_train = X_train[..., np.newaxis]
			X_test = X_test[..., np.newaxis]
			loaded_from = "huggingface"
		except Exception:
			# Synthetic fallback
			X_train, y_train, X_test, y_test = _make_synthetic()
			loaded_from = "synthetic"

	input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
	model = build_cnn_model(input_shape=input_shape, num_classes=10)

	model.fit(
		X_train,
		y_train,
		validation_split=0.1 if len(X_train) > 10 else 0.0,
		epochs=epochs,
		batch_size=batch_size,
		verbose=1,
		callbacks=_make_callbacks(None),
	)

	# Evaluate on test set and print accuracy
	test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
	print(f"Test accuracy ({loaded_from}): {test_acc:.4f}")

	# Ensure models directory exists
	os.makedirs("models", exist_ok=True)
	# Save model in Keras v3 format
	model_path = os.path.join("models", "model.keras")
	model.save(model_path)
	# Also save weights in HDF5 for compatibility with the prompt request
	weights_path = os.path.join("models", "model.weights.h5")
	model.save_weights(weights_path)
	return model_path


def kfold_cv_search(
	max_len: int = 200,
	n_splits: int = 8,
	epochs: int = 20,
	batch_size: int = 64,
	add_deltas: bool = True,
	param_grid: Optional[Dict[str, Iterable[Any]]] = None,
) -> Dict[str, Any]:
	"""Run K-fold CV over a modest hyperparameter grid and return best params.

	Parameters searched include activation, optimizer, learning rate, loss, use_gap, third block, dropout, l2.
	"""
	if param_grid is None:
		# Keep total combinations <= 20 (here: 2*2*2 = 8 combos)
		param_grid = {
			"activation_name": ["relu", "elu"],
			"optimizer_name": ["adam"],
			"learning_rate": [1e-3, 5e-4],
			"loss_name": ["sparse_categorical_crossentropy"],
			"label_smoothing": [0.0],
			"use_gap": [True],
			"add_third_block": [False],
			"use_batchnorm": [True],
			"dropout_rate": [0.3, 0.5],
			"l2_weight": [1e-4],
		}

	# Load data
	X_all, y_all = load_fsdd_from_hf(split="train", max_len=max_len, add_deltas=add_deltas)
	X_all = X_all[..., np.newaxis]

	skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
	def _iter_param_combinations(grid: Dict[str, Iterable[Any]]):
		keys = list(grid.keys())
		def rec(idx: int, current: Dict[str, Any]):
			if idx == len(keys):
				yield current.copy()
				return
			k = keys[idx]
			for v in grid[k]:
				current[k] = v
				yield from rec(idx + 1, current)
		return rec(0, {})

	# Materialize combinations for counting/progress
	combo_list = list(_iter_param_combinations(param_grid))
	total_combos = len(combo_list)
	print(f"[CV] Starting {n_splits}-fold search: epochs={epochs}, batch_size={batch_size}, total_combos={total_combos}")
	sys.stdout.flush()

	best_score = -1.0
	best_params: Dict[str, Any] = {}

	for idx, params in enumerate(combo_list, start=1):
		print(f"[CV] Combo {idx}/{total_combos}: {params}")
		sys.stdout.flush()
		fold_scores: list[float] = []
		for fold_i, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all), start=1):
			print(f"  - Fold {fold_i}/{n_splits}: training...")
			sys.stdout.flush()
			X_tr, X_val = X_all[train_idx], X_all[val_idx]
			y_tr, y_val = y_all[train_idx], y_all[val_idx]
			model = build_cnn_model(
				input_shape=X_tr.shape[1:],
				num_classes=10,
				use_batchnorm=params.get("use_batchnorm", True),
				dropout_rate=params.get("dropout_rate", 0.3),
				l2_weight=params.get("l2_weight", 1e-4),
				add_third_block=params.get("add_third_block", True),
				use_gap=params.get("use_gap", True),
				activation_name=params.get("activation_name", "relu"),
				optimizer_name=params.get("optimizer_name", "adam"),
				learning_rate=params.get("learning_rate", 1e-3),
				loss_name=params.get("loss_name", "sparse_categorical_crossentropy"),
				label_smoothing=params.get("label_smoothing", 0.0),
			)
			h = model.fit(
				X_tr,
				y_tr,
				validation_data=(X_val, y_val),
				epochs=epochs,
				batch_size=batch_size,
				verbose=0,
				callbacks=_make_callbacks(None),
			)
			val_acc_hist = h.history.get("val_accuracy", [0.0])
			val_acc = float(max(val_acc_hist))
			best_epoch = int(np.argmax(val_acc_hist) + 1) if len(val_acc_hist) > 0 else -1
			print(f"    val_acc={val_acc:.4f} (best_epoch={best_epoch})")
			sys.stdout.flush()
			fold_scores.append(val_acc)
		mean_acc = float(np.mean(fold_scores))
		if mean_acc > best_score:
			best_score = mean_acc
			best_params = params.copy()
			print(f"[CV] New best mean acc: {best_score:.4f}")
			sys.stdout.flush()

	print(f"[CV] Best mean accuracy: {best_score:.4f}")
	print(f"[CV] Best params: {best_params}")
	sys.stdout.flush()
	return {"best_params": best_params, "best_score": best_score}


if __name__ == "__main__":
	# Minimal CLI behavior to train, evaluate, and save
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--cv", action="store_true", help="Run 10-fold CV hyperparameter search")
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--max_len", type=int, default=200)
	parser.add_argument("--add_deltas", action="store_true")
	parser.add_argument("--folds", type=int, default=8)
	args = parser.parse_args()

	if args.cv:
		result = kfold_cv_search(max_len=args.max_len, n_splits=args.folds, epochs=args.epochs, batch_size=args.batch_size, add_deltas=args.add_deltas)
		print(result)
	else:
		train_and_save_model(max_len=args.max_len, epochs=args.epochs, batch_size=args.batch_size) 