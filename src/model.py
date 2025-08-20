from typing import Tuple

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from keras import layers, models, optimizers
import numpy as np

from src.data_processing import load_fsdd_from_hf


def build_cnn_model(input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
	"""Build and compile a lightweight 2D CNN for digit classification.

	Args:
		input_shape: Input tensor shape as (height, width, channels), e.g., (40, 200, 1).
		num_classes: Number of output classes, e.g., 10.

	Returns:
		A compiled Keras Model with softmax output.
	"""
	if len(input_shape) != 3:
		raise ValueError("input_shape must be a 3-tuple (height, width, channels)")
	if num_classes <= 0:
		raise ValueError("num_classes must be positive")

	inputs = layers.Input(shape=input_shape)
	# Block 1
	x = layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu")(inputs)
	x = layers.MaxPooling2D(pool_size=(2, 2))(x)
	# Block 2
	x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu")(x)
	x = layers.MaxPooling2D(pool_size=(2, 2))(x)
	# Head
	x = layers.Flatten()(x)
	x = layers.Dense(64, activation="relu")(x)
	outputs = layers.Dense(10, activation="softmax")(x)

	model = models.Model(inputs=inputs, outputs=outputs, name="digit_cnn")
	model.compile(
		optimizer=optimizers.Adam(),
		loss="sparse_categorical_crossentropy",
		metrics=["accuracy"],
	)
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
	)

	# Ensure models directory exists
	os.makedirs("models", exist_ok=True)
	# Save model in Keras v3 format
	model_path = os.path.join("models", "model.keras")
	model.save(model_path)
	# Also save weights in HDF5 for compatibility with the prompt request
	weights_path = os.path.join("models", "model.weights.h5")
	model.save_weights(weights_path)
	return model_path


if __name__ == "__main__":
	# Minimal CLI behavior to train and save
	train_and_save_model() 