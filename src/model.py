from typing import Tuple

import keras
from keras import layers, models, optimizers


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
	outputs = layers.Dense(num_classes, activation="softmax")(x)

	model = models.Model(inputs=inputs, outputs=outputs, name="digit_cnn")
	model.compile(
		optimizer=optimizers.Adam(),
		loss="sparse_categorical_crossentropy",
		metrics=["accuracy"],
	)
	return model 