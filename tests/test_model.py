import numpy as np
import pytest

from src.model import build_cnn_model


def test_build_cnn_model_output_shape_and_layers():
	input_shape = (40, 200, 1)
	num_classes = 10
	model = build_cnn_model(input_shape, num_classes)
	assert model.output_shape == (None, num_classes)
	# Expect at least Conv2D, MaxPool, Conv2D, MaxPool, Flatten, Dense, Dense
	assert len(model.layers) >= 7 