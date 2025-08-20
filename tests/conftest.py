import os
import sys

# Force Keras 3 to use the PyTorch backend for tests (avoids TensorFlow dependency)
os.environ.setdefault("KERAS_BACKEND", "torch")

# Add project root to sys.path for tests
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
	sys.path.insert(0, ROOT_DIR) 