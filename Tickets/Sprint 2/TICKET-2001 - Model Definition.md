### Ticket: Model Definition

- **Ticket Number**: TICKET-2001
- **Description**: In `src/model.py`, define a function `build_cnn_model(input_shape, num_classes)` that returns the compiled lightweight 2D CNN model. The model should include `Conv2D`, `MaxPooling2D`, `Flatten`, and `Dense` layers.
- **Requirements / Other docs**:
  - **Model**: The model must be a 2D CNN.
  - **Architecture**: The model should include the specified layers. The final layer must have 10 units with `softmax` activation.
  - **Tools**: `TensorFlow/Keras`.
- **Testing**:
  - **Test Path**: Write a unit test in `tests/test_model.py` that calls `build_cnn_model()` and asserts the model's output shape is `(None, 10)` and that it has the correct number of layers.
  - **Acceptance Criteria**: The test passes. 