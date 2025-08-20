### Ticket: Model Training Script

- **Ticket Number**: TICKET-2002
- **Description**: In `src/model.py`, implement a training script that loads the pre-generated data from Sprint 1, trains the model, and saves the trained model to the `models/` directory.
- **Requirements / Other docs**:
  - **Training Loop**: The script must use `model.fit()` to train the model.
  - **Model Saving**: The trained model should be saved to `models/model.h5`.
  - **Tools**: `TensorFlow/Keras`.
- **Testing**:
  - **Test Path**: Run the script and check that the `models/` folder contains the saved model file (e.g., `model.h5`).
  - **Acceptance Criteria**: The model file is created successfully. 