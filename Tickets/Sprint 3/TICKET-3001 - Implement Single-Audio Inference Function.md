### Ticket: Implement Single-Audio Inference Function

- **Ticket Number**: TICKET-3001
- **Description**: This ticket covers the implementation of the core inference function. It will take a pre-trained model and a new audio file path, preprocess the audio using the same pipeline as the training data, and return a single predicted digit.
- **Requirements / Other docs**:
  - **Function**: `predict_single_audio(audio_path, model_path)` in `src/inference.py`.
  - **Preprocessing**: The function must replicate the exact preprocessing steps from Sprint 1 (MFCC extraction, padding, normalization).
  - **Model Loading**: The function must load the saved model from the `models/` directory.
- **Acceptance Criteria**:
  - A test file can be successfully passed to the function, and it returns a valid digit prediction (0-9).
  - The function handles errors for invalid file paths. 