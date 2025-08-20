### Ticket: Implement Live Microphone Integration

- **Ticket Number**: TICKET-4002
- **Description**: This ticket implements a live, real-time classification system. It will capture audio from the user's microphone, preprocess it, and send it to the inference module for prediction.
- **Requirements / Other docs**:
  - **Tools**: `sounddevice`, `src/inference.py`.
  - **Logic**: Create a new script, `src/live_inference.py`, that captures audio in chunks, applies the same preprocessing pipeline, and feeds it to the `predict_single_audio` function for real-time classification.
- **Acceptance Criteria**:
  - The script correctly captures audio from the microphone and provides real-time predictions.
  - The latency is low enough for a fluid user experience. 