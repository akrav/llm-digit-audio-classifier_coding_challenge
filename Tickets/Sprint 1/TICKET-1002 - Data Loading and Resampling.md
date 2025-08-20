### Ticket: Data Loading and Resampling

- **Ticket Number**: TICKET-1002
- **Description**: In `src/data_processing.py`, implement a function `load_audio(file_path)` that loads an audio file and resamples it to a target sample rate (8kHz to match FSDD).
- **Requirements / Other docs**:
  - **Function**: `load_audio(file_path)` in `src/data_processing.py`.
  - **Logic**: Use `librosa.load()` to load the audio and resample to 8000 Hz.
  - **Tools**: `librosa`.
  - **Data Source**: Use the Free Spoken Digit Dataset from Hugging Face: https://huggingface.co/datasets/mteb/free-spoken-digit-dataset/viewer/default/train?views%5B%5D=train
- **Testing**:
  - **Test Path**: Create a `tests/test_data_processing.py` file. Write a test function that calls `load_audio()` on a test WAV file and asserts that the output has the correct sample rate.
  - **Acceptance Criteria**: The test passes. The function correctly loads and resamples both standard FSDD files and files with different original sample rates. 