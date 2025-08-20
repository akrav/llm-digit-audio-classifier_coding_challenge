### Ticket: MFCC Feature Extraction

- **Ticket Number**: TICKET-1003
- **Description**: In `src/data_processing.py`, implement a function `extract_mfcc_features(audio, sr)` that extracts 40 MFCCs from an audio array.
- **Requirements / Other docs**:
  - **Function**: `extract_mfcc_features(audio, sr)`.
  - **Features**: The function must extract 40 Mel-Frequency Cepstral Coefficients.
  - **Tools**: `librosa`.
- **Testing**:
  - **Test Path**: Write a test function in `tests/test_data_processing.py` that calls `extract_mfcc_features()` and asserts the output array's shape is `(40, num_frames)`.
  - **Acceptance Criteria**: The test passes without errors. 