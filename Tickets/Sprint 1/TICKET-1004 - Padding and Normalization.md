### Ticket: Padding and Normalization

- **Ticket Number**: TICKET-1004
- **Description**: In `src/data_processing.py`, implement a function `pad_features(mfcc_array, max_len)` to pad/truncate the MFCC array to a fixed length (e.g., 200). Also, implement a function `normalize_features(features)` to normalize the MFCC features.
- **Requirements / Other docs**:
  - **Functions**: `pad_features(mfcc_array, max_len)` and `normalize_features(features)`.
  - **Logic**: Padding should be applied to achieve a consistent shape. Normalization should use a `StandardScaler` to achieve a mean of ~0 and a standard deviation of ~1.
  - **Tools**: `numpy`, `scikit-learn` (`StandardScaler`).
- **Testing**:
  - **Test Path**: Write tests in `tests/test_data_processing.py` to check for the correct padded shape and to verify that normalization results in a mean of ~0 and a standard deviation of ~1.
  - **Acceptance Criteria**: The tests pass. 