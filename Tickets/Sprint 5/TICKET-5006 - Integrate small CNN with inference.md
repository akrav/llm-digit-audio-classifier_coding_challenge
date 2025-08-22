### Ticket: Integrate small CNN with inference

- **Ticket Number**: TICKET-5006
- **Description**: Ensure the new small CNN integrates with `src.inference.py` and `src/live_inference.py` so users can switch between architectures.
- **Requirements**:
  - Save small CNN to `models/small_cnn.keras` and add a flag `--arch small_cnn` to inference/live scripts.
  - In `src.inference.py`, load the Keras model and run a forward pass on MFCC inputs.
  - In `src/live_inference.py`, add support for segment-level emission for small CNN (like baseline/w2v2).
- **Acceptance Criteria**:
  - Single-file and live inference work with `--arch small_cnn` and produce accurate predictions. 