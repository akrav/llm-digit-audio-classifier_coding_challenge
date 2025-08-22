### Ticket: Training speed & caching improvements

- **Ticket Number**: TICKET-5002
- **Description**: Reduce epoch time and memory using feature caching, efficient data loaders, and robust callbacks.
- **Requirements**:
  - Add an on-disk feature cache (e.g., `data/cache_mfcc_{params}.npy`) to avoid recomputing MFCCs for repeated runs.
  - Implement minibatch generators to avoid loading all data into RAM when not needed.
  - Standard callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger.
  - Ensure reproducibility (seeds) and progress bars.
- **Acceptance Criteria**:
  - Repeat run with identical parameters reuses cache and shows reduced preprocessing time.
  - Epoch wall time improved vs prior CNN run; logs saved to `models/logs/`. 