### Ticket: Stratified Splits, Feature Normalization, and Regularization Hardening

- **Ticket Number**: TICKET-2008
- **Description**: Implement strictly stratified splits (no speaker grouping), dataset-level feature normalization fit on training only, and strengthen regularization settings (BN, Dropout, L2, GAP, SpecAugment) to reduce overfitting while preserving low latency.

- **Scope / Requirements**:
  1) Splits
     - Use `train_test_split(..., stratify=y)` for train/val and keep the provided test split as held-out
     - Remove any reliance on Keras `validation_split` and any speaker grouping logic
     - Log class distribution for train/val/test

  2) Feature Normalization
     - Normalize feature tensors consistently: when producing utterance-level features for baselines, fit a `StandardScaler` (training only) and apply to val/test
     - For CNN inputs, keep per-feature standardization over time windows (as done) but avoid per-sample leakage when computing dataset-level scalers

  3) Regularization Hardening
     - Ensure BatchNorm + Dropout (0.3–0.5), L2 weight decay (default 1e-4), GlobalAveragePooling2D head
     - Keep EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
     - Feature-level SpecAugment (time/frequency masking) on training only (bound widths sensibly)

  4) Flags and Defaults
     - Expose CLI flags in `src/train.py` for Dropout, L2, GAP, BN; default to a conservative regularized configuration
     - Default MFCC/log-mel params tuned for 8 kHz (no n_fft warnings)

- **Tools**: `scikit-learn`, `keras`, `numpy`, `librosa`

- **Testing**:
  - Verify stratified distribution and no use of `validation_split`
  - Quick smoke run (2–3 epochs) shows stable val accuracy (not collapsing to ~0)

- **Acceptance Criteria**:
  - Stratified splits in place; class distributions logged
  - Feature normalization follows “fit on train, apply to val/test”
  - Regularization defaults strengthened and documented
  - Evidence of improved generalization (val accuracy not near-zero and test accuracy improves relative to prior baseline) 