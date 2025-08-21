### Ticket: Robust Validation, Cross-Validation, and Baseline Comparison

- **Ticket Number**: TICKET-2006
- **Description**: Address the generalization gap (high train acc, low val/test acc) by implementing robust validation, cross-validation with hyperparameter search, MFCC parameter tuning for 8 kHz audio, and baseline (LogReg/SVM) comparisons. The goal is to diagnose and mitigate overfitting, verify data splits, and confirm that the CNN design and preprocessing are appropriate for FSDD.

- **Why**:
  - Current training shows strong training accuracy but poor validation/test accuracy, indicating overfitting and/or validation split issues.
  - Default librosa MFCC params (n_fft=2048) are not appropriate for 8 kHz short clips and raise warnings; better window/hop improves features.
  - Validation via `validation_split` is not stratified and uses the last fraction before shuffling; we need explicit stratified splits.
  - We should benchmark against classical baselines (LogReg/SVM) as a sanity check on the feature pipeline.

- **Scope / Requirements**:
  1) Validation & Splits
     - Replace `validation_split` in training with an explicit stratified train/val split using `StratifiedShuffleSplit` or `train_test_split(..., stratify=y)`.
     - Ensure no leakage: fit any dataset-level scalers only on the training split, apply to val/test.
     - Log split distributions (class frequencies) for train/val/test.

  2) MFCC Feature Fixes (8 kHz appropriate)
     - Update `extract_mfcc_features` to use `n_fft=512` and `hop_length=128` (or `n_fft=256`, `hop_length=64`), and document rationale for speech at 8 kHz.
     - Add optional deltas (Δ, ΔΔ) and liftering; benchmark with/without.

  3) Regularization & Augmentation
     - Enable SpecAugment-style time/frequency masking (feature-level) or light audio augmentations (noise) for training only.
     - Keep/improve current regularization knobs (BatchNorm, Dropout, L2, GAP). Add a small weight decay default.

  4) Cross-Validation & Hyperparameter Search
     - Implement k-fold CV on the training split (e.g., k=5 stratified) for a small grid/random search over:
       - MFCC params (n_fft/hop, use_deltas)
       - Model configs (BN, Dropout rates, L2, third block, GAP)
       - Learning rate schedule (ReduceLROnPlateau vs cosine)
     - Pick best config by mean CV val accuracy and retrain on full train set (train+val), then evaluate on held-out test.

  5) Baseline Models
     - Implement baselines using scikit-learn: LogisticRegression and SVM on flattened/pool-reduced MFCC features (with proper standardization).
     - Report their test accuracy to validate that the pipeline (features+splits) is sound.

  6) Reporting & Artifacts
     - Print final test accuracy, confusion matrix, and classification report for the best CNN and the best baseline.
     - Save best CNN checkpoint to `models/`.
     - Update `Build Documentation/Sprint-Progress.md` with results and key decisions.

- **Tools**: `keras` (Torch backend), `numpy`, `scikit-learn`, `datasets`, `librosa` (with tuned params), optional `audiomentations`.

- **Acceptance Criteria**:
  - Validation is stratified; no `validation_split` reliance.
  - MFCC warnings eliminated; params tuned for 8 kHz.
  - CV results reported; best config retrained and evaluated on test.
  - Baselines (LogReg/SVM) implemented and compared.
  - Measurable improvement in test accuracy and stable validation metrics.
  - Documentation updated with methodology and outcomes. 