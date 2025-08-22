### **Sprint 5: Small SOTA CNN, Faster Training, and Robust Tuning**

This sprint focuses on building a compact, state-of-the-art architecture for digit classification that matches Wav2Vec2 performance while keeping latency and complexity low. We will harden the training loop with efficient data handling, stratified cross-validation, grid search, and clear training visualizations, and ensure the model integrates with the existing inference pipeline (single-file and live mic).

-----

### **Major Functionality Targeted**

- Small, production-friendly CNN (MobileNet-style depthwise-separable blocks + GAP)
- Faster training via feature caching, smaller footprints, and callbacks
- Stratified K-fold CV orchestration with early stopping per fold
- Focused grid search over key hyperparameters (≤20 combos)
- Automatic plots (loss/accuracy) and comparison reporting
- Seamless integration with inference (single/HF/live)

-----

### **Planned Tickets**

- TICKET-5001 — Small SOTA CNN architecture for MFCCs (depthwise-separable + SE, GAP)
- TICKET-5002 — Training speed & caching improvements (feature caching, loaders, callbacks)
- TICKET-5003 — Robust cross-validation orchestration (stratified, early-stopped folds, report)
- TICKET-5004 — Grid search for small CNN (≤20 combos) with summary CSV
- TICKET-5005 — Training curves & comparison plots (per-run and aggregate)
- TICKET-5006 — Integrate small CNN with live inference and single-file inference
- TICKET-5007 — Documentation & Sprint 5 progress update

-----

### **Acceptance**

- Small CNN trains to competitive accuracy on FSDD (≥95% test) with faster epochs than prior CNN
- CV + grid search run end-to-end and produce summary artifacts
- Plots and reports saved to `models/plots/` with timestamps
- Inference scripts accept the new CNN model seamlessly 