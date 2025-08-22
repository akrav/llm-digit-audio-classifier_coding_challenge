### Ticket: Small SOTA CNN architecture for MFCCs

- **Ticket Number**: TICKET-5001
- **Description**: Implement a compact CNN using depthwise-separable convolutions (MobileNet-style), Squeeze-and-Excitation (SE) blocks, BatchNorm, Dropout, and Global Average Pooling. The model should target competitive accuracy (≥95% on FSDD test) with faster training and lower parameter count than the current CNN.
- **Requirements / Other docs**:
  - Build in `src/model.py` a new builder `build_small_cnn(input_shape, num_classes, ...)` with options for BN, SE, dropout, L2, GAP.
  - Replace heavy Flatten+Denses with GAP+small head.
  - He init, relu/elu toggles; compile with Adam / sparse CE.
  - Ensure compatibility with existing training scripts and inference shapes `(40, 200, 1)`.
- **Acceptance Criteria**:
  - Model summary includes depthwise-separable layers and GAP; params < 1M.
  - Trains end-to-end on FSDD and reaches ≥95% test accuracy in < prior epoch time.
  - Saves to `models/small_cnn.keras` and integrates with inference. 