### Ticket: Training curves & comparison plots

- **Ticket Number**: TICKET-5005
- **Description**: Generate and save training/validation loss and accuracy plots for the small CNN runs and CV aggregates.
- **Requirements**:
  - Save per-run curves to `models/plots/{timestamp}_loss.png` and `{timestamp}_acc.png`.
  - For CV: plot mean±std bands across epochs where available.
  - Export CSV logs (epoch, loss, acc, val_loss, val_acc) to `models/logs/`.
  - Optional comparison plot vs baseline SVM and Wav2Vec2 (final metrics bar chart).
- **Acceptance Criteria**:
  - Plots and CSV logs produced for a complete training run; images render correctly. 