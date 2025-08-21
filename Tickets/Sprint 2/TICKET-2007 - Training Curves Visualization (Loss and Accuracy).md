### Ticket: Training Curves Visualization (Loss and Accuracy)

- **Ticket Number**: TICKET-2007
- **Description**: Add clear visualizations of training vs validation loss and accuracy to diagnose overfitting/underfitting. Persist charts to disk after each training run for quick inspection.

- **Scope / Requirements**:
  - Modify `src/train.py` to:
    - Capture training history (loss, accuracy, val_loss, val_accuracy)
    - Save two charts using `matplotlib`: `loss_curve.png` and `accuracy_curve.png`
    - Save charts under `models/` (or `models/artifacts/`), including a timestamp in filenames
    - Print paths to the saved charts at the end of training
  - Ensure charts render cleanly (legends, axis labels, title with key hyperparameters)
  - Keep functionality headless (no GUI); simply write `.png` files

- **Tools**: `matplotlib`, `numpy`

- **Testing**:
  - Run a short training (1–3 epochs) and verify the images are created in `models/` with readable axes/legends

- **Acceptance Criteria**:
  - `loss_curve_*.png` and `accuracy_curve_*.png` are created after training and compare training vs validation curves
  - Filenames and console output clearly indicate where the images are saved 