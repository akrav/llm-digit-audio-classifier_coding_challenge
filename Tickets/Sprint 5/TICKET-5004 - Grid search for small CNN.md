### Ticket: Grid search for small CNN

- **Ticket Number**: TICKET-5004
- **Description**: Implement a focused grid search (≤20 combinations) over key hyperparameters for the small CNN.
- **Requirements**:
  - Search over: learning rate, dropout, L2, activation (relu/elu), use_SE (on/off), depth multiplier.
  - Limit combinations to ≤20; log parameters and scores.
  - Save ranked results to `models/reports/grid_results.csv`; print top-3 to console.
  - Allow running grid on a single train/val split or via nested CV (optional).
- **Acceptance Criteria**:
  - Grid search completes within reasonable time and exports CSV with scores and params. 