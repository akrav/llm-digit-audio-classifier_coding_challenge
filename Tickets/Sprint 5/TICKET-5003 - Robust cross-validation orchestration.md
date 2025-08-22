### Ticket: Robust cross-validation orchestration

- **Ticket Number**: TICKET-5003
- **Description**: Implement stratified K-fold CV for the small CNN with per-fold early stopping and clear reporting.
- **Requirements**:
  - K-fold (k=8–10) stratified splits; ensure label balance.
  - For each fold: train with EarlyStopping; record best val accuracy and epoch.
  - Aggregate metrics: mean, std, and per-fold table; save to `models/reports/cv_summary.csv`.
  - Optionally persist best-per-fold weights in `models/cv/` with timestamps.
- **Acceptance Criteria**:
  - A single command runs CV and produces the CSV and console summary.
  - Mean CV accuracy reported; artifacts saved. 