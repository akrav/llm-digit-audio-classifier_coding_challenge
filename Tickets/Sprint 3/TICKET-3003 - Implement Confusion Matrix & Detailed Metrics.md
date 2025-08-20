### Ticket: Implement Confusion Matrix & Detailed Metrics

- **Ticket Number**: TICKET-3003
- **Description**: This ticket enhances the model evaluation. The goal is to provide a detailed breakdown of the model's performance beyond simple accuracy.
- **Requirements / Other docs**:
  - **Evaluation Script**: Modify the evaluation script in `src/model.py`.
  - **Metrics**: Implement the generation of a confusion matrix and print the Precision, Recall, and F1-Score for each digit class.
  - **Tools**: Use `scikit-learn`'s `confusion_matrix` and `classification_report`.
- **Acceptance Criteria**:
  - Running the evaluation script produces a clear, formatted confusion matrix in the console.
  - The output includes class-by-class metrics (Precision, Recall, F1-Score). 