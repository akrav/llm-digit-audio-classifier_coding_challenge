### Ticket: Initial Evaluation and Validation

- **Ticket Number**: TICKET-2003
- **Description**: After training, add a step to the training script to evaluate the model's accuracy on the test set.
- **Requirements / Other docs**:
  - **Evaluation**: The script must use `model.evaluate()` on the test dataset.
  - **Logging**: The final accuracy should be printed to the console.
  - **Tools**: `TensorFlow/Keras`.
- **Testing**:
  - **Test Path**: Run the script and log the final validation accuracy. Check if the accuracy is better than a random guess (e.g., > 10%).
  - **Acceptance Criteria**: The model shows evidence of learning, with an accuracy greater than a random baseline. 