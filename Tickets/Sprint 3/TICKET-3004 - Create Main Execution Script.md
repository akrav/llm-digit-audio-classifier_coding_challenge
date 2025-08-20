### Ticket: Create Main Execution Script

- **Ticket Number**: TICKET-3004
- **Description**: This ticket creates the main entry point for the application. The script will orchestrate all parts of the pipeline, allowing a user to train the model, evaluate it, and perform inference from a single command.
- **Requirements / Other docs**:
  - **Main Script**: `src/main.py`.
  - **Logic**: The script should include command-line arguments to trigger different modes (e.g., `train`, `evaluate`, `predict`).
  - **Integration**: It will call the functions from `src/data_processing.py`, `src/model.py`, and `src/inference.py` as needed.
- **Acceptance Criteria**:
  - Executing `python src/main.py --train` successfully trains the model.
  - Executing `python src/main.py --evaluate` prints the evaluation metrics.
  - Executing `python src/main.py --predict path/to/audio.wav` returns a prediction. 