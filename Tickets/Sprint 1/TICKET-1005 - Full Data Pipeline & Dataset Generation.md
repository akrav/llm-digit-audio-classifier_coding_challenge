### Ticket: Full Data Pipeline & Dataset Generation

- **Ticket Number**: TICKET-1005
- **Description**: In `src/data_processing.py`, write a main script or function to load the entire FSDD dataset, iterate through each file, call the functions from tickets 1002 to 1004, and store the final features and labels.
- **Requirements / Other docs**:
  - **Dataset Generation**: The script should create a final dataset that includes all pre-processed features and their corresponding labels.
  - **Tools**: `os`, `numpy`.
- **Testing**:
  - **Test Path**: Run the script and verify that the final feature and label arrays have the correct dimensions (e.g., `(num_samples, 40, 200)`) and that the data types are correct.
  - **Acceptance Criteria**: The dataset is generated successfully and has the correct shape and data types. 