### Ticket: Implement Latency Measurement

- **Ticket Number**: TICKET-3002
- **Description**: This ticket adds performance measurement to the inference function. The goal is to accurately measure the time from loading the audio to getting the final prediction.
- **Requirements / Other docs**:
  - **Measurement**: Use `time.perf_counter()` to time the process.
  - **Best Practices**: The measurement should include a "warm-up" run to ensure the results are not skewed by initial loading times.
- **Acceptance Criteria**:
  - The `predict_single_audio` function logs or returns the total time taken for the inference process.
  - The measurement is stable across multiple runs on the same file. 