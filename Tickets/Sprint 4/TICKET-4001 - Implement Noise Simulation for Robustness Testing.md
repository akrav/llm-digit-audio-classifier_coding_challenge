### Ticket: Implement Noise Simulation for Robustness Testing

- **Ticket Number**: TICKET-4001
- **Description**: This ticket adds a data augmentation step to the training pipeline. The goal is to make the model more robust by training it on audio data with simulated noise.
- **Requirements / Other docs**:
  - **Tool**: `audiomentations`.
  - **Implementation**: Modify the data processing script to apply a random noise transformation (e.g., `AddGaussianNoise`, `AddBackgroundNoise`) to a portion of the training data.
- **Acceptance Criteria**:
  - The training script can successfully apply noise transformations without errors.
  - A model trained with noise shows improved performance on noisy test data compared to a model trained on clean data. 