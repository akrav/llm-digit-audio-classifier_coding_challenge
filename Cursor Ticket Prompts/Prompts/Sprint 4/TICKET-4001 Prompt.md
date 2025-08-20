### Cursor Prompt: Execute Ticket TICKET-4001

Please execute this ticket:

Ticket to execute:
- TICKET_ID: 4001
- TICKET_NAME: TICKET-4001 - Implement Noise Simulation for Robustness Testing.md
- TICKET_FILE: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Tickets/Sprint 4/TICKET-4001 - Implement Noise Simulation for Robustness Testing.md

Permanent references (always follow):
- API Reference: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/API-Reference.md
- Database Schema: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Schemas.md
- Project Structure: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/structure.md
- Troubleshooting: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Troubleshooting.md
- Sprint Progress: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Sprint-Progress.md

Environment and secrets:
- Use .env; never commit secrets.

Objective:
- Integrate noise augmentation (`audiomentations`: AddGaussianNoise, AddBackgroundNoise) into training data pipeline for robustness.

Constraints and style:
- Follow best practices; keep changes scoped; no secrets.

Required steps:
1) Read the ticket and acceptance criteria.
2) Modify data processing to augment a portion of training data with noise.
3) Train a model variant and compare on noisy test data.
4) Documentation: update ticket status, Sprint-Progress, Troubleshooting, and note impact.

Output:
- Concise status, augmentation details, and comparison results.

Success criteria:
- Training runs with noise without errors; improved performance on noisy tests vs. clean-trained model. 