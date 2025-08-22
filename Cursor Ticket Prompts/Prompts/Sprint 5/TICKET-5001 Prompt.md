### Cursor Prompt: Execute Ticket TICKET-5001

Please execute this ticket:

Ticket to execute:
- TICKET_ID: 5001
- TICKET_NAME: TICKET-5001 - Small SOTA CNN architecture for MFCCs.md
- TICKET_FILE: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Tickets/Sprint 5/TICKET-5001 - Small SOTA CNN architecture for MFCCs.md

Permanent references (always follow):
- API Reference: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/API-Reference.md
- Database Schema: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Schemas.md
- Project Structure: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/structure.md
- Troubleshooting: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Troubleshooting.md
- Sprint Progress: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Sprint-Progress.md

Environment and secrets:
- Use .env; never commit secrets.

Objective:
- Implement compact depthwise-separable + SE small CNN for MFCCs; add CLI to train/evaluate; prep for integration with inference/live.

Constraints and style:
- Follow ML best practices; keep changes scoped; ensure Torch backend for Keras.

Required steps:
1) Read ticket and acceptance criteria.
2) Add `build_small_cnn` in `src/model.py` (depthwise-separable, SE, BN, Dropout, GAP).
3) Add `--arch small_cnn` to training (`src/train.py`) and save to models.
4) Unit smoke: build model, print summary, 1-epoch train.
5) Documentation update: ticket status, progress, troubleshooting.

Output:
- Concise status, training command, saved artifact path(s).

Success criteria:
- Model builds, trains, and saves; ready for later CV/grid-search and inference integration. 