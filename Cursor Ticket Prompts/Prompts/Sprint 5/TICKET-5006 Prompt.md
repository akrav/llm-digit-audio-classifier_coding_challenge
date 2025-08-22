### Cursor Prompt: Execute Ticket TICKET-5006

Please execute this ticket:

Ticket to execute:
- TICKET_ID: 5006
- TICKET_NAME: TICKET-5006 - Integrate small CNN with inference.md
- TICKET_FILE: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Tickets/Sprint 5/TICKET-5006 - Integrate small CNN with inference.md

Permanent references (always follow):
- API Reference: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/API-Reference.md
- Database Schema: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Schemas.md
- Project Structure: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/structure.md
- Troubleshooting: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Troubleshooting.md
- Sprint Progress: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Sprint-Progress.md

Environment and secrets:
- Use .env; never commit secrets.

Objective:
- Wire the `small_cnn` into `src/inference.py` and `src/live_inference.py` with CLI options.

Constraints and style:
- Maintain consistent preprocessing params; support toggle/PTT/VAD; ensure low latency.

Required steps:
1) Read ticket and acceptance criteria.
2) Add `--arch keras` and `--keras_model` to inference/live; implement softmax and confidence handling.
3) Test with a HF sample and live mic; record commands.
4) Documentation update.

Output:
- Inference and live commands for Keras model; quick latency stats.

Success criteria:
- Keras path works end-to-end in single-file inference and live mic. 