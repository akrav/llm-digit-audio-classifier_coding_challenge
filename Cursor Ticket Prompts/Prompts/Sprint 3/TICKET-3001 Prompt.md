### Cursor Prompt: Execute Ticket TICKET-3001

Please execute this ticket:

Ticket to execute:
- TICKET_ID: 3001
- TICKET_NAME: TICKET-3001 - Implement Single-Audio Inference Function.md
- TICKET_FILE: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Tickets/Sprint 3/TICKET-3001 - Implement Single-Audio Inference Function.md

Permanent references (always follow):
- API Reference: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/API-Reference.md
- Database Schema: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Schemas.md
- Project Structure: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/structure.md
- Troubleshooting: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Troubleshooting.md
- Sprint Progress: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Sprint-Progress.md

Environment and secrets:
- Use .env; never commit secrets.

Objective:
- Implement `predict_single_audio(audio_path, model_path)` in `src/inference.py`, replicating training preprocessing and returning a digit 0-9. Handle invalid paths.

Constraints and style:
- Follow best practices; keep changes scoped; no secrets.

Required steps:
1) Read the ticket and acceptance criteria.
2) Implement preprocessing reuse and model loading; return prediction.
3) Add basic error handling and logging.
4) Documentation: update ticket status, Sprint-Progress, Troubleshooting.

Output:
- Concise status and how to run the function with a sample file.

Success criteria:
- Valid digit returned for test file; errors handled gracefully. 