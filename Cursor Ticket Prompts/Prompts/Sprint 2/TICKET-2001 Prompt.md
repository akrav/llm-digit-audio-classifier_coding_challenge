### Cursor Prompt: Execute Ticket TICKET-2001

Please execute this ticket:

Ticket to execute:
- TICKET_ID: 2001
- TICKET_NAME: TICKET-2001 - Model Definition.md
- TICKET_FILE: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Tickets/Sprint 2/TICKET-2001 - Model Definition.md

Permanent references (always follow):
- API Reference: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/API-Reference.md
- Database Schema: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Schemas.md
- Project Structure: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/structure.md
- Troubleshooting: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Troubleshooting.md
- Sprint Progress: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Sprint-Progress.md

Environment and secrets:
- Use .env (never commit secrets). If new keys are needed, add placeholders to .env.example and document in README.

Objective:
- Implement `build_cnn_model(input_shape, num_classes)` in `src/model.py` using Conv2D, MaxPooling2D, Flatten, Dense; compile with softmax output of 10 units. Add tests.

Constraints and style:
- Follow Engineering Best Practices (naming, error handling, logging). Keep changes scoped; no secrets.

Required steps:
1) Read the ticket and acceptance criteria.
2) Implement the model function and compile it.
3) Tests: create `tests/test_model.py` asserting output shape `(None, 10)` and reasonable layer count.
4) Documentation: update ticket status, Sprint-Progress, Troubleshooting, and structure if needed.
5) Testing: run `pytest -q`.

Output:
- Concise summary of changes and how to run tests.

Success criteria:
- Tests passing; docs updated; acceptance criteria met. 