### Cursor Prompt: Execute Ticket TICKET-1004

Please execute this ticket:

Ticket to execute:
- TICKET_ID: 1004
- TICKET_NAME: TICKET-1004 - Padding and Normalization.md
- TICKET_FILE: /Users/adam/Documents/GitHub/CloudWalk_Techinical_Challenge/Tickets/Sprint 1/TICKET-1004 - Padding and Normalization.md

Permanent references (always follow):
- API Reference: /Users/adam/Documents/GitHub/CloudWalk_Techinical_Challenge/Build Documentation/API-Reference.md
- Database Schema: /Users/adam/Documents/GitHub/CloudWalk_Techinical_Challenge/Build Documentation/Schemas.md
- Project Structure: /Users/adam/Documents/GitHub/CloudWalk_Techinical_Challenge/Build Documentation/structure.md
- Troubleshooting: /Users/adam/Documents/GitHub/CloudWalk_Techinical_Challenge/Build Documentation/Troubleshooting.md
- Sprint Progress: /Users/adam/Documents/GitHub/CloudWalk_Techinical_Challenge/Build Documentation/Sprint-Progress.md

Environment and secrets:
- Use .env (never commit secrets). If new keys are needed, add placeholders to .env.example and document in README.

Objective:
- Implement `pad_features(mfcc_array, max_len)` and `normalize_features(features)` using StandardScaler; add tests for shape and normalization (~0 mean, ~1 std).

Constraints and style:
- Follow best practices; keep changes scoped; no secrets.

Required steps:
1) Read the ticket and acceptance criteria.
2) Implement functions and tests in `tests/test_data_processing.py`.
3) Update docs: ticket status, Sprint-Progress, Troubleshooting, structure if needed.
4) Testing: run `pytest -q`.

Output:
- Concise status, files touched, and test commands.

Success criteria:
- Acceptance criteria met; tests passing; documentation updated. 