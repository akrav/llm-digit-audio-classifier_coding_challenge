### Cursor Prompt: Execute Ticket TICKET-5004

Please execute this ticket:

Ticket to execute:
- TICKET_ID: 5004
- TICKET_NAME: TICKET-5004 - Grid search for small CNN.md
- TICKET_FILE: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Tickets/Sprint 5/TICKET-5004 - Grid search for small CNN.md

Permanent references (always follow):
- API Reference: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/API-Reference.md
- Database Schema: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Schemas.md
- Project Structure: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/structure.md
- Troubleshooting: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Troubleshooting.md
- Sprint Progress: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Sprint-Progress.md

Environment and secrets:
- Use .env; never commit secrets.

Objective:
- Implement a focused grid search (≤20 combos) over `small_cnn` hyperparameters with 3–5 folds.

Constraints and style:
- Keep search compact for speed; log each combo; early stopping active.

Required steps:
1) Read ticket and acceptance criteria.
2) Add CLI to run grid search and pick best by mean val accuracy.
3) Save results table to `models/logs/grid_search.csv`; save best model.
4) Documentation update.

Output:
- Best config, per-combo metrics, saved model path.

Success criteria:
- Grid search runs within reasonable time; artifacts saved; best model reproducible. 