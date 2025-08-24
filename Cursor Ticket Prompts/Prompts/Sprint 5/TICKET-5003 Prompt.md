### Cursor Prompt: Execute Ticket TICKET-5003

Please execute this ticket:

Ticket to execute:
- TICKET_ID: 5003
- TICKET_NAME: TICKET-5003 - Robust cross-validation orchestration.md
- TICKET_FILE: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Tickets/Sprint 5/TICKET-5003 - Robust cross-validation orchestration.md

Permanent references (always follow):
- API Reference: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/API-Reference.md
- Database Schema: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Schemas.md
- Project Structure: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/structure.md
- Troubleshooting: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Troubleshooting.md
- Sprint Progress: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Sprint-Progress.md

Environment and secrets:
- Use .env; never commit secrets.

Objective:
- Add stratified K-fold CV for `small_cnn` with per-fold logs and a summary.

Constraints and style:
- Keep MFCC params consistent; seed and shuffle; save best model.

Required steps:
1) Read ticket and acceptance criteria.
2) Add `--cv`, `--kfolds`, `--cv_epochs`, `--cv_batch_size` flags; implement StratifiedKFold.
3) Save per-fold logs and summary CSV; save best model to models.
4) Run a 3-fold, 1-epoch smoke test.
5) Documentation update.

Output:
- Per-fold accuracies, mean±std; paths to logs and best model.

Success criteria:
- CLI CV completes; artifacts and summary CSV exist; best model saved. 