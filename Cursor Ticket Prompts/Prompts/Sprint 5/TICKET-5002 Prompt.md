### Cursor Prompt: Execute Ticket TICKET-5002

Please execute this ticket:

Ticket to execute:
- TICKET_ID: 5002
- TICKET_NAME: TICKET-5002 - Training speed & caching improvements.md
- TICKET_FILE: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Tickets/Sprint 5/TICKET-5002 - Training speed & caching improvements.md

Permanent references (always follow):
- API Reference: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/API-Reference.md
- Database Schema: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Schemas.md
- Project Structure: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/structure.md
- Troubleshooting: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Troubleshooting.md
- Sprint Progress: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Sprint-Progress.md

Environment and secrets:
- Use .env; never commit secrets.

Objective:
- Add MFCC on-disk caching, minibatch generator, and robust callbacks/logging to speed up training.

Constraints and style:
- Keep feature parity; avoid breaking existing CLI; ensure reproducibility.

Required steps:
1) Read ticket and acceptance criteria.
2) Implement cache in `load_fsdd_from_hf` and flags in `src/train.py` (`--cache`, `--cache_dir`, `--generator`).
3) Save logs under `models/logs/` and set seeds.
4) Verify first vs second run time (cache miss/hit) with 1 epoch.
5) Documentation update.

Output:
- Status, flags added, paths created, and timing improvement note.

Success criteria:
- Cache files are used on repeat runs; epoch wall time reduced; logs saved. 