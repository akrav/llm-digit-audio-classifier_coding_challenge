### Cursor Prompt: Execute Ticket TICKET-2007

Please execute this ticket:

Ticket to execute:
- TICKET_ID: 2007
- TICKET_NAME: TICKET-2007 - Training Curves Visualization (Loss and Accuracy).md
- TICKET_FILE: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Tickets/Sprint 2/TICKET-2007 - Training Curves Visualization (Loss and Accuracy).md

Permanent references (always follow):
- Project Structure: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/structure.md
- Troubleshooting: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Troubleshooting.md
- Sprint Progress: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Sprint-Progress.md

Objective:
- Save training vs validation loss/accuracy charts at the end of training for overfitting diagnosis.

Required steps:
1) Modify `src/train.py` to capture history and write `loss_curve_*.png` and `accuracy_curve_*.png` with legends/labels.
2) Print the saved paths in console output.

Environment:
- Use repo venv; install `matplotlib` if needed.

Output:
- Paths to the generated charts and a short note confirming they were created.

Success criteria:
- Charts are saved under `models/` (or subfolder) and show both train and validation curves clearly. 