### Cursor Prompt: Execute Ticket TICKET-2008

Please execute this ticket:

Ticket to execute:
- TICKET_ID: 2008
- TICKET_NAME: TICKET-2008 - Stratified Splits, Feature Normalization, and Regularization Hardening.md
- TICKET_FILE: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Tickets/Sprint 2/TICKET-2008 - Stratified Splits, Feature Normalization, and Regularization Hardening.md

Permanent references (always follow):
- Project Structure: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/structure.md
- Troubleshooting: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Troubleshooting.md
- Sprint Progress: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Sprint-Progress.md

Objective:
- Ensure stratified train/val splits (no grouping), proper feature normalization (fit on train, apply to val/test), and stronger default regularization to reduce overfitting.

Required steps:
1) Replace any Keras validation_split with explicit stratified splits; log class distributions.
2) Ensure scalers are fit on training only for baseline features; apply to val/test.
3) Keep BN+Dropout+L2+GAP, EarlyStopping/ReduceLROnPlateau; enable SpecAugment on training only.
4) Expose flags in `src/train.py` for these settings; document sensible defaults.

Environment:
- Use repo venv.

Output:
- Concise summary of split distributions, normalization approach, and final metrics.

Success criteria:
- Stratified split confirmed; normalization scheme correct; improved val/test stability and accuracy; docs updated. 