### Cursor Prompt: Execute Ticket TICKET-2005

Please execute this ticket:

Ticket to execute:
- TICKET_ID: 2005
- TICKET_NAME: TICKET-2005 - Achieve 95%+ Test Accuracy with Regularized Optimization.md
- TICKET_FILE: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Tickets/Sprint 2/TICKET-2005 - Achieve 95%+ Test Accuracy with Regularized Optimization.md

Permanent references (always follow):
- Project Structure: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/structure.md
- Troubleshooting: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Troubleshooting.md
- Sprint Progress: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Sprint-Progress.md

Objective:
- Iteratively optimize the CNN to surpass 95% test accuracy on FSDD without overfitting. Use regularization, augmentation, early stopping, and proper validation.

Required steps:
1) Add/modify model options in `src/model.py` (BatchNorm, Dropout, L2, extra conv, GAP, LR schedule).
2) Create `src/train.py` or extend training entrypoint to load HF dataset via `load_fsdd_from_hf`, split train/val, and train with callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger). Seed deterministically.
3) Add optional training-time augmentations for audio in `src/data_processing.py` or a helper.
4) Evaluate strictly on test split; produce accuracy, confusion matrix, and classification report.
5) Save best model to `models/` and record metrics in Sprint-Progress. Keep inference latency low.

Environment:
- Use the repo venv; install extras if needed: `pip install datasets audiomentations`.

Output:
- Summary of best configuration, final metrics (train/val/test), and saved model path.

Success criteria:
- Test accuracy ≥ 95%; checkpoint saved; docs updated; no overfitting (reasonable generalization). 