### Cursor Prompt: Execute Ticket TICKET-2006

Please execute this ticket:

Ticket to execute:
- TICKET_ID: 2006
- TICKET_NAME: TICKET-2006 - Robust Validation, Cross-Validation, and Baseline Comparison.md
- TICKET_FILE: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Tickets/Sprint 2/TICKET-2006 - Robust Validation, Cross-Validation, and Baseline Comparison.md

Permanent references (always follow):
- Project Structure: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/structure.md
- Troubleshooting: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Troubleshooting.md
- Sprint Progress: /Users/adam/Documents/GitHub/llm-digit-audio-classifier_coding_challenge/Build Documentation/Sprint-Progress.md

Objective:
- Fix generalization: implement stratified validation, k-fold CV+hyperparameter search for MFCC and model settings, add robust regularization/augmentation, and compare to classical baselines (LogReg/SVM). Target stable validation and improved test accuracy.

Required steps:
1) Splits:
   - Replace `validation_split` with explicit stratified train/val splitting; log class distributions.
2) MFCC params:
   - Tune for 8 kHz (e.g., n_fft=512, hop_length=128); optionally add deltas/liftering. Remove large-n_fft warnings.
3) Regularization & augmentation:
   - Ensure EarlyStopping, ReduceLROnPlateau, ModelCheckpoint; enable GAP, L2, Dropout. Add feature/audio masking for training only.
4) Cross-validation & search:
   - Implement k-fold CV (e.g., k=5) across MFCC/model hyperparams and LR schedule; pick best by mean CV val accuracy.
5) Baselines:
   - Add scikit-learn Pipeline baselines (LogReg/SVM) on MFCCs with StandardScaler; report test accuracy.
6) Reporting:
   - Retrain best config on train+val; evaluate on test with accuracy, confusion matrix, and classification report. Save best model.
7) Documentation:
   - Update Sprint-Progress with findings and final metrics.

Environment:
- Use repo venv; install as needed: `pip install datasets scikit-learn audiomentations`.

Output:
- Summary of best configuration, CV results, final test metrics (CNN and baseline), and saved model path(s).

Success criteria:
- Stratified validation in place; warnings resolved; CV reported; baselines compared; improved test accuracy with stable val performance; docs updated. 