
### Sprint 1 Progress

- TICKET-1001 - Project Setup and Initial Folder Structure: Completed on 2025-08-20 14:53
  - Created folders: src/, data/, models/, tests/
  - Added files: README.md, requirements.txt, Build Documentation/*.md
  - Added sanity tests: tests/test_structure.py

- TICKET-1002 - Data Loading and Resampling: Completed on 2025-08-20 14:53
  - Implemented `src/data_processing.py::load_audio` with resampling to 8000 Hz
  - Added tests: tests/test_data_processing.py

- TICKET-1003 - MFCC Feature Extraction: Completed on 2025-08-20 14:53
  - Implemented `extract_mfcc_features(audio, sr)` returning 40 MFCCs
  - Added tests asserting shape `(40, num_frames)`

- TICKET-1004 - Padding and Normalization: Completed on 2025-08-20 14:53
  - Implemented `pad_features(mfcc_array, max_len)` and `normalize_features(features)`
  - Added tests for padding shape and normalization stats (~0 mean, ~1 std)

- TICKET-1005 - Full Data Pipeline & Dataset Generation: Completed on 2025-08-20 14:53
  - Implemented `preprocess_audio_to_features`, `generate_dataset_from_filepaths`, and `load_fsdd_from_hf`
  - Added tests verifying dataset batch shapes
