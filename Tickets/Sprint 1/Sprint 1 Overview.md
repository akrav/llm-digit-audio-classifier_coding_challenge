### **Sprint 1: Data Processing Pipeline**

This sprint's purpose is to establish the project's technical bedrock by implementing a robust data processing pipeline. This sprint focuses on the foundational work of loading, transforming, and preparing the FSDD dataset for model training.

-----

### **Major Functionality Delivered**

  * **Project Initialization**: A structured project repository with key documentation files and folders.
  * **Data Ingestion**: A reusable function to load and resample audio files from the FSDD dataset.
  * **Feature Extraction**: The core logic to extract MFCC features from raw audio.
  * **Data Preparation**: Functions for padding, normalization, and generating the full dataset for training.

-----

### **Sprint Tickets**

The tickets are ordered to resolve dependencies, starting with project setup and moving through to the implementation of the full data pipeline.

-----

**Ticket Name:** Project Setup and Initial Folder Structure
\<br\> **Ticket Number:** TICKET-1001
\<br\> **Description:** Create the project repository. Initialize `README.md`, `requirements.txt`, `Sprint-Progress.md`, `Troubleshooting.md`, and `Structure.md`. Create the main folders: `src/`, `data/`, `models/`, and `tests/`.
\<br\> **Requirements/Other docs:**

  * **Folder Structure:** The specified folders must be created at the project root.
  * **Initial Files:** The listed markdown and text files must be present and contain basic placeholder content.
    \<br\> **Tools:** Git, your code editor.
    \<br\> **Testing:**
  * **Test Path:** Manually verify that the folder structure and files are created as specified.
  * **Acceptance Criteria:** The project directory is correctly structured.

-----

**Ticket Name:** Data Loading and Resampling
\<br\> **Ticket Number:** TICKET-1002
\<br\> **Description:** In `src/data_processing.py`, implement a function `load_audio(file_path)` that loads an audio file and resamples it to a target sample rate (8kHz to match FSDD).
\<br\> **Requirements/Other docs:**

  * **Function:** `load_audio(file_path)` in `src/data_processing.py`.
  * **Logic:** Use `librosa.load()` to load the audio and resample to 8000 Hz.
    \<br\> **Tools:** `librosa`.
    \<br\> **Testing:**
  * **Test Path:** Create a `tests/test_data_processing.py` file. Write a test function that calls `load_audio()` on a test WAV file and asserts that the output has the correct sample rate.
  * **Acceptance Criteria:** The test passes. The function correctly loads and resamples both standard FSDD files and files with different original sample rates.

-----

**Ticket Name:** MFCC Feature Extraction
\<br\> **Ticket Number:** TICKET-1003
\<br\> **Description:** In `src/data_processing.py`, implement a function `extract_mfcc_features(audio, sr)` that extracts 40 MFCCs from an audio array.
\<br\> **Requirements/Other docs:**

  * **Function:** `extract_mfcc_features(audio, sr)`.
  * **Features:** The function must extract 40 Mel-Frequency Cepstral Coefficients.
    \<br\> **Tools:** `librosa`.
    \<br\> **Testing:**
  * **Test Path:** Write a test function in `test_data_processing.py` that calls `extract_mfcc_features()` and asserts the output array's shape is `(40, num_frames)`.
  * **Acceptance Criteria:** The test passes without errors.

-----

**Ticket Name:** Padding and Normalization
\<br\> **Ticket Number:** TICKET-1004
\<br\> **Description:** In `src/data_processing.py`, implement a function `pad_features(mfcc_array, max_len)` to pad/truncate the MFCC array to a fixed length (e.g., 200). Also, implement a function `normalize_features(features)` to normalize the MFCC features.
\<br\> **Requirements/Other docs:**

  * **Functions:** `pad_features(mfcc_array, max_len)` and `normalize_features(features)`.
  * **Logic:** Padding should be applied to achieve a consistent shape. Normalization should use a `StandardScaler` to achieve a mean of \~0 and a standard deviation of \~1.
    \<br\> **Tools:** `numpy`, `scikit-learn` (`StandardScaler`).
    \<br\> **Testing:**
  * **Test Path:** Write tests in `test_data_processing.py` to check for the correct padded shape and to verify that normalization results in a mean of \~0 and a standard deviation of \~1.
  * **Acceptance Criteria:** The tests pass.

-----

**Ticket Name:** Full Data Pipeline & Dataset Generation
\<br\> **Ticket Number:** TICKET-1005
\<br\> **Description:** In `src/data_processing.py`, write a main script or function to load the entire FSDD dataset, iterate through each file, call the functions from tickets 1002 to 1004, and store the final features and labels.
\<br\> **Requirements/Other docs:**

  * **Dataset Generation:** The script should create a final dataset that includes all pre-processed features and their corresponding labels.
    \<br\> **Tools:** `os`, `numpy`.
    \<br\> **Testing:**
  * **Test Path:** Run the script and verify that the final feature and label arrays have the correct dimensions (e.g., `(num_samples, 40, 200)`) and that the data types are correct.
  * **Acceptance Criteria:** The dataset is generated successfully and has the correct shape and data types.

-----

**Ticket Name:** Initial Documentation & Progress Update
\<br\> **Ticket Number:** TICKET-1006
\<br\> **Description:** Update `Sprint-Progress.md` with the completed tickets and their outcomes. Document any initial roadblocks or insights in `Troubleshooting.md`.
\<br\> **Requirements/Other docs:**

  * **Documentation:** `Sprint-Progress.md` and `Troubleshooting.md` must be updated.
    \<br\> **Tools:** Markdown editor.
    \<br\> **Testing:**
  * **Test Path:** Manual verification of document content.
  * **Acceptance Criteria:** The documents are updated with accurate information from this sprint.
