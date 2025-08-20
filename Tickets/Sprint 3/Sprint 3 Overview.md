### **Sprint 3 Overview: Inference and Advanced Evaluation**

This sprint's purpose is to build the final prototype's real-time inference capability and to perform a detailed evaluation of the trained model. We will implement the core logic for single-audio prediction, add latency measurement, and generate advanced performance metrics to understand the model's strengths and weaknesses.

-----

### **Major Functionality Delivered**

  * **Real-Time Inference Module:** A standalone script capable of loading the trained model and predicting a single spoken digit from an audio file.
  * **Performance Measurement:** The system will accurately measure and report the end-to-end inference latency.
  * **Advanced Evaluation Metrics:** The model's performance will be thoroughly evaluated using a confusion matrix and class-specific metrics (Precision, Recall, F1-Score).
  * **End-to-End Testing:** The entire pipeline, from raw audio input to final prediction, will be tested to ensure seamless operation.

-----

### **Sprint Tickets**

The tickets are ordered to resolve dependencies, starting with the implementation of the core inference logic and moving through to the final evaluation metrics.

-----

**Ticket Name:** Implement Single-Audio Inference Function
\<br\> **Ticket Number:** TICKET-3001
\<br\> **Description:** This ticket covers the implementation of the core inference function. It will take a pre-trained model and a new audio file path, preprocess the audio using the same pipeline as the training data, and return a single predicted digit.
\<br\> **Requirements/Other docs:**

  * **Function:** `predict_single_audio(audio_path, model_path)` in `src/inference.py`.
  * **Preprocessing:** The function must replicate the exact preprocessing steps from Sprint 1 (MFCC extraction, padding, normalization).
  * **Model Loading:** The function must load the saved model from the `models/` directory.
    \<br\> **Acceptance Criteria:**
  * A test file can be successfully passed to the function, and it returns a valid digit prediction (0-9).
  * The function handles errors for invalid file paths.

-----

**Ticket Name:** Implement Latency Measurement
\<br\> **Ticket Number:** TICKET-3002
\<br\> **Description:** This ticket adds performance measurement to the inference function. The goal is to accurately measure the time from loading the audio to getting the final prediction.
\<br\> **Requirements/Other docs:**

  * **Measurement:** Use `time.perf_counter()` to time the process.
  * **Best Practices:** The measurement should include a "warm-up" run to ensure the results are not skewed by initial loading times.
    \<br\> **Acceptance Criteria:**
  * The `predict_single_audio` function logs or returns the total time taken for the inference process.
  * The measurement is stable across multiple runs on the same file.

-----

**Ticket Name:** Implement Confusion Matrix & Detailed Metrics
\<br\> **Ticket Number:** TICKET-3003
\<br\> **Description:** This ticket enhances the model evaluation. The goal is to provide a detailed breakdown of the model's performance beyond simple accuracy.
\<br\> **Requirements/Other docs:**

  * **Evaluation Script:** Modify the evaluation script in `src/model.py`.
  * **Metrics:** Implement the generation of a confusion matrix and print the Precision, Recall, and F1-Score for each digit class.
  * **Tools:** Use `scikit-learn`'s `confusion_matrix` and `classification_report`.
    \<br\> **Acceptance Criteria:**
  * Running the evaluation script produces a clear, formatted confusion matrix in the console.
  * The output includes class-by-class metrics (Precision, Recall, F1-Score).

-----

**Ticket Name:** Create Main Execution Script
\<br\> **Ticket Number:** TICKET-3004
\<br\> **Description:** This ticket creates the main entry point for the application. The script will orchestrate all parts of the pipeline, allowing a user to train the model, evaluate it, and perform inference from a single command.
\<br\> **Requirements/Other docs:**

  * **Main Script:** `src/main.py`.
  * **Logic:** The script should include command-line arguments to trigger different modes (e.g., `train`, `evaluate`, `predict`).
  * **Integration:** It will call the functions from `src/data_processing.py`, `src/model.py`, and `src/inference.py` as needed.
    \<br\> **Acceptance Criteria:**
  * Executing `python src/main.py --train` successfully trains the model.
  * Executing `python src/main.py --evaluate` prints the evaluation metrics.
  * Executing `python src/main.py --predict path/to/audio.wav` returns a prediction.

-----

**Ticket Name:** Final Documentation and Progress Update
\<br\> **Ticket Number:** TICKET-3005
\<br\> **Description:** This ticket is for finalizing documentation and updating the project's status.
\<br\> **Requirements/Other docs:**

  * **Update Documents:** Update the `Sprint-Progress.md`, `Troubleshooting.md`, and `Structure.md` documents to reflect the completed tasks in this sprint.
  * **Review:** Ensure that all previous tasks are fully documented.
    \<br\> **Acceptance Criteria:**
  * All project documentation is up-to-date and accurate.
  * All completed tasks in this sprint are marked as complete.