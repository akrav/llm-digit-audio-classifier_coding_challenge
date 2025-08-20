### **Sprint 2: Core Model Training**

This sprint's purpose is to implement the classification model, train it on the processed data from Sprint 1, and conduct initial evaluation to ensure it is learning effectively.

-----

### **Major Functionality Delivered**

  * **Model Definition:** A lightweight 2D CNN model is defined and compiled.
  * **Training Script:** A script to load the pre-processed data and train the model.
  * **Model Persistence:** The trained model is saved to a file for later use in the inference module.
  * **Initial Validation:** The model is evaluated on a test set to confirm it is learning.

-----

### **Sprint Tickets**

The tickets are ordered to build upon the data pipeline from Sprint 1 and produce a trained, functional model.

-----

**Ticket Name:** Model Definition
\<br\> **Ticket Number:** TICKET-2001
\<br\> **Description:** In `src/model.py`, define a function `build_cnn_model(input_shape, num_classes)` that returns the compiled lightweight 2D CNN model. The model should include `Conv2D`, `MaxPooling2D`, `Flatten`, and `Dense` layers.
\<br\> **Requirements/Other docs:**

  * **Model:** The model must be a 2D CNN.
  * **Architecture:** The model should include the specified layers. The final layer must have 10 units with `softmax` activation.
    \<br\> **Tools:** `TensorFlow/Keras`.
    \<br\> **Testing:**
  * **Test Path:** Write a unit test in `tests/test_model.py` that calls `build_cnn_model()` and asserts the model's output shape is `(None, 10)` and that it has the correct number of layers.
  * **Acceptance Criteria:** The test passes.

-----

**Ticket Name:** Model Training Script
\<br\> **Ticket Number:** TICKET-2002
\<br\> **Description:** In `src/model.py`, implement a training script that loads the pre-generated data from Sprint 1, trains the model, and saves the trained model to the `models/` directory.
\<br\> **Requirements/Other docs:**

  * **Training Loop:** The script must use `model.fit()` to train the model.
  * **Model Saving:** The trained model should be saved to `models/model.h5`.
    \<br\> **Tools:** `TensorFlow/Keras`.
    \<br\> **Testing:**
  * **Test Path:** Run the script and check that the `models/` folder contains the saved model file (e.g., `model.h5`).
  * **Acceptance Criteria:** The model file is created successfully.

-----

**Ticket Name:** Initial Evaluation and Validation
\<br\> **Ticket Number:** TICKET-2003
\<br\> **Description:** After training, add a step to the training script to evaluate the model's accuracy on the test set.
\<br\> **Requirements/Other docs:**

  * **Evaluation:** The script must use `model.evaluate()` on the test dataset.
  * **Logging:** The final accuracy should be printed to the console.
    \<br\> **Tools:** `TensorFlow/Keras`.
    \<br\> **Testing:**
  * **Test Path:** Run the script and log the final validation accuracy. Check if the accuracy is better than a random guess (e.g., \> 10%).
  * **Acceptance Criteria:** The model shows evidence of learning, with an accuracy greater than a random baseline.

-----

**Ticket Name:** Documentation & Progress Update
\<br\> **Ticket Number:** TICKET-2004
\<br\> **Description:** Update `Sprint-Progress.md` with the completed tickets and their outcomes. Document any initial roadblocks or insights in `Troubleshooting.md`.
\<br\> **Requirements/Other docs:**

  * **Documentation:** `Sprint-Progress.md` and `Troubleshooting.md` must be updated.
    \<br\> **Tools:** Markdown editor.
    \<br\> **Testing:**
  * **Test Path:** Manual verification of document content.
  * **Acceptance Criteria:** The documents are updated with accurate information from this sprint.