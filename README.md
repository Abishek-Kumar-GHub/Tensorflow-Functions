# Tensorflow-Functions
Tensor Flow python module's 15+ major functions and dependencies

# Introduction to TensorFlow

TensorFlow is a powerful and widely-used open-source library for numerical computation and machine learning. It offers flexibility and scalability for developing and deploying ML models efficiently. This project demonstrates the implementation of TensorFlow concepts using **Abalone** and **Iris** datasets.

---

## Dependencies Used
1. **TensorFlow**: Core library for building machine learning models.
2. **Scikit-learn**: For preprocessing and evaluation metrics.
3. **Pandas**: For data manipulation and analysis.
4. **NumPy**: For numerical computations.
5. **Matplotlib**: For data visualization.
6. **os**: For handling file paths and directories.

---

## Datasets
### **Abalone Dataset**
- Used for regression tasks (predicting age of abalone).
- CSV format.
- Features include physical measurements of abalones.

### **Iris Dataset**
- Used for classification tasks (classifying iris flower species).
- CSV format.
- Contains features like petal and sepal dimensions.

---

## Implementation Topics
### **Abalone Dataset**
1. **Create TensorFlow Data Pipeline**:
   - Load and preprocess the dataset using `tf.data` API.
   - Normalize the data for efficient training.

2. **Simple Linear Regression Model**:
   - Build a model to predict the age of abalones.
   - Use a single dense layer with linear activation.

3. **Multi-layer Perceptron (MLP) Model**:
   - Build an MLP with multiple hidden layers for better performance.

4. **Dropout Model**:
   - Add dropout layers to reduce overfitting.

5. **Batch Normalization Model**:
   - Normalize intermediate outputs during training to improve stability.

6. **Custom Loss Function**:
   - Implement and use a custom loss function (e.g., mean squared error with penalties).

7. **L1 Regularization Model**:
   - Apply L1 regularization to penalize large coefficients.

8. **L2 Regularization Model**:
   - Apply L2 regularization to avoid overfitting.

9. **Learning Rate Scheduling**:
   - Use learning rate schedules for adaptive training rates.

10. **Model Checkpointing**:
    - Save the best model during training for future use.

11. **Train Model with TensorFlow Callbacks**:
    - Use callbacks like `EarlyStopping` and `ReduceLROnPlateau`.

12. **Custom Metric: Mean Absolute Error**:
    - Define a custom metric for evaluation.

13. **Train with Custom Metric**:
    - Incorporate the custom metric in training.

14. **TensorBoard for Visualization**:
    - Visualize training and evaluation metrics using TensorBoard.

15. **Predict with the Model**:
    - Use the trained model to make predictions.

---

### **Iris Dataset**
16. **Multi-layer Perceptron (MLP)**:
    - Build an MLP for classifying iris species.

17. **Dropout for Iris Classification**:
    - Add dropout to reduce overfitting.

18. **Model Checkpointing**:
    - Save the best classification model during training.

19. **Predict with the Iris Model**:
    - Use the trained model to make predictions on new data.

20. **Summary**:
    - Use to tabulate model file created

---

## Plotting
Visualize data distributions and training metrics:
1. Data histograms for features.
2. Loss curves for training and validation.
3. Predicted vs. actual values for regression.
4. load_model function used to read the model

---
## Outputs
**Analone: Predicted vs Actual **
![image](https://github.com/user-attachments/assets/d244f849-03cd-4c51-abdb-c0e35d3a2bd0)
**Iris Classification**
![image](https://github.com/user-attachments/assets/9f80cb0e-3fff-42a0-b6ec-101827e9bb2b)

---

## Summary
This project explores a wide range of TensorFlow functionalities, from basic data pipelines to advanced techniques like regularization and custom metrics. Both regression and classification tasks are demonstrated using real-world datasets, providing a comprehensive introduction to TensorFlow.
