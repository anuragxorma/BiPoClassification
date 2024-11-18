# Decision Tree Classification for BiPo Event Classification

This directory contains the code and training details for the Decision tree models used to classify BiPo events in the OSIRIS liquid scintillator detector, specifically focusing on the BiPo-214 and BiPo-212 decay chains. Below is the breakdown of the workflow, results, and observations.

## Files
- `Model_decision_tree.ipynb`: Defines and trains the decision tree model for BiPo-214 and BiPo-212 chains.
- `decision_tree_best_model.pkl`: Trained Decision tree model for the BiPo-214 chain.
- `decision_tree_best_model2.pkl`: Trained Decision tree model for the BiPo-212 chain.
  
## Dataset Splitting
The dataset was split into training, validation, and test sets as follows:
- **Training Set:** Used to train the model.
- **Validation Set:** Used to fine-tune the hyperparameters.
- **Test Set:** Used to evaluate the model's performance.

## Model Training and Evaluation
### Decision Tree for BiPo214
- **Validation Accuracy:** 0.7113
- **Test Accuracy:** 0.7117
- **Classification Report (Validation):**
  - Precision: 0.8125
  - Recall: 0.7912
  - F1 Score: 0.7914
- **Classification Report (Test):**
  - Precision: 0.8128
  - Recall: 0.7917
  - F1 Score: 0.7917

### Decision Tree for BiPo212
- **Validation Accuracy:** 0.7384
- **Test Accuracy:** 0.7384
- **Classification Report (Validation):**
  - Precision: 0.8343
  - Recall: 0.8142
  - F1 Score: 0.8163
- **Classification Report (Test):**
  - Precision: 0.8342
  - Recall: 0.8143
  - F1 Score: 0.8167

## Hyperparameter Optimization (Grid Search)
Grid search was performed to find the best hyperparameters. The best configurations were:
- **BiPo214:**
  - `max_depth`: 10
  - `min_samples_split`: 2
  - `min_samples_leaf`: 5
  - Cross-Validation Accuracy: 0.79407
- **BiPo212:**
  - `max_depth`: 10
  - `min_samples_split`: 10
  - `min_samples_leaf`: 2
  - Cross-Validation Accuracy: 0.81505

## Confusion Matrices
### BiPo214
|              | Predicted 0 | Predicted 1 | Predicted 2 | Predicted 3 |
|--------------|-------------|-------------|-------------|-------------|
| **Actual 0** | 283,512     | 99,144      | 2           | 48,680      |
| **Actual 1** | 19,103      | 273,413     | 3           | 140,194     |
| **Actual 2** | 0           | 17          | 433,447     | 11          |
| **Actual 3** | 28          | 49,290      | 2           | 381,937     |

### BiPo212
|              | Predicted 0 | Predicted 3 | Predicted 4 | Predicted 5 |
|--------------|-------------|-------------|-------------|-------------|
| **Actual 0** | 309,727     | 47,360      | 74,984      | 0           |
| **Actual 3** | 2,909       | 368,721     | 59,429      | 0           |
| **Actual 4** | 4,883       | 128,170     | 291,756     | 0           |
| **Actual 5** | 0           | 0           | 0           | 422,891     |

## Performance Metrics
### BiPo214
- **Weighted Precision:** 0.8136
- **Weighted Recall:** 0.7938
- **Weighted F1 Score:** 0.7938

### BiPo212
- **Weighted Precision:** 0.8342
- **Weighted Recall:** 0.8143
- **Weighted F1 Score:** 0.8167

## Visualization
- Decision Tree for BiPo-214: Visualized using plot_tree, with depth limited to the best parameter (max_depth=10).
- Confusion Matrices: Displayed as heatmaps with normalized and absolute values.

## Insights and Analysis

1. **Performance Evaluation**: 
   - The Decision Tree models achieved overall test accuracy scores of approximately 71.2% and 73.8% for the BiPo-214 and BiPo-212 datasets, respectively. 
   - Precision, recall, and F1 scores indicated strong classification performance for the dominant classes (2.0 in BiPo-214 and 5.0 in BiPo-212). Weighted metrics (~79–81%) reflect a balanced model capable of handling class imbalances effectively.

2. **Hyperparameter Optimization**:
   - Grid search significantly improved model performance by tuning `max_depth`, `min_samples_split`, and `min_samples_leaf`. These parameters controlled the tree's complexity and ensured a balance between underfitting and overfitting.
   - The optimized depth of 10 strikes a balance between computational efficiency and accuracy.
   - 
3. **Confusion Matrix Analysis**:
   - For BiPo-214:
     - Classes 0.0 and 3.0 exhibited noticeable misclassifications, suggesting overlapping feature spaces or insufficient separation in the decision boundaries.
     - Class 2.0 showed high predictability, with minimal misclassification into other classes.
   - For BiPo-212:
     - Similar misclassification patterns appeared between classes 3.0 and 4.0, which might be due to shared characteristics in their temporal or spatial distributions.
     - Class 5.0 was the most distinct, resulting in high precision and recall.

4. **Model Strengths and Limitations**:
   - **Strengths**:
     - The Decision Tree models are interpretable, providing insight into feature importance for distinguishing classes.
     - They handle non-linear decision boundaries effectively, which is crucial for complex decay event classification.
   - **Limitations**:
     - Misclassifications in overlapping classes indicate potential shortcomings in feature representation or separability.
     - Decision Trees are prone to overfitting; while this was mitigated through hyperparameter tuning, advanced ensemble methods (e.g., Random Forest, Gradient Boosting) could enhance robustness.

5. **Future Directions**:
   - Explore ensemble techniques to improve resilience against noise and achieve better generalization.
   - Investigate additional feature engineering or dimensionality reduction to improve class separability.
   - Extend the analysis to other machine learning models (e.g., SVM, Gradient Boosting) for comparative performance evaluation.

## Usage

1. **Training the Model**:
   - Use the provided code to train the Decision Tree model on the preprocessed datasets for BiPo-214 and BiPo-212.
   - Predefined hyperparameters can be adjusted in the script to explore alternative configurations.

2. **Evaluation**:
   - Model evaluation includes key performance metrics, such as accuracy, precision, recall, and F1 scores, alongside confusion matrices to visualize classification outcomes.

3. **Experimentation**:
   - Users are encouraged to tweak the tree depth, minimum sample splits, or feature sets to test the model’s sensitivity to these parameters.

The Decision Tree models serve as a baseline for event classification, offering interpretable and reasonably accurate results. They provide a foundation for future experimentation and enhancements with more advanced machine learning techniques.

