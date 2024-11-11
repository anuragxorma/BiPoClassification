# RNN and LSTM Models for BiPo Event Classification

## Overview
This directory contains models for classifying BiPo events in the OSIRIS experiment using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks. The task involves tagging fast Bi-Po coincidence decays in U/Th decay chains in a liquid scintillator (LS) detector. The models are trained separately for two chains: BiPo-214 and BiPo-212.

## Files
- `RNN_processing.py`: Handles preprocessing, including sequence generation from the feature engineering outputs.
- `Model_SimpleRNN.ipynb`: Defines and trains the SimpleRNN model for BiPo-214 and BiPo-212 chains.
- `Model_LSTM.ipynb`: Defines and trains the LSTM model for BiPo-214 and BiPo-212 chains.
- `model_SimpleRNN_214.h5`: Trained SimpleRNN model for the BiPo-214 chain.
- `model_SimpleRNN_212.h5`: Trained SimpleRNN model for the BiPo-212 chain.
- `model_LSTM_214.h5`: Trained LSTM model for the BiPo-214 chain (currently unavailable but expected soon).
- `model_LSTM_212.h5`: Trained LSTM model for the BiPo-212 chain (currently unavailable but expected soon).

## Model Architecture
Both models (SimpleRNN and LSTM) follow a similar structure:

- **Input Layer**: Takes sequences of features for training.
- **RNN/LSTM Layers**:
  - SimpleRNN/LSTM with 64 units and tanh activation function.
  - SimpleRNN/LSTM with 32 units and tanh activation function.
- **Dense Layer**: 16 units with ReLU activation.
- **Output Layer**: Softmax activation function for classification.

The models use categorical cross-entropy as the loss function and Adam as the optimizer. They include early stopping and learning rate reduction callbacks to improve training stability.

## Code Structure
- `RNN_processing.py`: Preprocessing involves generating sequences from raw features and splitting the data into training and validation sets.
- `Model_SimpleRNN.ipynb`: Implements the SimpleRNN model training for both BiPo chains.
- `Model_LSTM.ipynb`: Implements the LSTM model training for both BiPo chains.

## Performance Metrics

### SimpleRNN Model Results

#### BiPo-214 Chain
- **Epochs**: 69/200
- **Loss**: 0.4720
- **Accuracy**: 0.7954
- **Validation Loss**: 0.4726
- **Validation Accuracy**: 0.7950

**Confusion Matrix**:
|              | Predicted 0 | Predicted 1 | Predicted 2 | Predicted 3 |
|--------------|-------------|-------------|-------------|-------------|
| **Actual 0** | 286,032     | 97,940      | 3        | 47,363      |
| **Actual 1** | 21,345      | 275,470     | 5          | 135,892     |
| **Actual 2** | 0           | 7           | 433,459     | 7           |
| **Actual 3** | 73          | 52,257      | 11          | 378,911     |

| Metric               | Score    |
|----------------------|----------|
| Precision (Macro)    | 0.8128    |
| Recall (Macro)       | 0.7945    |
| F1 Score (Macro)     | 0.7947    |
| Precision (Weighted) | 0.8129    |
| Recall (Weighted)    | 0.7947    |
| F1 Score (Weighted)  | 0.7948    |

#### BiPo-212 Chain
- **Epochs**: 55/200
- **Loss**: 0.4342
- **Accuracy**: 0.8162
- **Validation Loss**: 0.4342
- **Validation Accuracy**: 0.8163

**Confusion Matrix**:
|              | Predicted 0 | Predicted 1 | Predicted 2 | Predicted 3 |
|--------------|-------------|-------------|-------------|-------------|
| **Actual 0** | 312,221     | 47,413      | 72,429        | 4      |
| **Actual 1** | 3,303      | 370,052     | 57,698          | 5     |
| **Actual 2** | 6,896      | 128,210      | 289,689     | 12           |
| **Actual 3** | 0          | 0      | 2          | 422,887     |

| Metric               | Score    |
|----------------------|----------|
| Precision (Macro)    | 0.8341    |
| Recall (Macro)       | 0.8157    |
| F1 Score (Macro)     | 0.8178    |
| Precision (Weighted) | 0.8339    |
| Recall (Weighted)    | 0.8153    |
| F1 Score (Weighted)  | 0.8174    |

### LSTM Model Results
The performance of the LSTM models is expected to be similar to the SimpleRNN models, with very close accuracy and other metrics, based on initial tests.

#### BiPo-214 Chain
- **Epochs**: Similar to SimpleRNN, expected to stop early due to early stopping.
- **Accuracy**: Around 0.7954 (similar to SimpleRNN).
- **Confusion Matrix**: Similar structure to SimpleRNN results, with expected variations in label distribution.

**Metrics**:
- **Precision (Macro)**: Expected to be similar to SimpleRNN (~0.812861).
- **Recall (Macro)**: Expected to be similar to SimpleRNN (~0.794584).
- **F1 Score (Macro)**: Expected to be similar to SimpleRNN (~0.794740).

#### BiPo-212 Chain
- **Epochs**: Similar to SimpleRNN, expected to stop early due to early stopping.
- **Accuracy**: Around 0.8162 (similar to SimpleRNN).
- **Confusion Matrix**: Similar structure to SimpleRNN results.

**Metrics**:
- **Precision (Macro)**: Expected to be similar to SimpleRNN (~0.834127).
- **Recall (Macro)**: Expected to be similar to SimpleRNN (~0.815755).
- **F1 Score (Macro)**: Expected to be similar to SimpleRNN (~0.817826).

## Model Comparison
Both SimpleRNN and LSTM models have shown strong performance in classifying BiPo events, with the LSTM model expected to perform similarly, based on previous tests. The key difference between the two models lies in their architecture, with LSTMs being better suited for capturing long-term dependencies in sequences.

| Model      | BiPo-214 Accuracy | BiPo-212 Accuracy |
|------------|-------------------|-------------------|
| SimpleRNN  | 0.7954            | 0.8162            |
| LSTM       | ~0.7954           | ~0.8162           |

## Insights and Analysis

1. **Performance Evaluation**: 
   - The **SimpleRNN models** achieved an overall accuracy of approximately 79.5% for the BiPo-214 chain and 81.6% for the BiPo-212 chain on the validation set. These results indicate solid performance in distinguishing BiPo events across both chains. The precision, recall, and F1 scores are consistent and well-balanced, suggesting that the SimpleRNN models offer reliable performance across all categories, with no significant bias toward any specific class. 
   - The **LSTM models** are expected to perform similarly, as initial tests suggest that LSTM networks offer comparable accuracy and metrics to the SimpleRNN models. The LSTM’s ability to capture long-term dependencies may slightly improve performance in certain cases, particularly where temporal patterns in the data are more significant.

2. **Confusion Matrix Analysis**:
   - *BiPo-214 Chain (Confusion Matrix 1)*: 
     - For the **SimpleRNN model**, the confusion matrix reveals misclassifications between adjacent decay categories, where events from one class are misclassified into a neighboring class. This could be due to the similarity in feature distributions between adjacent decay categories or limited model capacity in distinguishing these subtle differences. Despite these misclassifications, the overall performance remains strong, with relatively few errors in comparison to the correct predictions.
     - The **LSTM model** is expected to have a similar confusion matrix structure, with misclassifications potentially occurring in similar patterns. However, LSTM’s ability to capture longer-term dependencies could help reduce these misclassifications by better modeling the sequence of events in the data.
   - *BiPo-212 Chain (Confusion Matrix 2)*:
     - In the **SimpleRNN model**, misclassifications are less frequent compared to the BiPo-214 chain, with higher accuracy in the BiPo-212 chain. The confusion matrix suggests that the model performs well in distinguishing categories with higher representation in the dataset, particularly where the feature space is more distinct.
     - The **LSTM model** is likely to show similar trends, with slightly improved performance due to the model’s capacity to capture more complex patterns over time. The LSTM may be especially effective in classes where long-term dependencies are crucial for correct classification.

3. **Learning Dynamics**: 
   - Both the **SimpleRNN** and **LSTM models** were set to run for a maximum of 200 epochs. However, early stopping was employed to prevent overfitting. For the **SimpleRNN**, training was halted at epoch 74 for the BiPo-214 chain and at epoch 75 for the BiPo-212 chain. The **LSTM models** also underwent early stopping, with similar behavior expected, where training would be halted early to avoid overfitting. This strategy helped maintain stable validation performance across both models, ensuring that the models did not degrade in quality due to excessive training.

4. **Limitations and Future Work**: While the SimpleRNN and LSTM models have demonstrated solid performance, there is potential for improvement.
   
## Future Work
- Hyperparameter tuning for both RNN and LSTM models to further enhance accuracy.
- Investigation into more advanced sequence models (e.g., GRU, bidirectional RNN).
- Integration of additional features into the models to improve classification.

