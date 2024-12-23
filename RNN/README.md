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

Both the SimpleRNN and LSTM models follow a similar structure:

- **Input Layer**: Takes sequences of features for training, which are generated in the `RNN_preprocessing.py` script. These sequences are derived from the raw features and are fed into the models for training.
- **RNN/LSTM Layers**:
  - SimpleRNN/LSTM with 64 units and tanh activation function.
  - SimpleRNN/LSTM with 32 units and tanh activation function.
- **Dense Layer**: 16 units with ReLU activation.
- **Output Layer**: Softmax activation function for classification.

The choice of this architecture was made due to the limitations of available hardware and memory resources, as well as the absence of a GPU for model training. Given these constraints, the architecture was designed to balance performance with computational efficiency. 
The models use **categorical cross-entropy** as the loss function and **Adam** as the optimizer. To improve training stability and prevent overfitting, **early stopping** and **learning rate reduction** callbacks are employed.

## Training Details

### Hyperparameters
- **Epochs**: Maximum of 200 (training may stop early due to callbacks)
- **Batch Size**: 10,000
- **Learning Rate**: Dynamically reduced with a ReduceLROnPlateau callback

### Callbacks
1. **ReduceLROnPlateau**: Reduces learning rate by 50% if validation loss plateaus for 4 consecutive epochs.
2. **EarlyStopping**: Stops training when validation loss does not improve for 10 epochs, restoring the best model weights.

### SimpleRNN Model Performance Metrics

#### 1. BiPo-214 Chain
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

#### 2. BiPo-212 Chain
- **Epochs**: 55/200
- **Loss**: 0.4342
- **Accuracy**: 0.8162
- **Validation Loss**: 0.4342
- **Validation Accuracy**: 0.8163

**Confusion Matrix**:
When analyzing the confusion matrix for this chain, it is important to remember that there was a label change during the conversion from one-hot encoding to numeric labels. The original truth labels were as follows:

- `0` (Internal Background)
- `3` (External Background)
- `4` (Bi-212)
- `5` (Po-212)

After conversion to numeric labels for the confusion matrix, the labels were adjusted as follows:

- `0` remains `0` (Int. Background)
- `3` is changed to `1` (Ext Background)
- `4` is changed to `2` (Bi-212)
- `5` is changed to `3` (Po-212)

Therefore, the confusion matrix should be interpreted with this mapping in mind.

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

### LSTM Model Performance Metrics

#### 1. BiPo-214 Chain
- **Epochs**: 63/200
- **Loss**: 0.4719
- **Accuracy**: 0.7954
- **Validation Loss**: 0.4727
- **Validation Accuracy**: 0.7949

**Confusion Matrix**:
|              | Predicted 0 | Predicted 1 | Predicted 2 | Predicted 3 |
|--------------|-------------|-------------|-------------|-------------|
| **Actual 0** | 286,095     | 97,947      | 3        | 47,293      |
| **Actual 1** | 21,371      | 275,670     | 6          | 135,665     |
| **Actual 2** | 0           | 7           | 433,462     | 4           |
| **Actual 3** | 45          | 52,495      | 12          | 378,700     |

| Metric               | Score    |
|----------------------|----------|
| Precision (Macro)    | 0.8129    |
| Recall (Macro)       | 0.7946    |
| F1 Score (Macro)     | 0.7948    |
| Precision (Weighted) | 0.813    |
| Recall (Weighted)    | 0.7947    |
| F1 Score (Weighted)  | 0.7949    |

#### 2. BiPo-212 Chain
- **Epochs**: 63/200
- **Loss**: 0.4342
- **Accuracy**: 0.8160
- **Validation Loss**: 0.4341
- **Validation Accuracy**: 0.8164

**Confusion Matrix**:
When analyzing the confusion matrix for this chain, it is important to remember that there was a label change during the conversion from one-hot encoding to numeric labels same as in the previous model.
After conversion to numeric labels for the confusion matrix, the labels were adjusted as follows:

- `0` remains `0` (Int. Background)
- `3` is changed to `1` (Ext Background)
- `4` is changed to `2` (Bi-212)
- `5` is changed to `3` (Po-212)

Therefore, the confusion matrix should be interpreted with this mapping in mind.

|              | Predicted 0 | Predicted 1 | Predicted 2 | Predicted 3 |
|--------------|-------------|-------------|-------------|-------------|
| **Actual 0** | 312,697     | 47,573      | 71,793        | 4      |
| **Actual 1** | 3,438     | 370,418     | 57,199          | 3     |
| **Actual 2** | 7,338      | 128,613      | 288,847     | 9           |
| **Actual 3** | 0          | 0      | 2          | 422,887     |

| Metric               | Score    |
|----------------------|----------|
| Precision (Macro)    | 0.8339    |
| Recall (Macro)       | 0.8157    |
| F1 Score (Macro)     | 0.8178    |
| Precision (Weighted) | 0.8337    |
| Recall (Weighted)    | 0.8153    |
| F1 Score (Weighted)  | 0.8174    |

## Model Comparison
Both SimpleRNN and LSTM models have shown strong performance in classifying BiPo events. The key difference between the two models lies in their architecture, with LSTMs being better suited for capturing long-term dependencies in sequences even though that isn't quite apparent in this specific case.

| Model      | BiPo-214 Accuracy | BiPo-212 Accuracy |
|------------|-------------------|-------------------|
| SimpleRNN  | 0.7954            | 0.8162            |
| LSTM       | 0.7954           | 0.8160           |

## Insights and Analysis

1. **Performance Evaluation**: 
   - The **SimpleRNN models** achieved an overall accuracy of approximately 79.5% for the BiPo-214 chain and 81.6% for the BiPo-212 chain on the validation set. These results indicate solid performance in distinguishing BiPo events across both chains. The precision, recall, and F1 scores are consistent and well-balanced, suggesting that the SimpleRNN models offer reliable performance across all categories, with no significant bias toward any specific class. 
   - The **LSTM models** perform similarly, achieving nearly identical accuracy and metrics to the SimpleRNN models. While LSTMs are designed to capture long-term dependencies in temporal data, this advantage is only slightly reflected in this case, suggesting that short-term dependencies may be sufficient for distinguishing between classes in the current dataset.

2. **Confusion Matrix Analysis**:
   - *BiPo-214 Chain (Confusion Matrix 1)*: 
     - For the **SimpleRNN model**, the confusion matrix reveals misclassifications between adjacent decay categories, where events from one class are misclassified into a neighboring class. This could be due to the similarity in feature distributions between adjacent decay categories or limited model capacity in distinguishing these subtle differences. Despite these misclassifications, the overall performance remains strong, with relatively few errors in comparison to the correct predictions.
     - The **LSTM model** exhibits similar patterns in the confusion matrix, with misclassifications occurring in adjacent categories. The LSTM’s ability to handle dependencies in sequences could help to reduce these misclassifications over longer training periods, as it better models temporal dependencies within the data.
   - *BiPo-212 Chain (Confusion Matrix 2)*:
     - In the **SimpleRNN model**, misclassifications are less frequent compared to the BiPo-214 chain, with higher accuracy in the BiPo-212 chain. The confusion matrix suggests that the model performs well in distinguishing categories with higher representation in the dataset, particularly where the feature space is more distinct.
     - The **LSTM model** demonstrates similar trends, effectively classifying categories with distinct features. This model may have a slight advantage in scenarios where temporal continuity aids classification, but this advantage is minor here given the close performance alignment with the SimpleRNN model.

3. **Learning Dynamics**: 
   - Both the **SimpleRNN** and **LSTM models** were set to run for a maximum of 200 epochs. However, early stopping was employed to prevent overfitting. For the **SimpleRNN**, training was halted at epoch 69 for the BiPo-214 chain and at epoch 55 for the BiPo-212 chain. The **LSTM models** also underwent early stopping at epoch 63 for both chains. This strategy helped maintain stable validation performance across both models, ensuring that the models did not degrade in quality due to excessive training.

4. **Limitations and Future Work**: While the SimpleRNN and LSTM models have demonstrated solid performance, there is potential for improvement.
   
## Future Work
- Hyperparameter tuning for both RNN and LSTM models to further enhance accuracy.
- Investigation into more advanced sequence models (e.g., GRU, bidirectional RNN).
- Integration of additional features into the models to improve classification.

