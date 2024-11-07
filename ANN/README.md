# ANN Model for BiPo Event Classification

This directory contains the code and training details for the Artificial Neural Network (ANN) models used to classify BiPo events in the OSIRIS liquid scintillator detector, specifically focusing on the BiPo-214 and BiPo-212 decay chains. These models aim to detect fast Bi-Po coincidence decays in uranium and thorium decay chains, enhancing the sensitivity of OSIRIS to extremely low U/Th concentrations in the detector.

## Model Architecture

The ANN model has a straightforward Multi-Layer Perceptron (MLP) architecture, designed as follows:
- **Input Layer**: Configured to match the dimensions of the preprocessed, feature-engineered input data. (For detailed information on input features, refer to the README file in the `Shared_dir` directory.)
- **Two Hidden Layers**: Each with 256 neurons and ReLU activation, optimized for the classification task.
- **Output Layer**: Softmax activation, providing probabilities for multiclass classification.

The models are compiled with categorical crossentropy loss, and the Adam optimizer, which adapts learning rates for efficient convergence.

## Training Details

### Hyperparameters
- **Epochs**: Maximum of 200 (training may stop early due to callbacks)
- **Batch Size**: 10,000
- **Learning Rate**: Dynamically reduced with a ReduceLROnPlateau callback

### Callbacks
1. **ReduceLROnPlateau**: Reduces learning rate by 50% if validation loss plateaus for 4 consecutive epochs.
2. **EarlyStopping**: Stops training when validation loss does not improve for 10 epochs, restoring the best model weights.

### Training Results

Training for both models stopped early due to the EarlyStopping callback, with performance summarized as follows:

#### BiPo-214 Chain
- **Stopped at Epoch**: 74
- **Final Training Loss**: 0.4722
- **Final Validation Loss**: 0.4728
- **Final Training Accuracy**: 79.53%
- **Final Validation Accuracy**: 79.50%

| Metric               | Score    |
|----------------------|----------|
| Precision (Macro)    | 0.813    |
| Recall (Macro)       | 0.795    |
| F1 Score (Macro)     | 0.795    |
| Precision (Weighted) | 0.813    |
| Recall (Weighted)    | 0.795    |
| F1 Score (Weighted)  | 0.795    |

**Confusion Matrix**:

|              | Predicted 0 | Predicted 1 | Predicted 2 | Predicted 3 |
|--------------|-------------|-------------|-------------|-------------|
| **Actual 0** | 285,736     | 98,272      | 4           | 47,327      |
| **Actual 1** | 21,021      | 276,288     | 10          | 135,394     |
| **Actual 2** | 0           | 7           | 433,464     | 4           |
| **Actual 3** | 34          | 53,553      | 15          | 378,655     |


#### BiPo-212 Chain
- **Stopped at Epoch**: 75
- **Final Training Loss**: 0.4341
- **Final Validation Loss**: 0.4337
- **Final Training Accuracy**: 81.61%
- **Final Validation Accuracy**: 81.62%

| Metric               | Score    |
|----------------------|----------|
| Precision (Macro)    | 0.834    |
| Recall (Macro)       | 0.816    |
| F1 Score (Macro)     | 0.818    |
| Precision (Weighted) | 0.834    |
| Recall (Weighted)    | 0.816    |
| F1 Score (Weighted)  | 0.818    |

**Confusion Matrix**:

|              | Predicted 0 | Predicted 1 | Predicted 2 | Predicted 3 |
|--------------|-------------|-------------|-------------|-------------|
| **Actual 0** | 312,592     | 47,644      | 71,829      | 6           |
| **Actual 1** | 3,321       | 370,720     | 57,003      | 15          |
| **Actual 2** | 7,247       | 128,636     | 288,913     | 13          |
| **Actual 3** | 0           | 0           | 0           | 422,891     |


## Insights and Analysis

1. **Performance Evaluation**: The models achieved overall accuracy scores of approximately 79.5% and 81.6% on the validation set for the BiPo-214 and BiPo-212 chains, respectively. Precision, recall, and F1 scores are well-aligned, reflecting balanced performance across classes.

2. **Confusion Matrix Analysis**:
   - The BiPo-214 model (Confusion Matrix 1) shows misclassifications among certain classes, with a significant number of events misclassified between adjacent decay categories.
   - The BiPo-212 model (Confusion Matrix 2) has a similar pattern, but with a higher precision and fewer misclassifications in certain decay categories, reflecting more accurate classification in the categories with higher representation.

3. **Learning Dynamics**: Training was set to run for a maximum of 200 epochs; however, early stopping prevented overfitting by halting at epochs 74 and 75 for the BiPo-214 and BiPo-212 models, respectively. This strategy helped stabilize validation performance, with no significant degradation due to overfitting.

4. **Limitations and Future Work**: While these ANN models achieved strong performance, further exploration with deeper neural networks, alternative architectures (e.g., CNNs or RNNs), or advanced regularization methods might yield even better accuracy and resilience against noise.

## Usage

1. **Training the Model**: Run the training code in this directory to retrain the ANN model on preprocessed input data.
2. **Evaluation**: Model outputs include performance metrics, such as precision, recall, and F1 scores, along with a confusion matrix for visualizing classification performance.
3. **Experimentation**: Users can adjust hyperparameters, such as batch size, learning rate, or network depth, to explore model behavior under different configurations.

This ANN model represents a promising approach to detecting low U/Th concentrations and improving event classification, laying the groundwork for potential extensions with more sophisticated machine learning techniques.

