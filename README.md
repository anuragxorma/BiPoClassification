# BiPo Event Classification in OSIRIS Using Machine Learning

## Project Overview
OSIRIS aims to detect ultra-low levels of uranium (U) and thorium (Th) decay within a liquid scintillator (LS) detector by tagging fast Bi–Po coincidence decays. These coincidence events provide a direct method to determine U/Th abundances through the detection of 214Bi–Po and 212Bi–Po decays, critical for assessing detector purity and background levels.

## Problem Statement
In a low-background environment, the fast Bi–Po decays are isolated from accidental background events through four stringent cuts:
- **Fiducial Volume (FV)**: Spatial filtering to reduce external event interference.
- **Energy Selection**: Isolation of events within specific energy ranges unique to Bi and Po decay.
- **Timing Cut**: Utilizing short lifetimes of polonium isotopes to distinguish true coincidences.
- **Distance Cut**: Filtering based on spatial proximity between Bi and Po candidates to minimize background noise.

The challenge is to implement machine learning models capable of accurately identifying true Bi–Po events and quantifying U/Th mass limits based on detected decay rates.

## Project Structure
The repository is organized into folders that contain files for various stages of data processing, feature engineering, model training, and evaluation.

### Main Branch Structure
- **`Shared_dir`**
  - `read_file.py`: Reads the CSV data file and processes it for model training.
  - `feature_engineering_scaling.py`: Performs feature engineering and scaling for input data used in model training.

- **Model-Specific Folders**
  Each model has its own dedicated folder, containing Jupyter notebooks and saved models:
  
  - **`ANN` Folder**:
    - **`train_model.ipynb`**: Jupyter notebook for training the Artificial Neural Network (ANN) model.
    - **Saved Models**: The trained models are saved as `.h5` files.
    - **`evaluate_model.ipynb`** (Planned): A notebook to apply the trained model to unseen data and evaluate performance.
  
  - **`RNN` Folder**:
    - **`SimpleRNN_model.ipynb`**: Jupyter notebook for training a SimpleRNN model.
    - **`LSTM_model.ipynb`**: Jupyter notebook for training an LSTM model.
    - **Saved Models**: Both models are saved as `.h5` files.
  
  - **`Decision_Tree` Folder**:
    - **`train_decision_tree.ipynb`**: Jupyter notebook for training a Decision Tree model with grid search for optimization.
    - **Alternative Models** (Planned): Potential inclusion of additional models such as logistic regression.


## Data Sources
- **Toy Monte Carlo Simulation of OSIRIS Detector Data**: Bi–Po decay event data from LS detectors.
- **Features**: Spatial coordinates, energy levels, decay time intervals, and calculated distances between detected events.
- **Labels**: True Bi–Po events (Bi-214, Po-214, Bi-212, Po-212), Background (both internal and external)

## Data Structure

The dataset used for this project contains events from both the BiPo-214 and BiPo-212 decay chains. Due to the nature of the data, the chains were separated for the training of the models. This separation results in some events being classified as belonging to both chains. As a result, statistical methods will need to be developed to distinguish these overlapping classifications.

One of the plans for the near future is to identify events that are classified as both BiPo-214 and BiPo-212, and devise a method to properly distinguish between them.

For more detailed information about the structure of the dataset, including the specific features, labels, and preprocessing steps, please refer to the README file located in the `Shared_dir` directory of the repository.

## Methods

### Data Preparation
- **Data Reading**: `read_file.py` (located in `Shared_dir`) reads and loads the CSV data file for processing.
- **Feature Engineering and Scaling**: `feature_engineering_scaling.py` standardizes and scales the input data, optimizing it for model performance.

### Event Selection Cuts
Custom algorithms are applied to filter out background noise and isolate potential Bi–Po events based on:
- **Fiducial Volume Cut**: Reduces external interference by spatial filtering.
- **Energy Selection**: Restricts events to specific energy ranges characteristic of Bi and Po decays.
- **Timing Cut**: Distinguishes true Bi–Po coincidences based on the short polonium lifetimes.
- **Distance Cut**: Minimizes background noise by limiting the spatial proximity between Bi and Po candidates.

### Model Training
- **Model Types**: Each model (ANN, RNN, LSTM, and Decision Tree) is trained separately within dedicated folders:
  - **ANN and RNN Models**: Models are trained using neural network architectures, with trained models saved as `.h5` files for easy loading and evaluation.
  - **Decision Tree Model**: A Decision Tree is implemented with Grid Search for hyperparameter optimization.
  - **Potential Additions**: Non-neural models like Logistic Regression or SVM may be added in the future to compare performance and robustness against neural models.

### Evaluation
- **Evaluation Notebooks**: Separate Jupyter notebooks within each model folder provide functionality for applying the models to unseen data and assessing their accuracy.
- **Sensitivity Curves**: Sensitivity curves are generated to evaluate the model’s performance over varying conditions and gauge its ability to distinguish true Bi–Po coincidences from background noise.

### Machine Learning Models
Classification models (ANN, RNN, LSTM, Decision Trees) are trained to differentiate between Bi–Po coincidence events and accidental background events.

## Installation Requirements
This project requires the following libraries:
- Pandas
- Numpy
- Scikit-Learn
- TensorFlow/Keras (for neural networks)
- Dask (for data processing)
- Matplotlib
- Seaborn
You can install all dependencies by running:

  ```bash
  pip install pandas numpy scikit-learn tensorflow dask matplotlib seaborn

## Usage
1. **Clone the repository:**
   ```bash
   git clone https://github.com/anuragxorma/BiPoClassification.git

2. **Run Data Preparation:**
  Execute `read_file.py` and `feature_engineering_scaling.py` located in the `Shared_dir` folder to load and preprocess the data.

3. **Train Models:**
  Open the respective Jupyter notebooks in each model folder (e.g., `ANN`, `RNN`, `Decision_Tree`) and execute the cells for training.

4. **Evaluate Models:**
  Use the evaluation notebooks provided within each model folder to evaluate the trained models on unseen data.


## Overview of Results

The classification models developed for Bi–Po event detection in OSIRIS achieved strong performance in distinguishing true Bi–Po coincidences from background noise. Below is a summary of key performance metrics for each model:

### BiPo-214 Chain Performance:

| Model         | Accuracy | F1 Score (Weighted) | Precision (Weighted) | Recall (Weighted) | Sensitivity to Background |
|---------------|----------|---------------------|----------------------|-------------------|---------------------------|
| **ANN**       | 79.5%      | 0.795                | 0.8132                 | 0.7948              | Moderate                      |
| **SimpleRNN** | 79.54%      | 0.7948                | 0.8129                 | 0.7947              | Moderate                  |
| **LSTM**      | 79.54%      | 0.79.48                | 0.8129                 | 0.7947              | Moderate                 |
| **Decision Tree** | 88%  | 0.87                | 0.88                 | 0.86              | Moderate                  |

### BiPo-212 Chain Performance:

| Model           | Accuracy | F1 Score (Weighted) | Precision (Weighted) | Recall (Weighted) | Sensitivity to Background |
|-----------------|----------|---------------------|----------------------|-------------------|---------------------------|
| **ANN**         | 81.62%      | 0.8175                | 0.8339                 | 0.8154              | Moderate                      |
| **SimpleRNN**   | 81.62%      | 0.8174                | 0.8339                 | 0.8153              | Moderate                  |
| **LSTM**        | 81.62%      | 0.8174                | 0.8339                 | 0.8153              | Moderate                 |
| **Decision Tree** | 72%    | 0.80                | 0.82                 | 0.78              | Moderate                  |


**Interpretation of Results**:
- **ANN and RNN models** achieved similar performance across both the BiPo-214 and BiPo-212 chains, with accuracy around 79.5% and 81.6%, respectively. These models exhibited balanced performance with moderate sensitivity to background noise. They are well-suited for general event classification tasks.
- **LSTM models** showed identical performance to the SimpleRNN in terms of accuracy, but its temporal pattern recognition capabilities give it an edge, particularly in tasks requiring sequential learning, though the sensitivity to background noise was similarly moderate.
- **Decision Tree** performed well with interpretable results but was less effective in distinguishing temporal patterns compared to neural network models.
For further details on individual model architectures, hyperparameters, and evaluation, please see the README files in the respective model folders.

## Project Limitations
While the model effectively distinguishes Bi–Po events, certain rare decay events might introduce noise, limiting model sensitivity under extremely low-background conditions.

## Future Work
Future plans include:
- Exploring additional non-neural models (e.g., SVM) for comparison.
- Enhancing feature engineering methods for better background noise suppression.
- Increasing interpretability through SHAP values or feature importance plots.

## Contributing
Contributions are welcome! Please submit an issue or a pull request to discuss proposed changes.

## Contact
For questions or feedback, please reach out to anuragsarma2001@gmail.com or open an issue on GitHub.

Thank you for exploring the **BiPo Event Classification in OSIRIS** project!

