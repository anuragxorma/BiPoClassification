# BiPo Event Classification in OSIRIS Using Machine Learning

## Project Overview

OSIRIS is the pre-detector of the JUNO experiment, designed to measure radiopurity in the liquid scintillator (LS) through the identification of fast Bi–Po coincidence decays. These decays, originating from the <sup>214</sup>Bi–Po and <sup>212</sup>Bi–Po chains, are critical for estimating uranium and thorium content and suppressing backgrounds in neutrino detection.

The thesis work was carried out using a toy Monte Carlo simulation with an artificially balanced dataset, where all six classes (internal background, external background, and Bi/Po decays) were equally represented, each making up roughly 16% of the data. This setup was useful for training and model comparison, but does not reflect the realistic signal-to-background ratio observed in actual simulations.

> A more realistic simulation with proper background-to-signal ratios has been developed after thesis submission and is available on the [`realistic-sim`](https://github.com/yourusername/BiPoClassification/tree/realistic-sim) branch. This version is intended for future integration into official analysis pipelines.

This project demonstrates applied skills in:
- Data preprocessing and engineering (Pandas, Dask)
- Deep learning with TensorFlow/Keras (RNNs, BiLSTM, attention)
- Classical ML with Scikit-learn (Decision Trees, Grid Search)
- Model evaluation and optimization (Optuna, confusion matrices, post-processing)
- Scientific writing and research-level documentation

---

## Problem Statement

In OSIRIS, Bi–Po coincidences are traditionally selected using a sequence of strict cuts:

- **Fiducial Volume Cut**
- **Energy Selection**
- **Timing Cut**
- **Distance Cut**

While these cuts can reduce background contamination, they lack flexibility and cannot easily model complex, overlapping feature spaces.

This project investigates whether machine learning models can improve classification accuracy and event retention without compromising purity.

---

## Repository Structure

The repository is organized by task:

  ```text
  BiPoClassification/
  ├── ANN/                          # Training & prediction notebooks for ANN models
  │   ├── ANN_whole_training.ipynb
  │   ├── ANN_whole_pred.ipynb
  │   ├── ANN_sep_chain_training.ipynb
  │   └── ANN_sep_chain_pred.ipynb
  ├── BiLSTM_Attention/             # Best model with post-processing
  │   ├── Bi_lstm_whole_training.ipynb
  │   ├── Bi_lstm_whole_pred.ipynb
  │   ├── Bi_lstm_sep_chain_training.ipynb
  │   └── Bi_lstm_sep_chain_pred.ipynb
  ├── RNN/                          # RNN and LSTM models
  │   ├── SimpleRNN_whole_training.ipynb
  │   ├── SimpleRNN_whole_pred.ipynb
  │   ├── SimpleRNN_sep_chain_training.ipynb
  │   ├── SimpleRNN_sep_chain_pred.ipynb
  │   ├── LSTM_whole_training.ipynb
  │   ├── LSTM_whole_pred.ipynb
  │   ├── LSTM_sep_chain_training.ipynb
  │   └── LSTM_sep_chain_pred.ipynb
  ├── decision_tree/                # Decision Tree models
  │   ├── decision_tree_whole_training.ipynb
  │   ├── decision_tree_whole_pred.ipynb
  │   ├── decision_tree_sep_chain_training.ipynb
  │   └── decision_tree_sep_chain_pred.ipynb
  ├── classical_cuts/               # Classical and optimized selection cuts
  │   ├── cuts_classical_and_optimized.ipynb
  │   ├── optuna_optimization.py
  │   ├── best_parameters.txt
  │   ├── opt_hist_214.png
  │   └── opt_hist_212.png
  ├── data_preprocessing/           # Dataset and preprocessing scripts
  │   ├── osiris_toydata_7.csv
  │   ├── read_file.py
  │   ├── feature_engineering_scaling.py
  │   ├── RNN_preprocessing_whole.py
  │   └── RNN_preprocessing_sep_chain.py
  ├── data_visualisation/           # Data visualization script
  │   └── data_viz.py
  ├── plots/                        # All plots used in analysis
  │   └── [subdirectories per model and task]
  └── models/                       # All trained models (.h5, .pkl)
```

---

## Dataset

- **Source**: Toy Monte Carlo simulation of OSIRIS detector data.
- **Classes**: Internal Background (Label 0), <sup>214</sup>Bi (Label 1), <sup>214</sup>Po (Label 2), External Background (Label 3), <sup>212</sup>Bi (Label 4), <sup>212</sup>Po (Label 5)
- **Features**: Event Time in seconds, photoevents (to be converted to energy by dividing by 280), Spatial coordinates(x, y, z)
- **Preprocessing**:
  - Feature engineering of cylindrical coordinates
  - MinMax/Robust scaling
  - Sequence generation for RNNs

---

## Installation

Install required dependencies using:

```bash
pip install pandas numpy scikit-learn tensorflow dask matplotlib seaborn joblib
```

## Usage

### Clone the Repository

```bash
git clone https://github.com/anuragxorma/BiPoClassification.git
cd BiPoClassification
```

### Prepare the Data

Run the following scripts in the `data_preprocessing/` directory:

```bash
python read_file.py
python feature_engineering_scaling.py
python RNN_preprocessing_whole.py   # or RNN_preprocessing_sep_chain.py
```

### Train and Evaluate Models

To train or evaluate the models, open the corresponding Jupyter notebooks located in the appropriate folders:

#### For ANN models:

- `ANN/ANN_whole_training.ipynb`
- `ANN/ANN_whole_pred.ipynb`
- `ANN/ANN_sep_chain_training.ipynb`
- `ANN/ANN_sep_chain_pred.ipynb`

#### For RNN and LSTM models:

- `RNN/SimpleRNN_whole_training.ipynb`
- `RNN/SimpleRNN_whole_pred.ipynb`
- `RNN/SimpleRNN_sep_chain_training.ipynb`
- `RNN/SimpleRNN_sep_chain_pred.ipynb`
- `RNN/LSTM_whole_training.ipynb`
- `RNN/LSTM_whole_pred.ipynb`
- `RNN/LSTM_sep_chain_training.ipynb`
- `RNN/LSTM_sep_chain_pred.ipynb`

#### For BiLSTM + Attention (best-performing model):

- `BiLSTM_Attention/Bi_lstm_whole_training.ipynb`
- `BiLSTM_Attention/Bi_lstm_whole_pred.ipynb`
- `BiLSTM_Attention/Bi_lstm_sep_chain_training.ipynb`
- `BiLSTM_Attention/Bi_lstm_sep_chain_pred.ipynb`

#### For Decision Tree models:

- `decision_tree/decision_tree_whole_training.ipynb`
- `decision_tree/decision_tree_whole_pred.ipynb`
- `decision_tree/decision_tree_sep_chain_training.ipynb`
- `decision_tree/decision_tree_sep_chain_pred.ipynb`

## Performance Summary

This project compares:

- Classical cuts  
- Optimized cuts via Optuna  
- Multiple ML models (ANN, RNNs, Decision Trees)  
- **BiLSTM with Attention** (best-performing model)

---

### Results

| Method                        | <sup>214</sup>BiPo TPR | Efficiency | <sup>212</sup>BiPo TPR | Efficiency |
|------------------------------|------------------|------------|------------------|------------|
| Classical Cuts               | 68.66%           | 47.34%     | 99.99%           | 14.54%     |
| Optimized Cuts               | 61.95%           | 46.81%     | 91.86%           | 51.52%     |
| BiLSTM + Attention (Pairs)   | 84.53%           | 93.94%     | 82.08%           | 94.45%     |

> **Conclusion:**  
> BiLSTM + Attention with post-processing significantly outperforms cut-based approaches in both decay chains and is recommended for practical application.

---

## Limitations

- Toy simulation with equal class balance (not representative of real OSIRIS data)  
- No training on real detector data yet  
- No uncertainty estimation (future work: Bayesian networks)  
- Short sequences used due to computational constraints  

---

## Future Work

- Retrain on full OSIRIS simulation once available  
- Incorporate class imbalance techniques  
- Extend to transformer models and attention-rich architectures  
- Add uncertainty quantification and explainability methods  
- Real-time inference integration  

---

## Summary

- Developed machine learning models (ANN, RNNs, BiLSTM+Attention) to classify Bi–Po decays in OSIRIS, a pre-detector of the JUNO experiment.
- Achieved >90% efficiency using BiLSTM+Attention, significantly outperforming traditional selection cuts.
- Implemented feature engineering, sequence preprocessing, Optuna-based hyperparameter optimization, and comprehensive model evaluation.
- Demonstrated potential for real-time event classification in neutrino detection pipelines

---

## Thesis

The full thesis describing this work will be made available here once submitted.

---

## Contact

anuragsarma2001@gmail.com
