# BiPo Event Classification in OSIRIS Using Machine Learning (Realistic Simulation)

## Project Overview

This branch contains a refined version of the BiPo classification pipeline originally developed for my master's thesis. It uses a more realistic class distribution, approximating actual signal-to-background ratios expected in OSIRIS, the pre-detector of the JUNO experiment.

In contrast to the thesis version, which used an artificially balanced dataset (equal class distribution), this version reflects the true imbalance:

| Class Label        | Count     |
|--------------------|-----------|
| External Background (3.0) | 6,300,000 |
| Internal Background (0.0) | 433,000   |
| <sup>214</sup>Bi (1.0)     | 2,707     |
| <sup>214</sup>Po (2.0)     | 2,707     |
| <sup>212</sup>Bi (4.0)     | 902       |
| <sup>212</sup>Po (5.0)     | 902       |

This version uses only the **BiLSTM + Attention** architecture with a **focal loss** (`CategoricalFocalCrossentropy(gamma=2.0)`) to address extreme class imbalance. Focal loss dynamically down-weights easy examples and focuses the training on hard, misclassified samples, making it especially effective for datasets where the signal classes are rare compared to dominant background events.

---

## Repository Structure

```text
BiPoClassification/
├── BiLSTM_Attention/             # Best model with post-processing
│   ├── Bi_lstm_whole_training_focal.ipynb
│   ├── Bi_lstm_whole_pred_focal.ipynb
├── data_preprocessing/           # Dataset and preprocessing scripts
│   ├── osiris_toydata_6.csv
│   ├── read_file.py
│   ├── feature_engineering_scaling.py
│   └── RNN_preprocessing_whole.py
├── data_visualisation/           # Data visualization script
│   └── data_viz.py
├── plots/                        # All plots used in analysis
│   └── [subdirectories per model and task]
└── model/                        # Trained model files (.h5, .pkl)
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
git clone --branch realistic-sim https://github.com/anuragxorma/BiPoClassification.git
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

Open the Jupyter notebooks in `BiLSTM_Attention/`:

- `Bi_lstm_whole_training_focal.ipynb`
- `Bi_lstm_whole_pred_focal.ipynb`

## Performance Summary

This project compares:

- Classical cuts  
- Optimized cuts via Optuna  
- Multiple ML models (ANN, RNNs, Decision Trees)  
- **BiLSTM with Attention** (best-performing model)

---

## Results

The model was evaluated on a highly imbalanced dataset using **focal loss** to focus on rare signal classes. The overall classification performance:

| Metric      | Score  |
|-------------|--------|
| Precision   | 0.94   |
| Recall      | 0.84   |
| F1-Score    | 0.89   |
| Accuracy    | 0.98   |

> **Conclusion:**  
> Even with extreme class imbalance, the BiLSTM + Attention model achieves high accuracy and strong precision/recall across classes, validating its suitability for realistic OSIRIS event classification.

---

## Limitations

- Toy simulation with class ratios derived from realistic assumptions, not true detector output
- Evaluation performed on synthetic data without realistic detector noise or systematic uncertainties

---

## Future Work

- Retrain and validate the model on real or full-scale OSIRIS simulation data
- Introduce uncertainty estimation (e.g., via Bayesian neural networks)
- Explore explainability methods (e.g., attention maps, SHAP) for physics insight
- Integrate trained model into a real-time OSIRIS data processing pipeline

---

## Summary

- This branch extends the original thesis work by incorporating a realistic signal-to-background ratio
- Maintains the same architecture (BiLSTM + Attention) but uses a focal loss function to address class imbalance
- Demonstrates that high classification accuracy and strong precision/recall can still be achieved under challenging conditions
- Intended as a more deployment-ready version of the BiPo classification pipeline, aligned with real OSIRIS use cases


---

## Thesis

The full thesis describing this work will be made available here once submitted.

---

## Contact

anuragsarma2001@gmail.com
