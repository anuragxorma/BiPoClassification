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

## Data Sources
- **Toy Monte Carlo Simulation of OSIRIS Detector Data**: Bi–Po decay event data from LS detectors.
- **Features**: Spatial coordinates, energy levels, decay time intervals, and calculated distances between detected events.
- **Labels**: True Bi–Po events (Bi-214, Po-214, Bi-212, Po-212), Background (both internal and external)

## Methods
- **Event Selection Cuts**: Custom algorithms to filter out background noise.
- **Machine Learning Models**: Classification models (ANN, RNN, LSTM, Decision Trees) trained to distinguish between Bi–Po coincidences and accidental events.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/anuragxorma/BiPoClassification.git
