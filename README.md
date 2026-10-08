# Machine Learning for Passive Circuit Modeling

A research project exploring the application of machine learning to predict the frequency responses of passive electrical circuits.

The goal is to develop surrogate models that approximate amplitude and phase responses based on circuit parameters and frequency, reducing the need for repeated numerical simulations.

## Project Overview

The project investigates seven circuit topologies:

- **RC filters:** low-pass (`RC_LP`) and high-pass (`RC_HP`)
- **RL filters:** low-pass (`RL_LP`) and high-pass (`RL_HP`)
- **RLC filters:** band-pass (`RLC_BP`) and notch (`RLC_NOTCH`)
- **RC ladder circuits:** variable-length structures with 1–4 sections (`RC_LADDER`)

Training datasets were generated using a custom Python implementation of Modified Nodal Analysis (MNA), including 2,400 fixed-topology configurations and 1,600 RC ladder configurations.

## Methodology

### Gradient Boosting

A custom Gradient Boosting Decision Tree (GBDT) algorithm was implemented for fixed-topology circuits.

- Physics-informed feature engineering based on cutoff frequencies, resonance parameters, and circuit characteristics.
- Multi-output regression for logarithmic transfer function magnitude and phase components (`sin φ`, `cos φ`).
- Group-based data splitting to prevent data leakage.
- Comparison against `HistGradientBoostingRegressor` from scikit-learn.

### Neural Network

A custom neural network was developed for RC ladder circuits with variable numbers of sections.

- Sequential and global feature representations.
- Section encoders and residual hidden-state updates.
- Separate prediction heads for amplitude and phase.
- Evaluation of generalization to unseen circuit lengths.

Two experimental scenarios were investigated:

- **Strict extrapolation:** training on 1–3 sections and testing on 4-section circuits.
- **Few-shot learning:** adding 4% of 4-section configurations to the training set.

## Results

### Fixed-Topology Circuits

The custom GBDT models outperformed the scikit-learn baseline across all six topologies.

| Circuit | GBDT MAE (dB) | Baseline MAE (dB) |
|---|---:|---:|
| RC_LP | 0.0057 | 0.0912 |
| RC_HP | 0.0033 | 0.0669 |
| RL_LP | 0.0038 | 0.0733 |
| RL_HP | 0.0083 | 0.0526 |
| RLC_BP | 0.0060 | 0.0746 |
| RLC_NOTCH | 0.0881 | 0.1741 |

### RC Ladder Circuits

| Metric | Strict Extrapolation | Few-Shot (4%) |
|---|---:|---:|
| Amplitude MAE (dB) | 1.133 | 0.859 |
| Phase MAE (°) | 27.063 | 4.895 |
| Filtered Phase MAE (°) | 3.156 | 2.124 |

Filtered phase MAE is evaluated at points where the transfer function magnitude exceeds 0.05. Results represent the selected best runs.

Adding only 4% of previously unseen 4-section configurations significantly improved prediction accuracy, particularly for phase response estimation.

## Technologies

**Python, NumPy, pandas, scikit-learn, PyTorch, Matplotlib**

## Conclusion

The results demonstrate the potential of machine learning for surrogate modeling of passive electrical circuits. Physics-informed gradient boosting provides accurate predictions for fixed-topology filters, while a structure-aware neural network enables modeling of RC ladder circuits with different numbers of sections.
