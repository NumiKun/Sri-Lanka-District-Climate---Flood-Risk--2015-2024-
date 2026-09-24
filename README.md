# Sri Lanka District Climate & Flood Risk (2015–2024)
## Multi-Day Ahead Flood Risk Early Warning System Using Deep Recurrent Neural Networks

An end-to-end machine learning and deep learning project that models daily hydrometeorological dynamics across all 25 administrative districts of Sri Lanka and generates operational, multi-day ahead (T+1 to T+7) flood risk forecasts using a Bidirectional LSTM (Long Short-Term Memory) architecture implemented in PyTorch.

---

## Executive Summary

Flooding represents one of the most severe climate-induced natural hazards in Sri Lanka, driven primarily by the Southwest (Yala) and Northeast (Maha) monsoon seasons. Effective disaster risk reduction (DRR) and emergency civil defense require anticipatory early warning systems rather than retrospective hazard assessments.

This repository implements a production-grade machine learning pipeline that transforms 10 consecutive years (2015–2024) of district-level daily climate observations into a multi-step sequence-to-vector forecasting engine. Given a 30-day historical window of hydrometeorological indicators, the model forecasts continuous flood risk severity scores ($0$ to $100$) simultaneously across lead times of 1 to 7 days ahead for any district in the country.

---

## Key Highlights

- Comprehensive Spatial Coverage: Models all 25 administrative districts across the Wet, Intermediate, and Dry climatic zones of Sri Lanka.
- 10-Year Temporal Record: 91,325 daily observations spanning from January 1, 2015 through December 31, 2024.
- Multi-Day Lead Time Forecasting: Direct sequence-to-vector output generating simultaneous forecasts for T+1, T+2, T+3, T+4, T+5, T+6, and T+7 horizons.
- Physics-Informed Feature Engineering: Integrates 42 engineered features encompassing soil saturation dynamics, multi-window antecedent precipitation accumulations (24h, 48h, 72h), rolling momentum, volatility measures, and harmonic seasonal encodings.
- Leakage-Free Temporal Validation: Strict walk-forward temporal splitting ensuring no lookahead bias across training (2015–2021), validation (2022), and out-of-sample test (2023–2024) periods.
- Robust PyTorch Architecture: Bidirectional LSTM combined with stacked recurrent and dense layers, trained using Huber Loss to maintain robustness against heavy-tailed flood events.
- Complete Production Artifacts: Full serialization of model checkpoints, MinMaxScaler transformers, LabelEncoders, configuration metadata, training history logs, and evaluation figures.

---

## System Architecture

```text
+-------------------------------------------------------------------------------+
|                       1. DATA ACQUISITION & INGESTION                         |
|  - 91,325 records, 10 raw features (2015-01-01 to 2024-12-31)                 |
|  - Climate, precipitation, soil moisture (0-7cm, 7-28cm), geographic features |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                    2. DOMAIN & HYDROLOGICAL FEATURE ENGINEERING               |
|  - Short-term antecedent rainfall accumulations: 24h, 48h, 72h                |
|  - Dynamic soil saturation indices & trend differentials                      |
|  - Multi-scale temporal lags: 1, 3, 7, 14 days                                |
|  - Moving statistics: 7, 14, 30-day rolling means and standard deviations     |
|  - Cyclical harmonic temporal encodings: sin/cos of month and day-of-year     |
|  - District and climatic zone spatial embeddings / label encodings            |
|  -> Total feature space: 42 predictor variables                               |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                 3. TEMPORAL PARTITIONING & SLIDING WINDOW MATRIX              |
|  - Train Set (2015–2021) | Validation Set (2022) | Test Set (2023–2024)       |
|  - Strict fit-on-train-only normalization via MinMaxScaler                    |
|  - Lookback Window (L = 30 days) -> Forecast Horizon (H = 7 days)             |
|  - Tensor Input Shape: (N, 30, 42) -> Tensor Target Shape: (N, 7)             |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                     4. PYTORCH DEEP LEARNING MODEL (BiLSTM)                   |
|  - Input Layer (42 features)                                                  |
|  - Layer 1: Bidirectional LSTM (128 hidden units per direction = 256 output)  |
|  - Layer 2: Stacked Unidirectional LSTM (64 hidden units)                     |
|  - Dropout Regularization: rate = 0.25                                        |
|  - Dense Non-Linear Layer: 64 units + ReLU                                    |
|  - Multi-Horizon Head: Linear projection (7 outputs) + Sigmoid activation     |
|  - Loss: Huber Loss (delta = 1.0) | Optimizer: Adam (lr = 0.001, wd = 1e-4)   |
|  - Learning Rate Scheduler: ReduceLROnPlateau (factor = 0.5, patience = 5)    |
|  - Regularization: Early stopping (patience = 15) & Gradient clipping (1.0)  |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                   5. EVALUATION, THRESHOLDING & ARTIFACT PERSISTENCE          |
|  - Multi-metric evaluation: RMSE, MAE, R-squared per lead time (T+1 to T+7)   |
|  - Operational alert mapping: Low, Advisory, High Warning, Critical Emergency  |
|  - Artifact persistence: weights (.pt), scalers (.pkl), logs & visuals (.png) |
+-------------------------------------------------------------------------------+
```

---

## Dataset Description

The primary dataset is located in `Dataset/sri_lanka_flood_risk_modeled.csv`. It comprises daily meteorological, hydrological, and geographic records across all administrative districts of Sri Lanka over a 10-year span.

### Raw Variables

| Feature Name | Type | Unit / Range | Description |
|---|---|---|---|
| `date` | Date | 2015-01-01 to 2024-12-31 | Daily timestamp of observation |
| `district` | Categorical | 25 districts | Administrative district name |
| `latitude` | Float | 5.91 to 9.83 | Geographic latitude (degrees North) |
| `longitude` | Float | 79.69 to 81.88 | Geographic longitude (degrees East) |
| `province` | Categorical | 9 provinces | Administrative provincial boundary |
| `climatic_zone` | Categorical | Wet, Intermediate, Dry | Major agro-climatic ecological classification |
| `precipitation_sum` | Float | mm | Total daily precipitation |
| `rain_sum` | Float | mm | Total daily rainfall liquid equivalent |
| `soil_moisture_0_to_7cm_mean` | Float | m3/m3 | Volumetric soil water content in topsoil layer |
| `soil_moisture_7_to_28cm_mean`| Float | m3/m3 | Volumetric soil water content in root-zone layer |
| `temperature_2m_max` | Float | Celsius | Maximum daily air temperature at 2 meters |
| `wind_speed_10m_max` | Float | km/h | Maximum daily sustained wind speed at 10 meters |
| `flood_risk_score` | Float | 0.0 to 100.0 | Ground-truth continuous flood risk severity index |
| `flood_category` | Categorical | 4 ordinal levels | Discrete operational classification |

### Operational Risk Thresholds

The continuous target score is structured into standard disaster management operational tiers:

- Score 0.0 to 29.9: Low Risk (Normal) - Routine baseline hydrological conditions.
- Score 30.0 to 59.9: Advisory (Waterlogging) - Saturated soil, localized urban or low-lying water pooling.
- Score 60.0 to 79.9: High Warning (Inundation) - Significant riverine bank overflow, localized infrastructure impact.
- Score 80.0 to 100.0: Critical Emergency (Catastrophic Overflow) - Severe basin-wide inundation, immediate evacuation protocol.

---

## Feature Engineering

A total of 42 input features are supplied to the recurrent network at each timestep $t$. The feature set incorporates domain-specific hydrological principles:

1. Antecedent Precipitation Accumulation:
   - `rain_24h`, `rain_48h`, `rain_72h`: Cumulative liquid precipitation over recent hours to capture runoff generation capacity.
2. Soil Moisture and Saturation Dynamics:
   - `soil_saturation_index`: Weighted ratio combining topsoil and subsurface volumetric moisture content against water-holding capacity.
   - `soil_saturation_trend`: First-order temporal difference indicating rapid wetting or drying phases.
3. Multi-Horizon Lag Variables:
   - 1-day, 3-day, 7-day, and 14-day lags for `precipitation_sum`, `flood_risk_score`, and `soil_saturation_index`.
4. Rolling Statistical Aggregations:
   - 7-day, 14-day, and 30-day rolling arithmetic mean and standard deviation for precipitation, flood risk, and soil saturation to reflect baseline climatic regimes.
5. Precipitation Momentum:
   - 7-day relative momentum ratio measuring precipitation intensification relative to monthly baselines.
6. Harmonic Seasonality:
   - Sine and cosine transformations of month-of-year and day-of-year to capture bimodal monsoonal periodicity without artificial boundary discontinuities.
7. Spatial Indicators:
   - Normalized label encodings for district identities and climatic zone classifications.

---

## Model Architecture & Training Protocol

### Neural Network Specifications

The predictive engine utilizes a stacked recurrent neural network architecture designed in PyTorch:

| Component | Layer Type | Specifications / Dimensions |
|---|---|---|
| Input | Tensor | (Batch Size, 30 timesteps, 42 features) |
| Recurrent Stage 1 | Bidirectional LSTM | 128 hidden units per direction (256 concatenated output) |
| Recurrent Stage 2 | Unidirectional LSTM | 64 hidden units |
| Regularization | Dropout | Rate = 0.25 applied after recurrent stages |
| Dense Projection | Linear + ReLU | 64 hidden units with non-linear activation |
| Output Head | Linear + Sigmoid | 7 output units scaled to [0, 1] range |

Total parameters: 265,671 (all trainable).

### Optimization Parameters

- Objective Function: Huber Loss ($\delta = 1.0$), providing quadratic penalties for minor estimation errors and linear penalties for extreme outliers, preventing gradient instability during flash flood anomalies.
- Optimizer: Adam with initial learning rate $\eta = 10^{-3}$ and L2 weight decay $\lambda = 10^{-4}$.
- Learning Rate Schedule: `ReduceLROnPlateau` monitoring validation loss, reducing learning rate by a factor of 0.5 upon 5 consecutive stagnant epochs.
- Gradient Clipping: Norm clipped at 1.0 to prevent exploding gradients through long backpropagation sequences.
- Early Stopping: Monitored validation loss with a patience threshold of 15 epochs.
- Batch Size: 512 samples per minibatch.

---

## Evaluation Results

The model was evaluated on a completely unseen, out-of-sample test split covering the full calendar years 2023 and 2024. All continuous metrics are reported in original flood risk score units (0–100 scale).

### Out-of-Sample Test Set Performance (2023–2024)

| Forecast Horizon | Lead Time | RMSE | MAE | R-squared |
|---|---|---|---|---|
| T+1 | 1 Day Ahead | 9.893 | 5.374 | 0.295 |
| T+2 | 2 Days Ahead | 10.224 | 5.487 | 0.246 |
| T+3 | 3 Days Ahead | 10.408 | 5.647 | 0.218 |
| T+4 | 4 Days Ahead | 10.493 | 5.664 | 0.200 |
| T+5 | 5 Days Ahead | 10.578 | 5.775 | 0.186 |
| T+6 | 6 Days Ahead | 10.635 | 5.812 | 0.177 |
| T+7 | 7 Days Ahead | 10.692 | 5.836 | 0.170 |
| Overall | All 7 Horizons | 10.421 | 5.657 | 0.213 |

### Performance Insights

- Strongest Short-Term Fidelity: Lead time T+1 delivers the lowest error rates (RMSE 9.89, MAE 5.37) and highest explained variance (R-squared = 0.295).
- Graceful Horizon Degradation: As lead time extends to 7 days ahead, MAE degrades by only 0.46 points (from 5.37 to 5.84), showing high temporal stability across the multi-day forecast window.
- Low Absolute Error: Across all horizons, the mean absolute error remains below 6.0 points on a 100-point scale, providing accurate discrimination between routine and elevated hazard states.

---

## Repository Structure

```text
Sri Lanka District Climate & Flood Risk (2015–2024)/
|-- Dataset/
|   `-- sri_lanka_flood_risk_modeled.csv    # 10-year daily hydrometeorological dataset
|-- Model/
|   |-- flood_risk_lstm_forecasting.ipynb  # End-to-end PyTorch modeling notebook
|   `-- artifacts/
|       |-- README.md                      # Artifact documentation and reload guide
|       |-- best_model.pt                  # Checkpoint of lowest validation loss weights
|       |-- flood_risk_lstm_weights.pt     # Final trained model state dictionary
|       |-- model_config.json              # Hyperparameter and feature configuration
|       |-- model_knowledge.json           # Comprehensive pipeline metadata and metrics
|       |-- training_log.json              # Per-epoch training and validation loss log
|       |-- feature_scaler.pkl             # Fitted MinMaxScaler for input predictors
|       |-- target_scaler.pkl              # Fitted MinMaxScaler for flood risk targets
|       |-- label_encoder_district.pkl     # Fitted LabelEncoder for district identities
|       |-- label_encoder_zone.pkl         # Fitted LabelEncoder for climatic zones
|       |-- actual_vs_predicted_multistep.png # Multi-step forecast comparison plots
|       |-- class_distribution.png         # Operational flood class distribution plot
|       |-- correlation_matrix.png         # Hydrological feature correlation heatmap
|       |-- district_risk_profile.png      # Spatial mean risk by district visualization
|       |-- per_horizon_metrics.png        # Degradation curves for RMSE, MAE, and R2
|       |-- residual_analysis.png          # Residual distribution and Q-Q diagnostic plot
|       |-- target_distribution.png        # Distribution of target flood risk scores
|       |-- temporal_patterns.png          # Monthly and annual risk trend analysis
|       `-- training_history.png           # Loss curves across training epochs
|-- LICENSE                                # MIT Open Source License
|-- README.md                              # Repository overview and documentation
`-- requirements.txt                       # Python dependencies and version specifications
```

---

## Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/NumiKun/Sri-Lanka-District-Climate---Flood-Risk--2015-2024-.git
cd "Sri Lanka District Climate & Flood Risk (2015–2024)"
```

### 2. Configure Virtual Environment

Using Python 3.10 or higher:

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows PowerShell
.\venv\Scripts\Activate.ps1

# Activate on Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage Guide

### Executing the Modeling Pipeline

To reproduce data preparation, feature engineering, model training, evaluation, and artifact generation:

1. Launch Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook
   ```
2. Open `Model/flood_risk_lstm_forecasting.ipynb`.
3. Execute all cells sequentially (`Cell` -> `Run All`). All training figures, serialized models, and JSON metadata will be written automatically to `Model/artifacts/`.

### Programmatic Model Inference

To load the trained model and run multi-step inference in Python:

```python
import json
import pickle
import torch
import torch.nn as nn
import numpy as np

# 1. Define model architecture matching training specifications
class FloodRiskLSTM(nn.Module):
    def __init__(self, input_size=42, hidden1=128, hidden2=64, dense_hidden=64, forecast_h=7, dropout=0.25):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden1 * 2, hidden2, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden2, dense_hidden)
        self.relu = nn.ReLU()
        self.out = nn.Linear(dense_hidden, forecast_h)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.bilstm(x)
        out, _ = self.lstm2(out)
        last_step = self.dropout(out[:, -1, :])
        dense = self.relu(self.fc1(last_step))
        return self.sigmoid(self.out(dense))

# 2. Load model weights and scalers
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FloodRiskLSTM(input_size=42, hidden1=128, hidden2=64, dense_hidden=64, forecast_h=7).to(device)
model.load_state_dict(torch.load('Model/artifacts/best_model.pt', map_location=device))
model.eval()

with open('Model/artifacts/target_scaler.pkl', 'rb') as f:
    target_scaler = pickle.load(f)

# 3. Simulate input tensor: 1 district, 30 days history, 42 normalized features
sample_input = torch.randn(1, 30, 42).to(device)

# 4. Generate multi-day ahead forecasts (T+1 to T+7)
with torch.no_grad():
    normalized_pred = model(sample_input).cpu().numpy()
    risk_scores = target_scaler.inverse_transform(normalized_pred)[0]

for day_ahead, score in enumerate(risk_scores, start=1):
    alert = (
        'Critical Emergency' if score >= 80.0 else
        'High Warning' if score >= 60.0 else
        'Advisory' if score >= 30.0 else
        'Low Risk'
    )
    print(f'Horizon T+{day_ahead}: Predicted Score = {score:5.1f} | Category = {alert}')
```

---

## License

This project is open-source and licensed under the terms of the [MIT License](LICENSE).

## Author

- NumiKun (Rizki Nugroho)
- GitHub: [@NumiKun](https://github.com/NumiKun)
- Email: nugrohorizki20@gmail.com
