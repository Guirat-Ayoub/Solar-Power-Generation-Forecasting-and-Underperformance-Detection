[README (3).md](https://github.com/user-attachments/files/29286530/README.3.md)
# Solar Power Generation Forecasting and Underperformance Detection

A machine learning system for forecasting photovoltaic power output from
weather conditions, using physics-informed feature engineering to capture
known photovoltaic behavior including temperature-induced efficiency
derating and DC-to-AC inverter conversion.

---

## Problem Statement

Solar farm operators need to know how much power a plant should be producing
given current weather conditions, in order to detect underperformance caused
by panel soiling, shading, or inverter faults. This project builds the
forecasting foundation required for that diagnostic: predicting expected
power output (both DC and AC) from irradiation and temperature readings.

---

## Dataset

- **Source:** Solar Power Generation Data — Kaggle
- **Link:** https://www.kaggle.com/datasets/anikannal/solar-power-generation-data
- **Plant:** Plant 1, 34 days of operation, 15-minute intervals
- **Generation data:** DC power, AC power, daily yield, total yield per inverter
- **Weather data:** Ambient temperature, module temperature, solar irradiation
- **Merged dataset size:** 68,774 records

The generation file contains one row per inverter (`SOURCE_KEY`) per
timestamp, while the weather file contains one row per timestamp
(plant-wide sensor). The two were merged on `DATE_TIME` and `PLANT_ID`,
correctly repeating each weather reading across all inverters active at
that timestamp.

---

## Feature Engineering

All engineered features are grounded in photovoltaic physics rather than
arbitrary transformations:

| Feature | Formula | Physical Meaning |
|---|---|---|
| `hour_sin`, `hour_cos` | Cyclical encoding of hour | Prevents the model from treating hour 23 and hour 0 as distant when they are adjacent in time |
| `temp_differential` | Module temp - Ambient temp | Panel self-heating from absorbed solar energy |
| `temp_derating` | 1 - 0.004 x (Module temp - 25) | Efficiency loss above 25°C — photovoltaic cells lose roughly 0.4% efficiency per °C above standard test conditions |
| `inverter_efficiency` | AC power / DC power | DC-to-AC conversion ratio — the core signal for inverter health diagnostics |
| `power_per_irradiation` | AC power / Irradiation | Output efficiency normalised by available sunlight |
| `irradiation_temp` | Irradiation x Module temp | Combined thermal-radiative stress on the panel |

---

## Modeling Approach

Two separate regression targets were modeled using identical weather-based
features, deliberately excluding any power measurement from the input set
to avoid circular prediction:

**Features used for both models:**
`IRRADIATION`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `hour_sin`,
`hour_cos`, `temp_differential`, `temp_derating`

**Evaluation:** 80/20 chronological train-test split (no shuffling, to
respect time order) plus 5-fold `TimeSeriesSplit` cross-validation to
confirm stability across different time windows.

---

## Results: AC Power Prediction

| Model | MAE (kW) | R² |
|---|---|---|
| Logistic Regression | — | — |
| Random Forest | 17.57 | 0.9786 |
| **Gradient Boosting** | **16.84** | **0.9801** |
| XGBoost | 16.92 | 0.9789 |

**Time series cross-validation MAE:** 21.48 ± 2.58 kW

Gradient Boosting achieved the best performance and was selected as the
final AC power model.

---

## Results: DC Power Prediction

| Model | MAE (kW) | R² |
|---|---|---|
| Random Forest | 180.59 | 0.9784 |
| **Gradient Boosting** | **171.62** | **0.9801** |
| XGBoost | 171.96 | 0.9793 |

**Time series cross-validation MAE:** 219.36 ± 27.08 kW

Note: DC power values are on a much larger scale than AC power in this
dataset (raw panel-side output before inverter conversion), which explains
the larger absolute MAE despite an identical R² to the AC model. Both
models explain approximately 98% of the variance in their respective
targets.

---

## Why Two Separate Power Models

Predicting DC power and AC power independently — both from weather alone —
enables fault localisation. If actual DC power matches the weather-based
prediction but actual AC power falls short of its prediction, the
discrepancy points to the inverter (DC-to-AC conversion stage). If DC power
itself underperforms relative to weather, the issue is upstream — at the
panel level (soiling, shading, or degradation). This separation is the
foundation for the underperformance detection module.

---

## Technologies Used

- Python 3.12
- pandas, NumPy
- scikit-learn (RandomForestRegressor, GradientBoostingRegressor, TimeSeriesSplit)
- XGBoost (XGBRegressor)
- Matplotlib, Seaborn

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/Guirat-Ayoub/solar-power-forecasting.git
cd solar-power-forecasting
pip install -r requirements.txt
```

Open the notebook in Google Colab:

https://colab.research.google.com/github/Guirat-Ayoub/solar-power-forecasting/blob/main/Solar_Power_Generation_Forecasting_and_Underperformance_Detection.ipynb

The dataset is available at:
https://www.kaggle.com/datasets/anikannal/solar-power-generation-data

---

## Key Findings

- Solar irradiation is the dominant predictor of power output, consistent
  with the near-linear relationship expected from photovoltaic theory
- Gradient Boosting consistently outperformed Random Forest and XGBoost on
  both targets, achieving R² of 0.98 with weather features alone
- The model's stable cross-validation performance (MAE std of 2.58 kW for
  AC power) across different time windows indicates the forecasting
  relationship generalises well across the 34-day observation period
- Module temperature consistently exceeds ambient temperature, confirming
  expected panel self-heating from absorbed solar radiation

---

## Limitations and Future Work

- Only Plant 1 was used for this phase; Plant 2 data is available for
  cross-plant validation
- The underperformance detection module (comparing actual vs predicted
  power per inverter) is the next development step, using the
  `inverter_efficiency` feature already engineered
- A Streamlit dashboard for real-time monitoring is planned
- Weather forecast data (rather than observed weather) would be required
  to make this a genuine forecasting system rather than a same-time
  power-from-weather estimator
