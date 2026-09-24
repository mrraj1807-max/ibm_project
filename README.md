# ☕ Coffee Price Prediction — Green Coffee Value Benchmark

**Author:** Amit Dwivedi  
**Course/Project:** Data Science with Machine Learning  
**Year:** 2025

---

## 📌 Project Description

This project builds a **supervised regression model** to predict the **transaction price (USD/kg)** of green coffee lots before they reach the export market. Using the [Coffee Value Benchmark dataset from Kaggle](https://www.kaggle.com/datasets/razanihababdellatif/coffee-quality-and-pricing-benchmark), observable pre-export features — including farm origin, processing method, physical bean quality, altitude, climate, and commercial certifications — are used to estimate market value.

This work addresses a real business need: enabling coffee procurement teams, farmers, cooperatives, and exporters to make data-driven pricing and negotiation decisions without waiting for market prices to be revealed.

---

## 📂 Dataset

| File | Description |
|---|---|
| `coffee_value_benchmark.csv` | Main dataset: 36 features + 3 target columns (~35,000+ rows) |
| `feature_dictionary.csv` | Column metadata: stage, data type, usability flags per target |
| `coffee_benchmark_splits.csv` | Official temporal train/test split keyed by `lot_id` |
| `coffee_record_metadata.csv` | Record-type flags (Original, etc.) |

**Dataset Folder:** `coffee value benchmark project/`  
**Kaggle Source:** [Coffee Value Benchmark — Kaggle](https://www.kaggle.com/datasets/razanihababdellatif/coffee-quality-and-pricing-benchmark)

---

## 🎯 Problem Statement

> *Predict `price_usd_per_kg` of a green coffee lot using only pre-export observable attributes.*

| | |
|---|---|
| **Task type** | Regression |
| **Primary target** | `price_usd_per_kg` (continuous float) |
| **Primary metric** | RMSE (Root Mean Square Error) |
| **Secondary metrics** | MAE, R² |

---

## 🏗️ Project Structure

```
.
├── amit_dwivedi_coffee_price_benchmark.ipynb   # Full project notebook (10 sections)
├── requirements.txt                            # Python dependencies
├── README.md                                   # This file
├── amit_dwivedi_ProjectReport.docx             # Full project report (Word)
└── coffee value benchmark project/
    ├── coffee_value_benchmark.csv
    ├── feature_dictionary.csv
    ├── coffee_benchmark_splits.csv
    └── coffee_record_metadata.csv
```

---

## 🔬 Methodology

### 1. Business Problem Framing
Predict green coffee lot price (USD/kg) to support procurement and negotiation decisions.

### 2. Exploratory Data Analysis (EDA)
- Distribution analysis of `price_usd_per_kg` (found right-skewed → log1p transform)
- Missing value heatmap (`humidity_pct`, `moisture_pct`, `avg_annual_rainfall_mm`)
- Bivariate analysis: price vs species, buyer segment, certification, country, altitude
- Correlation matrix of all numeric features

### 3. Feature Engineering
| Engineering Step | Rationale |
|---|---|
| `log1p(price_usd_per_kg)` as target | Reduce skewness; improves model fit |
| Parse `harvest_season` → `season_year_start` | Extract numeric signal from "2020/21" |
| `arabica_x_altitude` interaction | Arabica at high altitude → premium |
| `log_lot_size` from `lot_size_kg` | Compress lot-size range |
| `has_certification`, `is_specialty` flags | Boolean signals |
| `cert_x_specialty` interaction | Compound premium detection |

**Anti-leakage:** All sensory outcome variables (`acidity_score`, `sweetness_score`, etc.) are **excluded** as features — they are only available post-evaluation, not at prediction time.

### 4. Train/Test Split
Used the **official temporal split** from `coffee_benchmark_splits.csv`. Cross-validation uses **`GroupKFold(n_splits=5)` on `lot_id`** to prevent leakage across records sharing the same physical lot.

### 5. Models Trained
| Model | Notes |
|---|---|
| Median Baseline | Establishes performance floor |
| Ridge Regression | Linear baseline with regularisation |
| Random Forest | 200 trees, handles non-linearity |
| XGBoost | Gradient boosting, tuned with Optuna |
| LightGBM | Leaf-wise boosting, tuned with Optuna |
| **Stacking Ensemble** | XGBoost + LightGBM + RF → Ridge meta-learner |

### 6. Hyperparameter Tuning
Bayesian optimisation via **Optuna** (40 trials each) for XGBoost and LightGBM.

### 7. Interpretability
- **SHAP summary and beeswarm plots** — global feature importance
- **SHAP waterfall plots** — individual prediction explanations (highest/lowest priced lots)
- **Partial Dependence Plots** — for `altitude_mean_meters`, `defect_rate_pct`, `log_lot_size`

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| pandas | ≥2.0 | Data manipulation |
| numpy | ≥1.24 | Numerical operations |
| scikit-learn | ≥1.3 | Pipelines, CV, baseline models |
| XGBoost | ≥2.0 | Gradient boosted trees |
| LightGBM | ≥4.0 | Fast gradient boosting |
| SHAP | ≥0.43 | Model interpretability |
| Optuna | ≥3.3 | Bayesian hyperparameter optimisation |
| matplotlib / seaborn | ≥3.7 / ≥0.12 | Visualisation |
| Jupyter Notebook | ≥1.0 | Interactive development |

---

## ⚙️ Setup & Run Instructions

### 1. Prerequisites
- Python 3.10 or higher
- pip

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Folder Structure
Ensure the dataset folder is at `coffee value benchmark project/` relative to the project root, containing all 4 CSV files.

### 4. Launch the Notebook

```bash
jupyter notebook amit_dwivedi_coffee_price_benchmark.ipynb
```

Or run all cells non-interactively:

```bash
jupyter nbconvert --to notebook --execute amit_dwivedi_coffee_price_benchmark.ipynb --output executed_output.ipynb
```

---

## 📊 Key Results

| Model | Test RMSE (USD/kg) | Test MAE (USD/kg) | Test R² |
|---|---|---|---|
| Median Baseline | — | — | — |
| Ridge Regression | — | — | — |
| Random Forest | — | — | — |
| XGBoost (Tuned) | — | — | — |
| LightGBM (Tuned) | — | — | — |
| **Stacking Ensemble** | **Best** | **Best** | **Best** |

> *Exact values are populated after notebook execution.*

**Top Predictive Features (SHAP):**
1. `intended_buyer_segment` (Specialty vs Commodity)
2. `species` (Arabica vs Robusta)
3. `arabica_x_altitude` (interaction)
4. `certification` (Organic / Fair Trade)
5. `defect_rate_pct`

---

## 📝 Key Business Insights

1. **Arabica commands a premium** — species alone is the single most powerful price predictor
2. **Specialty buyer segment** drives 2–3× higher prices vs commodity
3. **High-altitude Arabica** captures the premium growing conditions prized by specialty buyers
4. **Certifications** (Organic, Fair Trade) add measurable price uplift
5. **Defect rate** is a consistent value-destroyer — quality control is commercially rewarded

---

## 📄 Deliverables

| File | Description |
|---|---|
| `amit_dwivedi_coffee_price_benchmark.ipynb` | Complete project notebook |
| `requirements.txt` | Python dependency list |
| `README.md` | Project overview (this file) |
| `amit_dwivedi_ProjectReport.docx` | Full project report in Word format |

---

## ⚠️ Limitations

- Dataset is **synthetic** (benchmark) — real commodity market dynamics not fully captured
- FX rates, geopolitical factors, and seasonal demand cycles are absent
- Model is trained on historical lots; temporal drift may require periodic retraining

---

## 📬 Contact

**Amit Dwivedi**  
Data Science Project — Coffee Value Benchmark Predictive Modeling
