# Project Handoff: PH Enrollment Forecasting

This document is a guide for the writers and project owners taking over the manuscript and presentation. It explains what the technical pipeline does, how to use the outputs, and which numbers go in which manuscript section.

Contributors:
Aliyah Salmorin, Jan Marvin Polintan & Junnel Francis Jonson

---

## 1. Project Summary

**Research question:** Can past Philippine school enrollment data (2010–2020) reliably predict future enrollment, and how trustworthy are those predictions when validated against real DepEd data?

**Final answer (one sentence):** Short-horizon forecasts (1–3 years) are reliable to within ~1% nationally, but a simple "same as last year" rule outperformed more sophisticated models, and longer-horizon forecasts (such as 2030) are not reliable from this dataset alone.

**Methodology framework:** SEMMA (Sample, Explore, Modify, Model, Assess).

---

## 2. How to Re-Run the Pipeline

The pipeline is fully reproducible. Anyone with Python installed can produce identical results.

### Requirements

- Python 3.10 or newer
- Packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

To install the packages (one-time):

```
pip install -r requirements.txt
```

### Run order

Open a terminal in the project folder and run, in this exact order:

```
python enrollment_cleaning.py
python enrollment_eda.py
python enrollment_modeling.py
python enrollment_validation.py
```

Each script prints a clean status report. Outputs land in their own subfolders.

---

## 3. File Structure

```
Dataset and Predictive Analysis Activitiy/
|
|-- ph_enrollment (final).xlsx          (original, kept for reference)
|-- ML_Ready_Dataset.csv                 (raw export from xlsx)
|-- enrollment_2024-25.csv               (real DepEd data for validation)
|
|-- enrollment_cleaning.py               (script 1)
|-- enrollment_eda.py                    (script 2)
|-- enrollment_modeling.py               (script 3)
|-- enrollment_validation.py             (script 4)
|
|-- enrollment_cleaned.csv               (output of script 1)
|
|-- eda_outputs/                         (output of script 2)
|-- model_outputs/                       (output of script 3)
|-- validation_outputs/                  (output of script 4)
|
|-- HANDOFF.md                           (this file)
```

---

## 4. What Each Script Does

### Script 1 — `enrollment_cleaning.py`

**Purpose:** Clean the raw export and prepare it for analysis.

**What it does in plain terms:**
- Removes the title banner row from the Excel export.
- Renames the long, multi-line column names to short, clean names.
- Converts numbers stored as text (like "1,008,026") into actual numbers.
- Adds "last year's value" columns (called lag features) which are the most important inputs for prediction.

**Output:** `enrollment_cleaned.csv` — a tidy version of the dataset, ready for analysis.

---

### Script 2 — `enrollment_eda.py`

**Purpose:** Look at the cleaned data and produce charts that describe it. This is the SEMMA "Explore" stage.

**What it does in plain terms:**
- Calculates summary statistics (averages, ranges, etc.).
- Draws charts of national enrollment over time, by sector, by region, and by education level.
- Builds a correlation heatmap that shows which numbers move together.
- Shows the distribution of total enrollment per region+sector.
- Writes a plain-language summary the manuscript can paraphrase.

**Outputs (in `eda_outputs/`):**

| File | What it shows |
|------|---------------|
| `summary_statistics.csv` | Min, max, average, etc. per number column |
| `enrollment_trend_national.png` | National total over time |
| `enrollment_by_sector.png` | Public vs Private vs SUC-LUC over time |
| `enrollment_by_region.png` | All 17 regions over time |
| `level_breakdown.png` | Elementary, HS, SHS stacked over time |
| `correlation_heatmap.png` | Required by the rubric |
| `correlation_table.csv` | Same numbers as the heatmap, for citing in text |
| `distribution_total_students.png` | Histogram + boxplot of enrollment values |
| `eda_summary.txt` | Plain-language narrative for the writers |

---

### Script 3 — `enrollment_modeling.py`

**Purpose:** Train four different "predictors" on the historical data, compare them, and use the best ones to forecast 2021–2024. This is the SEMMA "Model" stage.

**The four predictors:**

1. **Naive baseline** — predicts "same as last year". The simplest possible rule.
2. **Ridge Regression** — fits a straight-line trend to the data.
3. **Gradient Boosted Trees (GBT)** — learns a forest of small if-then rules.
4. **SVR** — tries to fit a smooth curve through the data.

**What the script does:**
- Splits the cleaned data so the predictors learn from 2011–2018 and are tested on 2019–2020.
- Trains and grades all four predictors using identical conditions for a fair comparison.
- Reports which model came closest based on average miss (MAE).
- Lists the most important inputs each model relied on.
- Generates a "predicted vs. actual" scatter chart and a residual plot.
- Forecasts 2021, 2022, 2023, and 2024 using a rolling method (each year's prediction feeds into the next year's input).

**Outputs (in `model_outputs/`):**

| File | What it contains |
|------|------------------|
| `model_comparison.csv` | Grade card for all four models on the test years |
| `feature_importance_ridge.csv` | Ridge's coefficient for each input |
| `feature_importance_gbt.csv` | GBT's importance score for each input |
| `forecast_2021_to_2024.csv` | Year-by-year predictions per region+sector for Naive, Ridge, GBT |
| `predicted_vs_actual_test.png` | Scatter and residual plots for the test years |
| `modeling_summary.txt` | Plain-language narrative for the writers |

**Important note:** SVR is excluded from forecasting because it failed on the test set (negative R²). This is itself a finding worth mentioning briefly in the manuscript's discussion.

---

### Script 4 — `enrollment_validation.py`

**Purpose:** Compare the 2024 predictions against real DepEd 2024-25 enrollment data. This is the SEMMA "Assess" stage and the most important deliverable.

**What it does in plain terms:**
- Loads the predictions from script 3.
- Loads the real DepEd 2024-25 data (school-level, 60,000+ rows).
- Drops PSO (overseas schools) since the training data did not include them.
- Remaps NIR (Negros Island Region) schools back to their pre-2015 regions so the comparison is fair.
- Fixes naming differences (e.g., "SUC/LUC" becomes "SUC-LUC").
- Sums up the school-level numbers to match the prediction's region+sector level.
- Joins predictions with actuals and computes the gap for each model.
- Generates a final scatter chart and writes a narrative summary.

**Outputs (in `validation_outputs/`):**

| File | What it contains |
|------|------------------|
| `actuals_2024_per_region_sector.csv` | The cleaned real 2024 numbers, ready to compare |
| `validation_2024_per_region.csv` | Predicted vs actual per region+sector, with errors |
| `validation_summary_metrics.csv` | Final grade card on real 2024 data |
| `validation_predicted_vs_actual.png` | Scatter chart for the manuscript and slides |
| `validation_summary.txt` | Plain-language narrative for the writers |

---

## 5. Manuscript Section Mapping

This shows which output goes in which manuscript section, so the writers know exactly where to look.

| Manuscript Section | Use these outputs |
|--------------------|-------------------|
| **7a. Data Source** | `eda_outputs/eda_summary.txt` (dataset shape paragraph) |
| **7b. EDA + Correlation** | All charts in `eda_outputs/`, especially `correlation_heatmap.png` and `correlation_table.csv`. Paraphrase from `eda_summary.txt` |
| **7c. Data Preprocessing** | Walk through `enrollment_cleaning.py` step by step. Each step is commented in plain English |
| **7d. Machine Learning Models** | `modeling_summary.txt` setup section. List the four algorithms and the train/test split |
| **8. Results — Model Performance** | `model_comparison.csv` and `validation_summary_metrics.csv` |
| **8. Results — Model Interpretation** | `feature_importance_gbt.csv` and `feature_importance_ridge.csv` |
| **8. Visuals (in lieu of confusion matrix)** | `predicted_vs_actual_test.png` and `validation_predicted_vs_actual.png` |
| **9. Discussion + Conclusion** | Paraphrase from `validation_summary.txt`, especially the "interpretation" and "implications" sections |
| **9. Limitations** | "Notes / caveats" section in `validation_summary.txt`, plus the SVR failure point |

---

## 6. Headline Numbers for Abstract and Results

These are the figures the manuscript will likely highlight. Lift them directly into the abstract, results, and conclusion.

| Number | What it represents |
|--------|--------------------|
| **26,375,327** | Real Philippine enrollment in SY 2024-25 (DepEd) |
| **26,149,200** | Naive baseline prediction for 2024-25 |
| **-0.86%** | Naive's national error vs reality |
| **24,076,908** | Ridge Regression prediction for 2024-25 |
| **-8.71%** | Ridge's national error |
| **25,619,685** | GBT prediction for 2024-25 |
| **-2.86%** | GBT's national error |
| **17** | Number of regions covered |
| **3** | Number of sectors (Public, Private, SUC-LUC) |
| **560** | Rows in the cleaned dataset |
| **2010–2020** | Training data time range |
| **r = 0.999** | Correlation between last year's enrollment and this year's (the key finding from EDA) |
| **+14.8%** | National enrollment growth from 2010-11 to 2020-21 |
| **86%** | Public sector share of total enrollment |

---

## 7. Key Findings (for Discussion section)

1. **Simple beat complex.** The naive "predict last year's value" baseline outperformed Ridge Regression, Gradient Boosted Trees, and SVR on both the test set and the real 2024 validation. This is unusual and worth highlighting.

2. **The trend-fitting models were misled by COVID.** Both Ridge and GBT projected enrollment would decline in 2024 because they learned a downward trend from the 2020 dip. Reality showed enrollment stayed nearly flat. This is a real-world example of how anomalous training data can fool sophisticated models.

3. **Year-over-year persistence dominates the signal.** The correlation between this year's enrollment and last year's is 0.999 — almost perfect. This means there is very little additional pattern for fancier models to exploit.

4. **Three-year forecasts are reliable; longer horizons are not.** At a 3–4 year horizon, the best model was off by less than 1% nationally. At a 9-year horizon (to 2030), errors would compound — likely to 5–10% or more. A 2030 prediction from this dataset alone should be treated as a scenario projection with explicit assumptions, not a forecast.

5. **SVR completely failed** for this kind of data — the wide variance in regional sizes (5K to 3M students) confused the algorithm. Worth noting as a methodological observation.

---

## 8. Limitations (for the Limitations section)

- The dataset only spans 2010–2020 (11 annual data points), which is a small sample for time-series modeling.
- The 2020-21 row reflects the COVID disruption and influences trend-based models negatively.
- Predictions are at the region+sector level, not per-school. Individual school forecasting was out of scope.
- PSO (overseas schools) and NIR (a region created post-2015) had to be excluded or remapped during validation, so the comparison is not 100% complete.
- Two region+sector combinations (Region II SUC-LUC, Region XI SUC-LUC) had no matching actuals in 2024 and were dropped during validation.
- Reliable long-horizon forecasts (5+ years out) would require external data such as PSA birth rates and DepEd cohort survival rates.

---

## 9. Future Work (for the Future Work section)

- Add demographic data (PSA birth rates) as an external anchor for long-horizon projections.
- Include more recent years (2021–2024) in training to update the model's view of post-COVID patterns.
- Try cohort-tracking instead of region-level aggregation — follow Grade 1 students of 2015 as they progress through the system.
- Test ensemble methods that combine the naive baseline with the trend-aware models.
- Build per-region models if more data per region becomes available.

---

## 10. Reproducibility Statement

All results in this project are reproducible. Running the four scripts in order will produce byte-for-byte identical CSVs and visually identical PNGs. Random number generators are seeded with `random_state=42` throughout.

**Software versions used:**
- Python 3.13
- scikit-learn 1.8.0
- pandas, numpy, matplotlib, seaborn (latest stable)

The manuscript may include this exact statement in the methods section.

---

## 11. Contact / Questions

For any questions about how the code works, refer to the comments inside each `.py` file — every step is explained in plain English. The plain-language summaries inside each output folder (`eda_summary.txt`, `modeling_summary.txt`, `validation_summary.txt`) are written to be paraphrased directly into the manuscript.
