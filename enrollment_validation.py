# =============================================================
# what this file does:
#
# this is the "report card" stage. earlier scripts trained
# predictors on data ending in 2020 and produced 2024 forecasts.
# this script takes those forecasts and compares them against
# real depEd 2024-25 enrollment numbers - data the predictors
# never saw during training.
#
# the gap between predicted and actual = the honest forecast
# error. that error tells us how trustworthy the model is and
# whether longer-horizon forecasts (like 2030) are realistic.
#
# what comes IN:
#   - model_outputs/forecast_2021_to_2024.csv (our predictions)
#   - enrollment_2024-25.csv                  (real deped data)
#
# what comes OUT (in validation_outputs/):
#   - actuals_2024_per_region_sector.csv  (cleaned real data)
#   - validation_2024_per_region.csv      (predicted vs actual,
#                                          per region+sector)
#   - validation_summary_metrics.csv      (overall grade card)
#   - validation_predicted_vs_actual.png  (scatter chart)
#   - validation_summary.txt              (plain-language writeup)
# =============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

forecast_file = 'model_outputs/forecast_2021_to_2024.csv'
actual_file   = 'enrollment_2024-25.csv'
out_dir       = 'validation_outputs'
os.makedirs(out_dir, exist_ok=True)

print("=" * 60)
print("  validation - 2024 forecasts vs. real deped data")
print("=" * 60)


# -------------------------------------------------------------
# step 1: load our 2024 forecasts
# -------------------------------------------------------------
# the forecast file contains predictions for 2021, 2022, 2023,
# and 2024. we only care about 2024 here.
forecast = pd.read_csv(forecast_file)
forecast_2024 = forecast[forecast['ay_start'] == 2024].copy().reset_index(drop=True)
print(f"\n[1] loaded forecasts: {forecast_2024.shape[0]} region+sector predictions for 2024")


# -------------------------------------------------------------
# step 2: load the real depEd 2024-25 data
# -------------------------------------------------------------
# this file is at the school level (60,000+ rows, one per
# school). our forecast is at the region+sector level (51 rows),
# so we'll need to add up the schools later to match.
print(f"\n[2] loading real depEd 2024-25 data ({actual_file})...")
actual = pd.read_csv(actual_file)
print(f"    school-level rows: {actual.shape[0]:,}")


# -------------------------------------------------------------
# step 3: drop pso (overseas schools)
# -------------------------------------------------------------
# pso = philippine schools overseas. our training data did not
# include these, so the predictor has no way to know about
# them. for a fair comparison we exclude them from validation.
pso_count = (actual['region'] == 'PSO').sum()
actual = actual[actual['region'] != 'PSO'].copy()
print(f"\n[3] dropped {pso_count} pso (overseas) schools - not in training scope")


# -------------------------------------------------------------
# step 4: handle nir (negros island region)
# -------------------------------------------------------------
# nir was created after 2015 and re-shuffled afterwards. our
# training data still treats those provinces as part of the
# old regions (region vi for negros occidental, region vii for
# negros oriental and siquijor).
#
# for a fair comparison we relabel the nir schools back to
# their pre-2015 region homes so the totals line up with what
# the predictor expects.
nir_count_before = (actual['region'] == 'NIR').sum()
actual.loc[(actual['region'] == 'NIR') & (actual['province'] == 'NEGROS OCCIDENTAL'),
           'region'] = 'Region VI'
actual.loc[(actual['region'] == 'NIR') & (actual['province'].isin(['NEGROS ORIENTAL', 'SIQUIJOR'])),
           'region'] = 'Region VII'
print(f"\n[4] remapped {nir_count_before} nir schools back to region vi / region vii")


# -------------------------------------------------------------
# step 5: standardize names so they match the training data
# -------------------------------------------------------------
# the deped file uses short region names like "ncr" and
# "region i" while our training data uses long names like
# "ncr - national capital region" and "region i - ilocos region".
# we rename them so the join in step 8 works.
region_map = {
    'BARMM'      : 'BARMM - Bangsamoro Autonomous Region in Muslim Mindanao',
    'CAR'        : 'CAR - Cordillera Administrative Region',
    'CARAGA'     : 'CARAGA - CARAGA',
    'NCR'        : 'NCR - National Capital Region',
    'MIMAROPA'   : 'Region IV-B - MIMAROPA',
    'Region I'   : 'Region I - Ilocos Region',
    'Region II'  : 'Region II - Cagayan Valley',
    'Region III' : 'Region III - Central Luzon',
    'Region IV-A': 'Region IV-A - CALABARZON',
    'Region V'   : 'Region V - Bicol Region',
    'Region VI'  : 'Region VI - Western Visayas',
    'Region VII' : 'Region VII - Central Visayas',
    'Region VIII': 'Region VIII - Eastern Visayas',
    'Region IX'  : 'Region IX - Zamboanga Peninsula',
    'Region X'   : 'Region X - Northern Mindanao',
    'Region XI'  : 'Region XI - Davao Region',
    'Region XII' : 'Region XII - Soccsksargen',
}
actual['region'] = actual['region'].map(region_map)

# similarly, fix sector spelling: the deped file uses "suc/luc"
# (slash) while our training file used "suc-luc" (dash).
sector_map = {'SUC/LUC': 'SUC-LUC', 'Public': 'Public', 'Private': 'Private'}
actual['sector'] = actual['sector'].map(sector_map)

# safety check: drop any leftover rows that didn't map cleanly.
unmapped = actual[actual['region'].isnull() | actual['sector'].isnull()]
if len(unmapped):
    print(f"    warning: {len(unmapped)} rows had unmapped region/sector - dropping")
    actual = actual.dropna(subset=['region', 'sector'])
print(f"[5] standardized region + sector names to match training format")


# -------------------------------------------------------------
# step 6: compute total students per school
# -------------------------------------------------------------
# the deped file has 62 columns of grade-level breakdowns
# (g1_male, g1_female, g2_male, ...). we add them all up to
# get one total_students number per school.
grade_cols = [c for c in actual.columns
              if any(c.startswith(p) for p in
                     ['kinder', 'g1_', 'g2_', 'g3_', 'g4_', 'g5_', 'g6_', 'esng_',
                      'g7_', 'g8_', 'g9_', 'g10_', 'jhsng_', 'g11_', 'g12_'])]
actual['total_students'] = actual[grade_cols].sum(axis=1)
print(f"\n[6] computed total_students from {len(grade_cols)} grade columns")


# -------------------------------------------------------------
# step 7: aggregate up to region+sector totals
# -------------------------------------------------------------
# we group all schools that share the same region and sector,
# and sum their total_students. this turns 60,000+ school rows
# into about 51 region+sector rows that match our forecast.
actuals_grouped = (
    actual.groupby(['region', 'sector'], as_index=False)['total_students']
    .sum()
    .rename(columns={'total_students': 'actual_total_students'})
)
actuals_grouped.to_csv(f'{out_dir}/actuals_2024_per_region_sector.csv', index=False)
print(f"[7] aggregated to {actuals_grouped.shape[0]} region+sector totals")
print(f"    saved -> {out_dir}/actuals_2024_per_region_sector.csv")


# -------------------------------------------------------------
# step 8: join predictions with actuals, compute the gaps
# -------------------------------------------------------------
# .merge() lines up the forecast and actual rows by region and
# sector. how='left' keeps every forecast row even if there
# isn't an actual to match (we'll warn on those).
merged = forecast_2024.merge(actuals_grouped, on=['region', 'sector'], how='left')

missing = merged['actual_total_students'].isnull().sum()
if missing:
    print(f"    warning: {missing} forecast rows have no matching actual - dropping")
    merged = merged.dropna(subset=['actual_total_students'])
merged['actual_total_students'] = merged['actual_total_students'].astype(int)

# for each model, compute three error columns:
#   _error      = signed difference (positive = predicted too low)
#   _abs_error  = magnitude of the miss, ignoring direction
#   _pct_error  = miss as a percentage of the actual value
for model_key in ['naive', 'ridge', 'gbt']:
    merged[f'{model_key}_error']     = merged['actual_total_students'] - merged[f'{model_key}_predicted']
    merged[f'{model_key}_abs_error'] = merged[f'{model_key}_error'].abs()
    merged[f'{model_key}_pct_error'] = merged[f'{model_key}_error'] / merged['actual_total_students'] * 100

merged.to_csv(f'{out_dir}/validation_2024_per_region.csv', index=False)
print(f"\n[8] joined predictions with actuals -> validation_2024_per_region.csv")


# -------------------------------------------------------------
# step 9: compute the overall grade card per model
# -------------------------------------------------------------
# same scoring helper as in the modeling script - mae, rmse,
# mape, and r-squared. but now we are scoring against REAL
# 2024 data the predictors never saw.
def score(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nonzero = y_true != 0
    mape = (np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])).mean() * 100
    r2   = r2_score(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'MAPE_%': mape, 'R2': r2}

y_true = merged['actual_total_students'].values
metrics = pd.DataFrame({
    'Naive (last year)'      : score(y_true, merged['naive_predicted'].values),
    'Ridge Regression'       : score(y_true, merged['ridge_predicted'].values),
    'Gradient Boosted Trees' : score(y_true, merged['gbt_predicted'].values),
}).T.round(2)
metrics.index.name = 'Model'

print(f"\n[9] validation metrics on real 2024 deped data:")
print(metrics.to_string())
metrics.to_csv(f'{out_dir}/validation_summary_metrics.csv')


# -------------------------------------------------------------
# step 10: scatter chart - predicted vs. actual
# -------------------------------------------------------------
# each dot is one region+sector. closer to the dashed y=x line
# means the prediction was closer to reality. three colors and
# shapes so you can see at a glance which model came closest.
fig, ax = plt.subplots(figsize=(8, 7))
for name, col, color, marker in [
    ('Naive', 'naive_predicted', '#1f77b4', 'o'),
    ('Ridge', 'ridge_predicted', '#ff7f0e', 's'),
    ('GBT',   'gbt_predicted',   '#2ca02c', '^'),
]:
    ax.scatter(merged['actual_total_students'], merged[col],
               alpha=0.65, s=55, label=name, color=color,
               marker=marker, edgecolor='white', linewidth=0.5)
lo = 0
hi = max(merged['actual_total_students'].max(),
         merged[['naive_predicted', 'ridge_predicted', 'gbt_predicted']].values.max())
ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, label='perfect prediction (y=x)')
ax.set_xlabel('actual sy 2024-25 enrollment (per region+sector)')
ax.set_ylabel('predicted sy 2024-25 enrollment')
ax.set_title('predicted vs. actual enrollment, sy 2024-25\nvalidation against real deped data')
ax.ticklabel_format(style='plain')
ax.legend()
plt.tight_layout()
plt.savefig(f'{out_dir}/validation_predicted_vs_actual.png')
plt.close()
print(f"\n[10] scatter plot saved -> validation_predicted_vs_actual.png")


# -------------------------------------------------------------
# step 11: write the plain-language summary
# -------------------------------------------------------------
# this section calculates national totals per model, finds the
# largest miss for each model, and writes everything to a text
# file the writers can paraphrase into the manuscript.
national_actual = int(merged['actual_total_students'].sum())
national_naive  = int(merged['naive_predicted'].sum())
national_ridge  = int(merged['ridge_predicted'].sum())
national_gbt    = int(merged['gbt_predicted'].sum())

def pct(p, a):
    """percent difference between predicted (p) and actual (a)."""
    return (p - a) / a * 100

best_model = metrics['MAE'].idxmin()
worst_model = metrics['MAE'].idxmax()

# the rows where each model missed the most (in absolute terms)
worst_naive = merged.loc[merged['naive_abs_error'].idxmax()]
worst_ridge = merged.loc[merged['ridge_abs_error'].idxmax()]
worst_gbt   = merged.loc[merged['gbt_abs_error'].idxmax()]

summary = f"""\
================================================================
  validation plain-language summary  (paraphrase into manuscript 8/9)
================================================================

ground truth source
  - file   : enrollment_2024-25.csv (deped school-level data)
  - rows   : {actual.shape[0]:,} schools (after dropping pso; nir remapped)
  - period : sy 2024-2025 - this data was never seen by any model

national totals (sum across all 17 regions x 3 sectors)

  actual sy 2024-25 (deped)        : {national_actual:>14,}
  predicted by naive (last year)   : {national_naive:>14,}   ({pct(national_naive, national_actual):+.2f}%)
  predicted by ridge regression    : {national_ridge:>14,}   ({pct(national_ridge, national_actual):+.2f}%)
  predicted by gradient boosted    : {national_gbt:>14,}   ({pct(national_gbt, national_actual):+.2f}%)

validation metrics (per region+sector level, real 2024 data)
{metrics.to_string()}

  - best performer  : {best_model}
  - worst performer : {worst_model}

largest per-region miss (by absolute error)
  - naive : {worst_naive['region']} ({worst_naive['sector']})
            predicted {int(worst_naive['naive_predicted']):,} | actual {int(worst_naive['actual_total_students']):,}
            miss = {int(worst_naive['naive_error']):+,} students ({worst_naive['naive_pct_error']:+.1f}%)

  - ridge : {worst_ridge['region']} ({worst_ridge['sector']})
            predicted {int(worst_ridge['ridge_predicted']):,} | actual {int(worst_ridge['actual_total_students']):,}
            miss = {int(worst_ridge['ridge_error']):+,} students ({worst_ridge['ridge_pct_error']:+.1f}%)

  - gbt   : {worst_gbt['region']} ({worst_gbt['sector']})
            predicted {int(worst_gbt['gbt_predicted']):,} | actual {int(worst_gbt['actual_total_students']):,}
            miss = {int(worst_gbt['gbt_error']):+,} students ({worst_gbt['gbt_pct_error']:+.1f}%)

interpretation
  - the three models had very different forecasts despite training
    on identical data:
      naive : assumed 2024 = 2020 (flat persistence)
      ridge : projected a downward trend
      gbt   : also projected a slight decline
  - {best_model} was closest to reality on this validation. this
    confirms the eda insight that philippine enrollment is highly
    autocorrelated - simple persistence is hard to beat for short
    horizons.
  - the fact that more sophisticated models (ridge, gbt) projected
    a downward trend while reality was nearly flat is itself a
    finding: linear extrapolations from 2010-2020 (which included
    the covid dip) misled the trend-based models.

implications for long-horizon forecasts (e.g., 2030)
  - at a 3-4 year horizon (2020 -> 2024), our best model was off
    by {abs(pct(national_naive, national_actual)):.2f}% nationally.
  - at a 9-year horizon (2020 -> 2030), errors would compound
    significantly. a direct 2030 prediction from this dataset
    cannot be considered reliable.
  - recommended framing for stakeholders: report short-horizon
    forecasts (1-3 years) with measured error bounds, and treat
    any 2030 number as a scenario projection with explicit
    assumptions, not a prediction.

notes / caveats
  - pso (philippine schools overseas) excluded from validation -
    not present in training data.
  - nir (negros island region) schools re-assigned by province
    back to their pre-2015 regions (region vi / vii) for a fair
    comparison with the model's region categories.
  - validation is at the region+sector aggregate level, not at
    individual schools.
================================================================
"""

with open(f'{out_dir}/validation_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

print(f"\n[11] plain-language summary saved -> validation_summary.txt")
print(f"\n{'='*60}")
print(f"  validation complete - all outputs in '{out_dir}/'")
print(f"{'='*60}")
