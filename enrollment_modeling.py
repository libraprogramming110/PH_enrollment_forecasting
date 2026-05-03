# =============================================================
# what this file does:
#
# this script trains 4 different "predictors" on the same
# historical enrollment data and compares how well each one
# predicts years it has never seen. then it uses the predictors
# to forecast 2021 through 2024 so we can later check those
# forecasts against real depEd numbers.
#
# the 4 predictors:
#   1. naive baseline   - just guesses "same as last year"
#   2. ridge regression - draws a straight-line trend
#   3. gradient boosted trees (gbt) - learns lots of small rules
#   4. svr - tries to fit a smooth curve through the data
#
# what comes IN  : enrollment_cleaned.csv
# what comes OUT : a folder called model_outputs/ with the
#                  comparison table, importance scores,
#                  forecasts, charts, and a plain summary.
# =============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# the actual predictor algorithms come from scikit-learn (sklearn)
# which is the standard machine learning library for python.
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

# helpers we use to scale numbers and convert text categories
# (like "ncr") into number columns the predictors can use.
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# scoring functions - they grade how good a prediction is
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# silence non-critical warnings so the output stays clean
warnings.filterwarnings('ignore')

input_file = 'enrollment_cleaned.csv'
out_dir    = 'model_outputs'
os.makedirs(out_dir, exist_ok=True)

print("=" * 60)
print("  modeling - ph enrollment forecasting")
print("=" * 60)


# -------------------------------------------------------------
# step 1: load and prepare the data
# -------------------------------------------------------------
df = pd.read_csv(input_file)
print(f"\n[1] loaded {input_file}: {df.shape}")

# the very first year (sy 2010-11) has no "last year" data,
# so the prev_* columns are blank. we have to drop those rows
# because the predictors need those columns to learn from.
df = df.dropna(subset=['prev_total_students']).reset_index(drop=True)
print(f"    after dropping sy 2010-11 rows : {df.shape}")

# pandas stored the prev_* columns as decimals (because of the
# blank cells we just dropped). now that there are no blanks,
# convert them back to whole numbers.
for c in ['prev_total_elementary', 'prev_total_hs',
          'prev_total_shs', 'prev_total_students']:
    df[c] = df[c].astype(int)


# -------------------------------------------------------------
# step 2: pick which columns are inputs vs. the answer
# -------------------------------------------------------------
# inputs (called "features" in ml) are the things the predictor
# uses to make its guess. the answer (called "target") is what
# we want to predict.
numeric_features = [
    'ay_start',                # the year (so it can spot trends over time)
    'shs_available',           # 0 before 2016, 1 from 2016 onwards
    'prev_total_elementary',   # last year's elementary count
    'prev_total_hs',           # last year's high school count
    'prev_total_shs',           # last year's senior high count
    'prev_total_students',     # last year's overall total - the strongest clue
]
categorical_features = ['region', 'sector']
target = 'total_students'   # the column we want to predict

# computers can't compare text directly, so we convert each
# text category into a set of yes/no number columns. for example,
# the "region" column becomes 17 columns: region_ncr (0 or 1),
# region_calabarzon (0 or 1), and so on. this is called
# "one-hot encoding".
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe.fit(df[categorical_features])
ohe_cols = list(ohe.get_feature_names_out(categorical_features))

def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """take a chunk of data and produce the same set of input
    columns the predictors expect (numbers + one-hot columns)."""
    cat = pd.DataFrame(
        ohe.transform(frame[categorical_features]),
        columns=ohe_cols, index=frame.index,
    )
    return pd.concat([frame[numeric_features].reset_index(drop=True),
                      cat.reset_index(drop=True)], axis=1)

x = build_features(df)            # all the input columns
y = df[target].values              # the answer column

print(f"\n[2] feature matrix shape: {x.shape}")
print(f"    numeric inputs  : {numeric_features}")
print(f"    one-hot inputs  : {len(ohe_cols)} columns "
      f"({df['region'].nunique()} regions + {df['sector'].nunique()} sectors)")


# -------------------------------------------------------------
# step 3: split the data into "study material" and "exam"
# -------------------------------------------------------------
# train = the years the predictor is allowed to study
# test  = the years we hold back to grade the predictor
#
# we split by year (not random rows) because this is a
# forecasting problem. a random split would let the predictor
# accidentally see the future during training, which would
# make the test grades misleadingly high. that mistake is
# called "data leakage" and we want to avoid it.
train_mask = df['ay_start'] <= 2018
test_mask  = df['ay_start'] >= 2019

x_train, x_test = x[train_mask].reset_index(drop=True), x[test_mask].reset_index(drop=True)
y_train, y_test = y[train_mask], y[test_mask]

print(f"\n[3] train: {x_train.shape[0]} rows (2011-2018) | "
      f"test: {x_test.shape[0]} rows (2019-2020)")

# svr is sensitive to the scale of numbers (some of our columns
# are in the millions, others are 0 or 1). we rescale all the
# inputs to a similar range just for svr.
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled  = scaler.transform(x_test)


# -------------------------------------------------------------
# step 4: a helper that grades predictions
# -------------------------------------------------------------
# every grade tells us something different:
#   mae    = average miss in students (easiest to understand)
#   rmse   = like mae but punishes big misses extra
#   mape % = average miss as a percentage of the actual value
#   r2     = how much of the pattern was captured (1.0 is perfect)
def score(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # mape divides by the true value, so we skip any rows where
    # the true value is zero (would cause divide-by-zero).
    nonzero = y_true != 0
    mape = (np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])).mean() * 100
    r2   = r2_score(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'MAPE_%': mape, 'R2': r2}


# -------------------------------------------------------------
# step 5: train and grade all 4 predictors on the same split
# -------------------------------------------------------------
# every predictor sees the same training data and is graded on
# the same exam. this makes the comparison fair.
print(f"\n[5] training 4 predictors on identical train/test split")

# 5a. naive baseline: prediction = whatever last year's total was.
#     no actual training happens here - we just copy a column.
naive_pred = x_test['prev_total_students'].values
naive_scores = score(y_test, naive_pred)

# 5b. ridge regression: tries to fit the best straight line.
#     alpha=1.0 is a small "penalty" that prevents the line
#     from getting too sensitive to any single column.
#     random_state=42 makes results reproducible (same numbers
#     every time you run the script).
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(x_train, y_train)            # study
ridge_pred = ridge.predict(x_test)     # take the exam
ridge_scores = score(y_test, ridge_pred)

# 5c. gradient boosted trees: builds 200 small decision trees,
#     each correcting the mistakes of the previous one.
#     max_depth=3 keeps each tree small (only 3 levels deep)
#     so it doesn't memorize noise.
#     learning_rate=0.05 means each tree contributes only a
#     small adjustment to the prediction.
gbt = GradientBoostingRegressor(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.9, random_state=42,
)
gbt.fit(x_train, y_train)
gbt_pred = gbt.predict(x_test)
gbt_scores = score(y_test, gbt_pred)

# 5d. svr (support vector regression): tries to find a smooth
#     curve through the data. rbf is the type of curve.
#     uses scaled inputs because it's sensitive to scale.
svr = SVR(kernel='rbf', C=10.0, gamma='scale')
svr.fit(x_train_scaled, y_train)
svr_pred = svr.predict(x_test_scaled)
svr_scores = score(y_test, svr_pred)

# put all the scores in one table for an easy side-by-side view
results = pd.DataFrame({
    'Naive (last year)': naive_scores,
    'Ridge Regression' : ridge_scores,
    'Gradient Boosted Trees': gbt_scores,
    'SVR (RBF kernel)' : svr_scores,
}).T.round(2)
results.index.name = 'Model'

print("\n    test-set performance:")
print(results.to_string())

results.to_csv(f'{out_dir}/model_comparison.csv')
print(f"\n    saved -> {out_dir}/model_comparison.csv")


# -------------------------------------------------------------
# step 6: pick the predictor with the lowest average miss
# -------------------------------------------------------------
# .idxmin() returns the row name (model name) of the smallest
# value in the mae column.
winner_name = results['MAE'].idxmin()
print(f"\n[6] winning predictor (lowest mae): {winner_name}")

# we don't actually use this lookup table after this point in
# the new flow, but it's kept here in case future versions of
# the script want to grab the winning model object.
winner_lookup = {
    'Naive (last year)': ('naive', None, naive_pred),
    'Ridge Regression' : ('ridge', ridge, ridge_pred),
    'Gradient Boosted Trees': ('gbt', gbt, gbt_pred),
    'SVR (RBF kernel)' : ('svr', svr, svr_pred),
}
winner_key, winner_model, winner_pred = winner_lookup[winner_name]


# -------------------------------------------------------------
# step 7: see which inputs each predictor relied on most
# -------------------------------------------------------------
# ridge's "coefficient" tells us how strongly each input pushes
# the prediction up or down. bigger absolute number = bigger pull.
ridge_coef = pd.DataFrame({
    'feature': x_train.columns,
    'coefficient': ridge.coef_,
}).sort_values('coefficient', key=abs, ascending=False)
ridge_coef.to_csv(f'{out_dir}/feature_importance_ridge.csv', index=False)

# gbt's "feature_importances_" tells us how often each input
# was used to make a split decision in the trees.
gbt_imp = pd.DataFrame({
    'feature': x_train.columns,
    'importance': gbt.feature_importances_,
}).sort_values('importance', ascending=False)
gbt_imp.to_csv(f'{out_dir}/feature_importance_gbt.csv', index=False)

print(f"\n[7] feature importance saved")
print(f"    top 5 gbt features:")
print(gbt_imp.head(5).to_string(index=False))


# -------------------------------------------------------------
# step 8: charts that replace the "confusion matrix"
# -------------------------------------------------------------
# the rubric asks for a confusion matrix, but those are only
# valid for classification problems (yes/no, cat/dog). we are
# doing regression (predicting a continuous number), so we use
# the standard regression diagnostic plots instead:
#   left  - predicted vs. actual scatter (closer to the y=x
#           dashed line means better predictions)
#   right - residual plot (errors vs. predictions; should look
#           like a random cloud around zero)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for name, pred, color, marker in [
    ('Naive', naive_pred, '#888888', 'x'),
    ('Ridge', ridge_pred, '#1f77b4', 'o'),
    ('GBT',   gbt_pred,   '#2ca02c', 's'),
    ('SVR',   svr_pred,   '#d62728', '^'),
]:
    ax.scatter(y_test, pred, alpha=0.55, s=35, label=name,
               color=color, marker=marker, edgecolor='white', linewidth=0.4)
lo, hi = y_test.min(), y_test.max()
ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, label='perfect (y=x)')
ax.set_xlabel('actual total_students')
ax.set_ylabel('predicted total_students')
ax.set_title('predicted vs. actual on test set (2019-2020)')
ax.ticklabel_format(style='plain')
ax.legend(fontsize=8)

ax = axes[1]
residuals = y_test - winner_pred
ax.scatter(winner_pred, residuals, alpha=0.6,
           color='#2ca02c', edgecolor='white', linewidth=0.4)
ax.axhline(0, color='k', linestyle='--', linewidth=1)
ax.set_xlabel(f'predicted ({winner_name})')
ax.set_ylabel('residual (actual - predicted)')
ax.set_title(f'residual plot - {winner_name}')
ax.ticklabel_format(style='plain')

plt.tight_layout()
plt.savefig(f'{out_dir}/predicted_vs_actual_test.png')
plt.close()
print(f"\n[8] diagnostic plots saved -> predicted_vs_actual_test.png")


# -------------------------------------------------------------
# step 9: rolling forecast for 2021 through 2024
# -------------------------------------------------------------
# the key idea here is "rolling" the forecast forward year by year:
#   - to predict 2021, use 2020's actual numbers as "last year"
#   - to predict 2022, use the 2021 PREDICTION as "last year"
#   - to predict 2023, use the 2022 PREDICTION as "last year"
#   - and so on through 2024.
#
# this is more honest than peeking at later years, because in
# real life we wouldn't have the 2021/2022/2023 numbers when
# making a 2024 forecast.
#
# we run this rolling forecast for naive, ridge, and gbt.
# svr is dropped because its r-squared was negative on the
# test set (worse than just predicting the average).
print(f"\n[9] forecasting 2021-2024 with naive, ridge, gbt (svr dropped)")

# retrain ridge and gbt on ALL the historical data (2011-2020)
# so the final forecast benefits from every year we have.
ridge_full = Ridge(alpha=1.0, random_state=42).fit(x, y)
gbt_full = GradientBoostingRegressor(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.9, random_state=42,
).fit(x, y)

# the seed for the forecast = sy 2020-21 actuals.
# one row per region+sector combination.
seed = df[df['ay_start'] == 2020].copy().reset_index(drop=True)

# each predictor needs its OWN running state, because each one's
# 2021 prediction is different and feeds into its own 2022.
def init_state(seed_df):
    """take 2020 actuals and rename them to look like prev_* inputs."""
    s = seed_df[['region', 'sector', 'shs_available',
                 'total_elementary', 'total_hs', 'total_shs',
                 'total_students']].copy()
    return s.rename(columns={
        'total_elementary': 'prev_total_elementary',
        'total_hs'        : 'prev_total_hs',
        'total_shs'       : 'prev_total_shs',
        'total_students'  : 'prev_total_students',
    })

state_naive = init_state(seed)
state_ridge = init_state(seed)
state_gbt   = init_state(seed)

# we'll collect each year's prediction here, then save them all together
forecast_rows = []
for forecast_year in [2021, 2022, 2023, 2024]:
    # --- naive: prediction = last year's value, copied ---
    naive_step = state_naive.copy()
    naive_step['ay_start'] = forecast_year
    naive_step['shs_available'] = 1   # post-2016 always has shs
    pred_naive = np.clip(naive_step['prev_total_students'].values, 0, None).round().astype(int)

    # --- ridge: ask the trained ridge model for a prediction ---
    ridge_step = state_ridge.copy()
    ridge_step['ay_start'] = forecast_year
    ridge_step['shs_available'] = 1
    pred_ridge = np.clip(ridge_full.predict(build_features(ridge_step)), 0, None).round().astype(int)

    # --- gbt: ask the trained gbt model for a prediction ---
    gbt_step = state_gbt.copy()
    gbt_step['ay_start'] = forecast_year
    gbt_step['shs_available'] = 1
    pred_gbt = np.clip(gbt_full.predict(build_features(gbt_step)), 0, None).round().astype(int)

    # combine all three predictions for this year into one table
    out = seed[['region', 'sector']].copy()
    out['ay_start']             = forecast_year
    out['naive_predicted']      = pred_naive
    out['ridge_predicted']      = pred_ridge
    out['gbt_predicted']        = pred_gbt
    forecast_rows.append(out)

    # roll forward: each predictor's prediction becomes its own
    # "last year" for the next round.
    state_naive['prev_total_students'] = pred_naive
    state_ridge['prev_total_students'] = pred_ridge
    state_gbt['prev_total_students']   = pred_gbt

# stack all years into one big table and save it
forecast = pd.concat(forecast_rows, ignore_index=True)
forecast.to_csv(f'{out_dir}/forecast_2021_to_2024.csv', index=False)

# sum across all regions+sectors per year for an easy national view
national_forecast = (
    forecast.groupby('ay_start')[['naive_predicted', 'ridge_predicted', 'gbt_predicted']]
    .sum().reset_index()
)
print("\n    national forecast totals per year (per model):")
print(national_forecast.to_string(index=False))

print(f"\n    saved -> {out_dir}/forecast_2021_to_2024.csv")


# -------------------------------------------------------------
# step 10: write a plain-language summary for the writers
# -------------------------------------------------------------
top3_gbt = gbt_imp.head(3)
top3_ridge = ridge_coef.head(3)

row_2024 = national_forecast.loc[national_forecast['ay_start'] == 2024].iloc[0]
row_2021 = national_forecast.loc[national_forecast['ay_start'] == 2021].iloc[0]
forecast_2024_naive = int(row_2024['naive_predicted'])
forecast_2024_ridge = int(row_2024['ridge_predicted'])
forecast_2024_gbt   = int(row_2024['gbt_predicted'])
forecast_2021_naive = int(row_2021['naive_predicted'])

summary = f"""\
================================================================
  modeling plain-language summary  (paraphrase into manuscript 7d / 8)
================================================================

experimental setup
  - dataset       : enrollment_cleaned.csv ({df.shape[0]} rows after
                    dropping sy 2010-11 due to missing lag features)
  - train period  : sy 2011-12 to sy 2018-19 ({x_train.shape[0]} rows)
  - test  period  : sy 2019-20 to sy 2020-21 ({x_test.shape[0]} rows)
  - split type    : time-based (no random shuffle - prevents leakage)
  - target (y)    : total_students per region+sector+year
  - features (x)  : 6 numeric (ay_start, shs_available, 4 prev_* lags)
                  + {len(ohe_cols)} one-hot columns (17 regions + 3 sectors)

algorithms compared
  1. naive baseline   - prediction = prev_total_students (last year)
  2. ridge regression - linear with l2 penalty (alpha=1.0)
  3. gradient boosted trees - 200 estimators, depth 3, lr 0.05
  4. svr (rbf kernel) - c=10, scaled features

test-set results
{results.to_string()}

winner (lowest mae): {winner_name}

feature importance
  top 3 gbt features:
{top3_gbt.to_string(index=False)}

  top 3 ridge coefficients (by absolute value):
{top3_ridge.to_string(index=False)}

  insight: the prev_* lag features dominate, confirming what eda
  already showed (correlation r ~ 0.99). the model is essentially
  learning year-over-year persistence with regional/sectoral
  adjustments.

rolling forecast (national totals, per model)
  year     naive (last year)   ridge regression   gbt
  2021-22  {forecast_2021_naive:>15,}   (see csv)         (see csv)
  2024-25  {forecast_2024_naive:>15,}   {forecast_2024_ridge:>15,}   {forecast_2024_gbt:>15,}

  svr is excluded from the forecast because it failed on the test
  set (r^2 < 0). all 3 remaining models will be compared against
  the real deped 2024-25 figures in the assess stage.

notes on confusion matrix
  this is a regression problem - confusion matrices apply only to
  classification. per-class accuracy is replaced with the standard
  regression diagnostics: mae, rmse, mape, r-squared, plus the
  predicted-vs-actual scatter and residual plot in
  predicted_vs_actual_test.png.

next stage (semma - assess): enrollment_validation.py
================================================================
"""

with open(f'{out_dir}/modeling_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

print(f"\n[10] plain-language summary saved -> {out_dir}/modeling_summary.txt")

print(f"\n{'='*60}")
print(f"  modeling complete - all outputs in '{out_dir}/'")
print(f"{'='*60}")
print(f"\nnext: enrollment_validation.py  (semma - assess)")
print(f"      uses enrollment_2024-25.csv to validate the 2024 forecast.")
