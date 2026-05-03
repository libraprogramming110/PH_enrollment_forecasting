# =============================================================
# what this file does:
#
# this script looks at the cleaned data and makes charts and
# tables to help us understand it. think of it like a doctor
# looking at x-rays before doing surgery. we are not building
# predictions yet, we are just trying to see what is in the data.
#
# the formal name for this stage is "exploratory data analysis"
# (eda for short). it answers basic questions like:
#   - how big are the numbers?
#   - what does enrollment look like over time?
#   - which regions are biggest?
#   - which numbers move together?
#
# what comes IN  : enrollment_cleaned.csv
# what comes OUT : a folder called eda_outputs/ with charts,
#                  tables, and a plain-language summary.
# =============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   # used to draw charts
import seaborn as sns             # makes the charts look nicer

input_file = 'enrollment_cleaned.csv'
out_dir    = 'eda_outputs'

# create the output folder if it doesn't exist yet.
# exist_ok=true means "don't complain if it's already there".
os.makedirs(out_dir, exist_ok=True)

# set a nicer default style for all the charts in this script.
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 110

print("=" * 60)
print("  exploratory data analysis - ph enrollment")
print("=" * 60)

# -------------------------------------------------------------
# load the cleaned data
# -------------------------------------------------------------
df = pd.read_csv(input_file)
print(f"\nloaded: {input_file}")
print(f"  rows x columns: {df.shape[0]} x {df.shape[1]}")
print(f"  years covered : {df['ay_start'].min()} to {df['ay_start'].max()}")


# -------------------------------------------------------------
# 1. summary statistics
# -------------------------------------------------------------
# this gives us min, max, average, etc. for each number column.
# .describe() is a built-in shortcut. .T flips the table so each
# column of the data becomes one row in the summary - easier to
# read when there are many columns.
numeric_cols = [
    'total_elementary', 'total_hs', 'total_shs', 'total_students',
    'prev_total_elementary', 'prev_total_hs',
    'prev_total_shs', 'prev_total_students',
]
summary = df[numeric_cols].describe().T.round(0)
summary.to_csv(f'{out_dir}/summary_statistics.csv')
print(f"\n[1/8] summary_statistics.csv  - basic stats per column")


# -------------------------------------------------------------
# 2. national enrollment trend (one line for the whole country)
# -------------------------------------------------------------
# we add up all the regions and sectors per year to get a
# single national total. then we draw a line chart over time.
#
# we also draw a vertical dashed line at year 2016 to mark
# when senior high school was introduced (the k-12 reform).
national = df.groupby('ay_start')['total_students'].sum().reset_index()

plt.figure(figsize=(9, 5))                 # set chart size
plt.plot(national['ay_start'], national['total_students'],
         marker='o', linewidth=2.2, color='#1f77b4')
plt.title('national total enrollment by school year (ph, k-12)',
          fontsize=13, fontweight='bold')
plt.xlabel('academic year start')
plt.ylabel('total students')
plt.ticklabel_format(style='plain', axis='y')   # no scientific notation
plt.axvline(2016, color='gray', linestyle='--', alpha=0.6)
plt.text(2016.05, national['total_students'].min(),
         '  k-12 / shs rollout', fontsize=9, color='gray')
plt.tight_layout()
plt.savefig(f'{out_dir}/enrollment_trend_national.png')
plt.close()
print(f"[2/8] enrollment_trend_national.png  - one line for the whole ph")


# -------------------------------------------------------------
# 3. enrollment by sector (public vs private vs suc-luc)
# -------------------------------------------------------------
# instead of one national line, we want three lines - one for
# each sector - so we can see how each sector grew over time.
sector_year = df.groupby(['ay_start', 'sector'])['total_students'].sum().reset_index()

plt.figure(figsize=(9, 5))
# loop through the three sectors and draw one line each.
# the colors are picked manually so each sector keeps the same
# color across all charts in the project.
for sector, color in zip(['Public', 'Private', 'SUC-LUC'],
                          ['#1f77b4', '#ff7f0e', '#2ca02c']):
    sub = sector_year[sector_year['sector'] == sector]
    plt.plot(sub['ay_start'], sub['total_students'],
             marker='o', linewidth=2, label=sector, color=color)
plt.title('total enrollment by sector', fontsize=13, fontweight='bold')
plt.xlabel('academic year start')
plt.ylabel('total students')
plt.ticklabel_format(style='plain', axis='y')
plt.legend(title='sector')
plt.tight_layout()
plt.savefig(f'{out_dir}/enrollment_by_sector.png')
plt.close()
print(f"[3/8] enrollment_by_sector.png  - public vs private vs suc-luc")


# -------------------------------------------------------------
# 4. enrollment by region (one line per region, 17 lines total)
# -------------------------------------------------------------
# this chart is busier because there are 17 regions, but it
# helps us see which regions are biggest (ncr, calabarzon)
# and which are smallest (small visayas/mindanao regions).
region_year = df.groupby(['ay_start', 'region'])['total_students'].sum().reset_index()

plt.figure(figsize=(11, 6.5))
palette = sns.color_palette('tab20', n_colors=df['region'].nunique())
for (region, sub), color in zip(region_year.groupby('region'), palette):
    plt.plot(sub['ay_start'], sub['total_students'],
             marker='o', linewidth=1.4, label=region, color=color)
plt.title('total enrollment by region', fontsize=13, fontweight='bold')
plt.xlabel('academic year start')
plt.ylabel('total students')
plt.ticklabel_format(style='plain', axis='y')
# the legend has 17 entries so we put it OUTSIDE the chart on the right
plt.legend(fontsize=7, loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
plt.savefig(f'{out_dir}/enrollment_by_region.png')
plt.close()
print(f"[4/8] enrollment_by_region.png  - all 17 regions over time")


# -------------------------------------------------------------
# 5. breakdown by education level (elementary, hs, shs)
# -------------------------------------------------------------
# this is a "stacked" chart: elementary on the bottom, hs on
# top of that, and shs on top of that. you see the total height
# (total students) and how each level contributes to it.
#
# you'll see the green (shs) layer suddenly appear in 2016 -
# that's when senior high school was introduced.
level_year = df.groupby('ay_start')[
    ['total_elementary', 'total_hs', 'total_shs']
].sum().reset_index()

plt.figure(figsize=(9, 5))
plt.stackplot(
    level_year['ay_start'],
    level_year['total_elementary'],
    level_year['total_hs'],
    level_year['total_shs'],
    labels=['elementary', 'high school', 'senior hs'],
    colors=['#4c72b0', '#dd8452', '#55a467'],
    alpha=0.85,
)
plt.title('enrollment by education level (stacked)',
          fontsize=13, fontweight='bold')
plt.xlabel('academic year start')
plt.ylabel('total students')
plt.ticklabel_format(style='plain', axis='y')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(f'{out_dir}/level_breakdown.png')
plt.close()
print(f"[5/8] level_breakdown.png  - elementary / hs / shs stacked")


# -------------------------------------------------------------
# 6. correlation heatmap (which numbers move together?)
# -------------------------------------------------------------
# correlation is a number between -1 and +1.
#   +1 means "when one goes up, the other always goes up"
#    0 means "no relationship at all"
#   -1 means "when one goes up, the other always goes down"
#
# the heatmap colors each cell by the correlation. red means
# strong positive, blue means strong negative.
#
# this is the chart the project rubric specifically asks for.
corr_cols = [
    'ay_start', 'shs_available',
    'total_elementary', 'total_hs', 'total_shs', 'total_students',
    'prev_total_elementary', 'prev_total_hs',
    'prev_total_shs', 'prev_total_students',
]
corr = df[corr_cols].corr().round(3)

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True,
            cbar_kws={'label': 'pearson correlation'})
plt.title('correlation heatmap - numeric features',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{out_dir}/correlation_heatmap.png')
plt.close()
# also save the raw numbers as a csv in case writers want to
# quote specific values in the paper.
corr.to_csv(f'{out_dir}/correlation_table.csv')
print(f"[6/8] correlation_heatmap.png + correlation_table.csv")


# -------------------------------------------------------------
# 7. distribution of the target column (total_students)
# -------------------------------------------------------------
# left chart: a histogram showing how many region+sector+year
#   combos fall into each "bucket" of student counts.
# right chart: a boxplot showing the typical spread and any
#   extreme values (outliers) like ncr.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].hist(df['total_students'], bins=30, color='#4c72b0', edgecolor='white')
axes[0].set_title('distribution of total_students')
axes[0].set_xlabel('total students (per region+sector+year)')
axes[0].set_ylabel('frequency')
axes[0].ticklabel_format(style='plain', axis='x')

axes[1].boxplot(df['total_students'], vert=False)
axes[1].set_title('boxplot of total_students')
axes[1].set_xlabel('total students')
axes[1].ticklabel_format(style='plain', axis='x')

plt.tight_layout()
plt.savefig(f'{out_dir}/distribution_total_students.png')
plt.close()
print(f"[7/8] distribution_total_students.png  - histogram + boxplot")


# -------------------------------------------------------------
# 8. plain-language summary file
# -------------------------------------------------------------
# this section calculates a few headline numbers (national
# growth, sector share, top regions, strongest correlations)
# and writes them to a text file the writers can paraphrase
# directly into the manuscript.

# top 4 features most correlated with total_students.
# we drop the self-correlation (total_students vs itself = 1.0)
# and rank by absolute value because both strong positive and
# strong negative correlations are "interesting".
top_corrs = (
    corr['total_students']
    .drop('total_students')
    .abs().sort_values(ascending=False).head(4)
)
top_corr_lines = "\n".join(
    f"     - {feat}: r = {corr['total_students'][feat]:.3f}"
    for feat in top_corrs.index
)

# national totals at the start vs end of our window
national_2010 = int(national.loc[national['ay_start'] == 2010, 'total_students'].iloc[0])
national_2020 = int(national.loc[national['ay_start'] == 2020, 'total_students'].iloc[0])
growth_pct = (national_2020 - national_2010) / national_2010 * 100

# sector share = each sector's total divided by overall total
sector_share = (
    df.groupby('sector')['total_students'].sum()
    / df['total_students'].sum() * 100
).round(1).sort_values(ascending=False)
sector_lines = "\n".join(
    f"     - {sector}: {share:.1f}%" for sector, share in sector_share.items()
)

# top 3 regions by cumulative enrollment across all years
top_regions = (
    df.groupby('region')['total_students'].sum()
    .sort_values(ascending=False).head(3)
)
top_region_lines = "\n".join(
    f"     - {region}: {int(total):,} total students (cumulative)"
    for region, total in top_regions.items()
)

summary_text = f"""\
================================================================
  eda plain-language summary  (paraphrase into manuscript 7b)
================================================================

dataset shape
  - rows   : {df.shape[0]} (region+sector+year combinations)
  - years  : {df['ay_start'].min()} to {df['ay_start'].max()}
  - regions: {df['region'].nunique()}
  - sectors: {', '.join(sorted(df['sector'].unique()))}

national trend
  - total enrollment in {df['ay_start'].min()}-{df['ay_start'].min()+1}: {national_2010:,} students
  - total enrollment in {df['ay_start'].max()}-{df['ay_start'].max()+1}: {national_2020:,} students
  - net change over the period: {growth_pct:+.1f}%
  - the 2020 figure shows a visible dip vs. 2019 (covid-19
    school-year disruption); this anomaly will affect any model
    trained on years that include 2020.

structural break
  - senior high school (shs) enrollment is zero before 2016
    because the k-12 reform introduced grades 11-12 starting
    sy 2016-2017. the shs_available flag captures this break.

sector share (across all years)
{sector_lines}
  - public sector dominates total enrollment; private and
    suc-luc are an order of magnitude smaller.

top 3 regions by cumulative enrollment
{top_region_lines}
  - these large urban regions appear as outliers in the
    distribution but are legitimate data, not errors.

strongest correlations with total_students
{top_corr_lines}
  - prev_total_students has near-perfect correlation (~0.99),
    indicating strong year-over-year persistence. this is why
    lag features are the core predictors in the modeling stage.

key takeaways for modeling
  1. use lag (prev_*) features as primary predictors.
  2. include shs_available flag so models can absorb the
     2016 k-12 structural break cleanly.
  3. treat 2020 as an anomalous year - keep it in training
     but flag it as a known source of forecast error.
  4. model per region+sector context; absolute scale varies
     hugely between ncr and smaller regions.

next stage (semma - model): enrollment_modeling.py
================================================================
"""

# write the summary to a text file
with open(f'{out_dir}/eda_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary_text)
print(f"[8/8] eda_summary.txt  - plain-language summary for writers")


# -------------------------------------------------------------
# done
# -------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  eda complete - all outputs in '{out_dir}/'")
print(f"{'='*60}")
print(f"\nmanuscript section 7b (eda) and the presentation's")
print(f"methodology slide can pull directly from this folder.")
print(f"\nnext: enrollment_modeling.py  (semma - model)")
