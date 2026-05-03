# =============================================================
# what this file does:
#
# this script takes the raw enrollment data we exported from
# excel and cleans it up so the rest of the pipeline can use it.
#
# think of it like washing vegetables before cooking. the raw
# data has small problems (extra title row, weird column names,
# numbers stored with commas like "1,008,026") and this script
# fixes all of them in one go.
#
# what comes IN  : ml_ready_dataset.csv  (the raw export)
# what comes OUT : enrollment_cleaned.csv (cleaned up version)
#
# we also add one extra column called "last year's number"
# (technically called a lag feature) so the predictors later on
# have something useful to learn from.
# =============================================================

import pandas as pd

# the file we are reading from and the file we will write to
raw_file   = 'ML_Ready_Dataset.csv'
clean_file = 'enrollment_cleaned.csv'


# -------------------------------------------------------------
# step zero: load the raw file
# -------------------------------------------------------------
# the raw file has a problem: row 1 is a title banner, not
# real data. so we tell pandas to skip the first row.
#
# also, numbers in the file are written like "1,008,026" with
# commas. we tell pandas that comma means "thousands separator"
# so it reads them as actual numbers, not text.
df = pd.read_csv(raw_file, skiprows=1, thousands=',')

print("=" * 60)
print("  ph enrollment forecasting - data cleaning report")
print("=" * 60)
print(f"\nloaded the raw export")
print(f"  rows x columns  : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  original headers: {list(df.columns)}")


# -------------------------------------------------------------
# step 1: rename the columns to short, friendly names
# -------------------------------------------------------------
# the original column names had line breaks inside them
# (things like "ay start\n(feature)") which is annoying to type
# and looks bad in code. so we replace them with short names
# that use only lowercase letters and underscores.
df.columns = [
    'ay_start',          # year the school year starts (e.g. 2020 means sy 2020-2021)
    'ay_end',            # year the school year ends
    'sector',            # public, private, or suc-luc (state/local universities)
    'sector_encoded',    # same as sector but as a number (0, 1, or 2)
    'region',            # the region name (e.g. ncr, calabarzon)
    'region_encoded',    # same as region but as a number (0 to 16)
    'shs_available',     # 0 if before 2016 (no senior high yet), 1 if 2016 onwards
    'total_elementary',  # total students in elementary
    'total_hs',          # total students in junior high school
    'total_shs',         # total students in senior high school
    'total_students',    # grand total (elementary + hs + shs)
]

print(f"\nstep 1: renamed columns to short, friendly names")
print(f"  new headers: {list(df.columns)}")


# -------------------------------------------------------------
# step 2: check for missing values
# -------------------------------------------------------------
# missing values are blank cells in the data. they cause errors
# in calculations later, so we count them and decide what to do.
missing = df.isnull().sum().sum()
print(f"\nstep 2: check for missing values")
print(f"  total blanks: {missing}")
if missing > 0:
    # if there are missing cells, we drop those rows (safer than guessing).
    print("  blanks per column:")
    print(df.isnull().sum()[df.isnull().sum() > 0].to_string())
    df = df.dropna()
    print(f"  action: dropped rows with blanks. {len(df)} rows remain")
else:
    print("  action: nothing to do, the data is complete")


# -------------------------------------------------------------
# step 3: remove duplicate rows
# -------------------------------------------------------------
# a duplicate is the exact same row appearing twice. duplicates
# would make some regions count double, which would be wrong.
dups = df.duplicated().sum()
before = len(df)
df = df.drop_duplicates()
print(f"\nstep 3: remove duplicate rows")
print(f"  duplicates found: {dups}")
print(f"  rows before: {before} | rows after: {len(df)}")


# -------------------------------------------------------------
# step 4: clean up the text columns
# -------------------------------------------------------------
# sometimes text fields have extra spaces (like "  public ")
# which makes "public" and " public" look different to the
# computer even though they mean the same thing. .str.strip()
# removes those leading and trailing spaces.
df['sector'] = df['sector'].str.strip()
df['region'] = df['region'].str.strip()

print(f"\nstep 4: cleaned up the text columns")
print(f"  sectors found  : {sorted(df['sector'].unique())}")
print(f"  regions counted: {df['region'].nunique()} unique regions")


# -------------------------------------------------------------
# step 5: make sure number columns are stored as integers
# -------------------------------------------------------------
# student counts should be whole numbers (you cannot have
# 1234.5 students). we force these columns to be integers so
# nothing accidentally becomes a decimal later.
int_cols = [
    'ay_start', 'ay_end',
    'sector_encoded', 'region_encoded', 'shs_available',
    'total_elementary', 'total_hs', 'total_shs', 'total_students',
]
for c in int_cols:
    df[c] = df[c].astype(int)

print(f"\nstep 5: ensured number columns are whole numbers")
print(f"  columns affected: {int_cols}")


# -------------------------------------------------------------
# step 6: drop any row where total students is zero
# -------------------------------------------------------------
# if a region+sector+year has zero total students, that row
# carries no useful information. it could also be a data
# entry error. we remove these rows just to be safe.
zero_rows = df[df['total_students'] == 0]
if len(zero_rows):
    print(f"\nstep 6: remove rows where total students is zero")
    print(zero_rows.to_string())
    df = df[df['total_students'] > 0].copy()
    print(f"  removed {len(zero_rows)} row(s)")
else:
    print(f"\nstep 6: no zero-total rows found, nothing to remove")


# -------------------------------------------------------------
# step 7: add "last year's number" columns (the most important step)
# -------------------------------------------------------------
# to predict next year's enrollment, the most useful clue is
# what last year looked like. so for every row, we want to add
# columns that say "this is what enrollment was last year for
# this same region+sector".
#
# how it works:
#   1. sort the data so that for each region+sector, the years
#      are in order (2010, 2011, 2012, ...)
#   2. for each region+sector group, "shift" the values down
#      by one row. the row for 2011 then has access to the
#      2010 values via the new prev_* columns.
#
# this gives the predictors a memory of last year's data.
df = df.sort_values(['region', 'sector', 'ay_start']).reset_index(drop=True)

for col in ['total_elementary', 'total_hs', 'total_shs', 'total_students']:
    # groupby ensures we only look at the same region+sector when shifting,
    # so ncr public's 2011 row gets ncr public's 2010 numbers, not some
    # other region's leftover values.
    df[f'prev_{col}'] = df.groupby(['region', 'sector'])[col].shift(1)

# the very first year (2010-2011) for each region+sector cannot
# have a "last year" because there is no earlier year in the
# dataset. those cells will be blank for now.
nulls_introduced = df['prev_total_students'].isnull().sum()
print(f"\nstep 7: added last-year columns (the key for prediction)")
print(f"  sorted by region then sector then year")
print(f"  added: prev_total_elementary, prev_total_hs, prev_total_shs, prev_total_students")
print(f"  blanks introduced for the very first year: {nulls_introduced}")
print(f"  (these blanks are expected and the modeling step will skip those rows)")


# -------------------------------------------------------------
# step 8: save the cleaned file and print a final summary
# -------------------------------------------------------------
df = df.reset_index(drop=True)

print(f"\n{'='*60}")
print(f"  final cleaned dataset summary")
print(f"{'='*60}")
print(f"  rows x columns : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  years covered  : {df['ay_start'].min()} to {df['ay_start'].max()}")
print(f"  regions        : {df['region'].nunique()}")
print(f"  sectors        : {df['sector'].nunique()}")
print(f"  remaining blanks: {df.isnull().sum().sum()} "
      f"(only in the prev_* columns for the first year - this is fine)")
print(f"\n  final columns:")
for c in df.columns:
    print(f"    - {c}")

# write the cleaned data to a new csv so the next scripts can use it.
df.to_csv(clean_file, index=False)
print(f"\nsaved: {clean_file}")
print(f"{'='*60}")
