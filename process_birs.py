"""
Download BIRS data from MNB, flatten all sheets into a single table,
and export to xlsx and csv.
"""

import urllib.request
import pandas as pd
import os
import sys
import io

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

URL = "https://www.mnb.hu/letoltes/birs.xls"
RAW_FILE = os.path.join("data", "birs.xls")
OUTPUT_XLSX = os.path.join("data", "birs_flat.xlsx")
OUTPUT_CSV = os.path.join("data", "birs_flat.csv")

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# 1. Download the file
print(f"Downloading {URL} ...")
urllib.request.urlretrieve(URL, RAW_FILE)
print(f"Saved raw file: {RAW_FILE} ({os.path.getsize(RAW_FILE):,} bytes)")

# 2. Read all sheets
xls = pd.ExcelFile(RAW_FILE)
sheet_names = xls.sheet_names
print(f"\nFound {len(sheet_names)} sheet(s): {sheet_names}")

# Standard column mapping: normalize Hungarian tenor names to English
COL_MAP = {}
for n in range(1, 31):
    COL_MAP[f'{n} év'] = f'{n} years'
    COL_MAP[f'{n} év '] = f'{n} years'
COL_MAP['dátum'] = 'date of fixing'

# 3. Process year-data sheets only (skip error list)
all_data = []

for name in sheet_names:
    # Skip the error list sheet
    if 'hiba' in name.lower() or 'error' in name.lower():
        print(f"\n  Skipping error list sheet: '{name}'")
        continue

    df = pd.read_excel(xls, sheet_name=name, header=None)
    print(f"\n--- Sheet: '{name}' (raw shape: {df.shape}) ---")

    # Find the header row: look for tenor labels (év / year)
    header_row = 0
    for i in range(min(10, len(df))):
        row_vals = [str(v).strip().lower() for v in df.iloc[i].values if pd.notna(v)]
        if any('tum' in v or 'date' in v or 'fixing' in v or 'év' in v or 'year' in v for v in row_vals):
            header_row = i
            break

    # Build header names; NaT/NaN header cells become '_empty_N'
    raw_headers = df.iloc[header_row].values
    headers = []
    for j, v in enumerate(raw_headers):
        if pd.notna(v):
            headers.append(str(v).strip().lower())
        else:
            headers.append(f'_empty_{j}')

    data = df.iloc[header_row + 1:].copy()
    data.columns = headers

    # Drop fully empty rows
    data = data.dropna(how='all')

    # Rename columns using the mapping
    renamed = {}
    for col in data.columns:
        col_clean = col.strip().lower()
        for pattern, target in COL_MAP.items():
            if col_clean == pattern.strip().lower():
                renamed[col] = target
                break
        if col not in renamed:
            if 'tum' in col_clean or 'date' in col_clean or 'fixing' in col_clean:
                renamed[col] = 'date of fixing'

    data = data.rename(columns=renamed)

    # If we still don't have 'date of fixing', look for:
    # 1) An _empty_ column whose values look like dates, OR
    # 2) The raw first column of the original dataframe
    if 'date of fixing' not in data.columns:
        # Check _empty_ columns for date-like values
        found_date_col = False
        for col in [c for c in data.columns if c.startswith('_empty_')]:
            sample = data[col].dropna().head(5)
            if len(sample) > 0 and all(isinstance(v, pd.Timestamp) for v in sample):
                data = data.rename(columns={col: 'date of fixing'})
                found_date_col = True
                break
        if not found_date_col:
            # Last resort: rename first column
            first_col = data.columns[0]
            data = data.rename(columns={first_col: 'date of fixing'})

    # Now drop remaining _empty_ columns
    data = data[[c for c in data.columns if not c.startswith('_empty_')]]
    data = data.dropna(axis=1, how='all')

    # Coerce the date column to datetime
    data['date of fixing'] = pd.to_datetime(data['date of fixing'], errors='coerce')

    # Drop rows where date is NaT
    data = data.dropna(subset=['date of fixing'])

    # Add source sheet column
    data['source_sheet'] = name

    print(f"  Header row: {header_row}")
    print(f"  Data shape: {data.shape}")
    print(f"  Columns: {list(data.columns)}")
    if len(data) > 0:
        print(f"  Date range: {data['date of fixing'].min()} to {data['date of fixing'].max()}")

    all_data.append(data)

# 4. Combine all sheets
if not all_data:
    print("\nERROR: No data extracted from any sheet!")
    sys.exit(1)

combined = pd.concat(all_data, ignore_index=True)

# Sort by date
combined = combined.sort_values('date of fixing').reset_index(drop=True)

# Reorder columns: date first, then tenors in order, then source_sheet
tenor_cols = [c for c in combined.columns if 'years' in c.lower()]
def tenor_sort_key(col_name):
    try:
        return int(col_name.split()[0])
    except:
        return 999
tenor_cols.sort(key=tenor_sort_key)

# Coerce tenor columns to numeric to avoid object dtype issues
for col in tenor_cols:
    combined[col] = pd.to_numeric(combined[col], errors='coerce')

col_order = ['date of fixing'] + tenor_cols + ['source_sheet']
col_order = [c for c in col_order if c in combined.columns]
combined = combined[col_order]

print(f"\n{'='*60}")
print(f"COMBINED TABLE")
print(f"{'='*60}")
print(f"Shape: {combined.shape}")
print(f"Columns: {list(combined.columns)}")
print(f"Date range: {combined['date of fixing'].min()} to {combined['date of fixing'].max()}")
print(f"\nFirst 5 rows:")
print(combined.head().to_string())
print(f"\nLast 5 rows:")
print(combined.tail().to_string())

# 5. Export
combined.to_excel(OUTPUT_XLSX, index=False, engine='openpyxl')
print(f"\nExported to {OUTPUT_XLSX} ({os.path.getsize(OUTPUT_XLSX):,} bytes)")

combined.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
print(f"Exported to {OUTPUT_CSV} ({os.path.getsize(OUTPUT_CSV):,} bytes)")

OUTPUT_PKL = os.path.join("data", "birs_flat.pkl")
combined.to_pickle(OUTPUT_PKL)
print(f"Exported to {OUTPUT_PKL} ({os.path.getsize(OUTPUT_PKL):,} bytes)")

print("\nDone!")
