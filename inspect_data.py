
from pathlib import Path
import pandas as pd

file_path = Path("FAO,DF_SDG_2_1_2,1.0+all.csv")

if not file_path.exists():
    raise FileNotFoundError(f"File not found: {file_path}")

df = pd.read_csv(file_path, low_memory=False)

print("\n--- DATASET SIZE ---")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- FIRST 5 ROWS ---")
print(df.head().to_string())

print("\n--- DATA TYPES ---")
print(df.dtypes.to_string())

print("\n--- MISSING VALUES ---")
print(df.isna().sum().to_string())

print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())


print("\n--- SERIES COUNTS ---")
print(df["SERIES"].value_counts(dropna=False).to_string())

print("\n--- INDICATOR COUNTS ---")
print(df["INDICATOR"].value_counts(dropna=False).to_string())

print("\n--- OBSERVATION STATUS ---")
print(df["OBS_STATUS"].value_counts(dropna=False).to_string())

print("\n--- SAMPLE WITH VALUES ---")
print(
    df.loc[
        df["OBS_VALUE"].notna(),
        [
            "REF_AREA_ISO3",
            "SERIES",
            "TIME_PERIOD",
            "OBS_VALUE",
            "OBS_STATUS",
            "UNIT_MEASURE",
        ],
    ].head(15).to_string(index=False)
)
