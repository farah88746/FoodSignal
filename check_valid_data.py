
from pathlib import Path
import pandas as pd

file_path = Path("FAO,DF_SDG_2_1_2,1.0+all.csv")

df = pd.read_csv(file_path, low_memory=False)

print("ORIGINAL COLUMN NAMES:")
print(df.columns.tolist())

df["OBS_VALUE_NUMERIC"] = pd.to_numeric(
    df["OBS_VALUE"], errors="coerce"
)

subset = df[
    (df["REF_AREA_ISO3"] == "AFG") &
    (pd.to_numeric(df["TIME_PERIOD"], errors="coerce") == 2015) &
    (df["SERIES"] == "AG_PRD_FIESMS")
]

print("\nAFGHANISTAN 2015 - ORIGINAL RECORDS:")
print("Records found:", len(subset))

# Display only columns that differ across these records.
varying_columns = [
    col for col in subset.columns
    if subset[col].nunique(dropna=False) > 1
]

print("\nColumns that vary:")
print(varying_columns)

print("\nRelevant original values:")
preferred = [
    "REF_AREA", "REF_AREA_ISO2", "REF_AREA_ISO3",
    "TIME_PERIOD", "TIME_DETAIL", "SERIES",
    "OBS_VALUE", "OBS_STATUS", "UNIT_MEASURE",
    "SOURCE_DETAIL", "COMMENT_OBS", "SEX", "AGE"
]

show = [col for col in preferred if col in subset.columns]
print(subset[show].to_string(index=False))
