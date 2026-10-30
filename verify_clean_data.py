
from pathlib import Path
import pandas as pd

path = Path("data/processed/food_insecurity_percentages.csv")
df = pd.read_csv(path)

print("DATASET CHECK")
print("=" * 40)
print("Rows:", len(df))
print("Unique countries:", df["country_code"].nunique())
print("Year range:", df["year"].min(), "-", df["year"].max())
print("Missing values:")
print(df[["country_code", "year", "food_insecurity_percent"]].isna().sum())

print("\nDuplicate country-year groups:")
duplicates = df.duplicated(
    subset=["country_code", "year"], keep=False
)
print(df.loc[duplicates].sort_values(
    ["country_code", "year"]
).head(20).to_string(index=False))

print("\nAfghanistan records:")
print(
    df[df["country_code"] == "AFG"][
        ["country_code", "year", "food_insecurity_percent", "time_detail"]
    ].to_string(index=False)
)

print("\nSaved dataset verification complete.")
