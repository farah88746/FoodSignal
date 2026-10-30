from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/processed/food_insecurity_percentages.csv")

def validate(df):
    required = {"country_code", "year", "food_insecurity_percent"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df[["country_code", "year"]].isna().any().any():
        raise ValueError("Country code and year must not be missing")
    if df.duplicated(["country_code", "year"]).any():
        raise ValueError("Duplicate country-year key")
    values = pd.to_numeric(df["food_insecurity_percent"], errors="coerce")
    if values.isna().any():
        raise ValueError("Percentage must be numeric and non-missing")
    if not values.between(0, 100).all():
        raise ValueError("Percentage must be between 0 and 100")

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {DATA_PATH}. Run this script from the project root."
        )
    df = pd.read_csv(DATA_PATH)
    validate(df)
    print("DATA VALIDATION: PASS")
    print(f"Rows: {len(df):,}")
    print(f"Countries/areas: {df['country_code'].nunique()}")
    print(f"Reference years: {int(df['year'].min())}–{int(df['year'].max())}")
    print(f"Duplicate country-year keys: {int(df.duplicated(['country_code','year']).sum())}")
    print(
        "Percentage range: "
        f"{pd.to_numeric(df['food_insecurity_percent']).min():.1f}%–"
        f"{pd.to_numeric(df['food_insecurity_percent']).max():.1f}%"
    )

if __name__ == "__main__":
    main()
