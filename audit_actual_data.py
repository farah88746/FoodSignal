from pathlib import Path
import sys
import pandas as pd

# Run from project root. Uses project's existing data layer to preserve app logic.
from foodshield.data import load_observations, coverage_summary, observations_for_year, country_change

DATA_PATH = Path("data/processed/food_insecurity_percentages.csv")
OUT = Path("reports")
REFERENCE_YEAR = 2024
MIN_YEARS_FOR_DIRECTION = 4
STALE_AFTER_YEARS = 1

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cannot find {DATA_PATH}; run this script from project root.")
    OUT.mkdir(exist_ok=True)
    loaded = load_observations(DATA_PATH)
    df = loaded.data
    if df.empty:
        raise ValueError("No valid observations found.")
    latest_dataset_year = int(df["year"].max())
    ref_year = min(REFERENCE_YEAR, latest_dataset_year)

    coverage = coverage_summary(df)
    coverage["years_since_latest_observation"] = latest_dataset_year - coverage["latest_year"]
    coverage["data_freshness_flag"] = coverage["years_since_latest_observation"].apply(
        lambda n: "Current to dataset end" if n == 0 else ("1 year behind" if n == 1 else "Stale: review needed")
    )
    coverage["history_adequacy_flag"] = coverage["n_years_observed"].apply(
        lambda n: "At least 4 observed years" if n >= MIN_YEARS_FOR_DIRECTION else "Limited history: interpret trend cautiously"
    )
    coverage["reference_year_comparable"] = coverage["latest_year"] == ref_year
    coverage["review_reasons"] = coverage.apply(
        lambda r: "; ".join(
            reason for condition, reason in [
                (r["latest_year"] < ref_year, f"No observation for reference year {ref_year}"),
                (r["n_years_observed"] < MIN_YEARS_FOR_DIRECTION, "Fewer than 4 observed years"),
                (bool(r["latest_year_conflict"]), "Latest observation has conflicting values"),
                (int(r["n_conflict_years"]) > 0, "One or more conflicting years in history"),
            ] if condition
        ) or "No automatic coverage flag",
        axis=1,
    )
    coverage.to_csv(OUT/"actual_country_coverage_audit.csv", index=False)

    comparable, withheld = observations_for_year(df, ref_year)
    peer = comparable[["country_code","country_name","year","window_label","food_insecurity_percent","source","observation_status"]].copy()
    if not peer.empty:
        peer["relative_burden_rank"] = peer["food_insecurity_percent"].rank(method="min", ascending=False).astype(int)
        peer["higher_burden_quartile"] = peer["food_insecurity_percent"] >= peer["food_insecurity_percent"].quantile(.75)
        peer = peer.sort_values(["food_insecurity_percent","country_name"], ascending=[False,True])
    peer.to_csv(OUT/"reference_year_comparable_peer_audit.csv", index=False)

    # Assess trend output and gaps, without treating first-to-last as a forecast.
    trend_rows = []
    for code, group in df.groupby("country_code"):
        years = sorted(int(y) for y in group.loc[~group["conflict"], "year"].unique())
        result = country_change(group)
        if result is None:
            trend_rows.append({
                "country_code": code,
                "country_name": group["country_name"].iloc[0],
                "first_unambiguous_year": years[0] if years else None,
                "last_unambiguous_year": years[-1] if years else None,
                "unambiguous_years": len(years),
                "span_years": (years[-1]-years[0]+1) if len(years)>=2 else None,
                "change_pp": None,
                "trend_flag": "Insufficient unambiguous years",
                "review_note": "Do not assign a direction",
            })
        else:
            start_y, end_y = int(result["start_year"]), int(result["end_year"])
            n_years = len(years)
            span = end_y-start_y+1
            gap_count = span - n_years
            change = float(result["change_pp"])
            trend_rows.append({
                "country_code": code,
                "country_name": group["country_name"].iloc[0],
                "first_unambiguous_year": start_y,
                "last_unambiguous_year": end_y,
                "unambiguous_years": n_years,
                "span_years": span,
                "change_pp": change,
                "trend_flag": (
                    "Limited history" if n_years < MIN_YEARS_FOR_DIRECTION
                    else "Large gaps: review" if gap_count >= 3
                    else "Descriptive increase" if change > 0
                    else "Descriptive decrease" if change < 0
                    else "No net change"
                ),
                "review_note": (
                    "First-to-last change is historical only; do not interpret as forecast."
                    if gap_count < 3 else
                    f"{gap_count} missing years inside span; interpret first-to-last change cautiously."
                ),
            })
    trends = pd.DataFrame(trend_rows).sort_values(["trend_flag","country_name"])
    trends.to_csv(OUT/"actual_historical_signal_audit.csv", index=False)

    conflict_rows = df[df["conflict"]].copy()
    conflict_rows.to_csv(OUT/"conflicting_observations_audit.csv", index=False)

    summary = [
        ("valid_observation_rows", len(df)),
        ("countries_areas", df["country_code"].nunique()),
        ("dataset_year_start", int(df["year"].min())),
        ("dataset_year_end", latest_dataset_year),
        ("reference_year_audited", ref_year),
        ("countries_with_reference_year_estimate", comparable["country_code"].nunique()),
        ("countries_missing_reference_year", coverage.loc[coverage["latest_year"] < ref_year, "country_code"].nunique()),
        ("countries_with_less_than_4_observed_years", int((coverage["n_years_observed"] < MIN_YEARS_FOR_DIRECTION).sum())),
        ("countries_with_any_conflicting_year", int((coverage["n_conflict_years"] > 0).sum())),
        ("conflicting_rows_withheld_in_reference_year", len(withheld)),
        ("countries_with_large_historical_gaps_flagged", int(trends["trend_flag"].eq("Large gaps: review").sum())),
    ]
    pd.DataFrame(summary, columns=["metric","value"]).to_csv(OUT/"actual_data_audit_summary.csv", index=False)
    print("ACTUAL DATA AUDIT: COMPLETE")
    for k, v in summary:
        print(f"{k}: {v}")
    print("Reports written to reports/:")
    for name in [
        "actual_data_audit_summary.csv",
        "actual_country_coverage_audit.csv",
        "reference_year_comparable_peer_audit.csv",
        "actual_historical_signal_audit.csv",
        "conflicting_observations_audit.csv",
    ]:
        print(f" - {name}")

if __name__ == "__main__":
    main()
