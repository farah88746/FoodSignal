"""Auditable cleaning pipeline for FoodShield AI.

Run from any directory:
    python clean_data.py

The pipeline preserves valid, non-identical observations and their source/period
metadata. Exact duplicates are detected against the full raw row. Conflicting
country-year values are flagged for review but are not averaged or silently
discarded from the dashboard CSV; the dashboard must exclude them from rankings.
"""
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SOURCE_FILE = BASE_DIR / "FAO,DF_SDG_2_1_2,1.0+all.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_FILE = OUTPUT_DIR / "food_insecurity_percentages.csv"
REVIEW_FILE = OUTPUT_DIR / "records_for_review.csv"
INVALID_FILE = OUTPUT_DIR / "invalid_records.csv"
DUPLICATE_FILE = OUTPUT_DIR / "duplicate_records_removed.csv"
SUMMARY_FILE = OUTPUT_DIR / "cleaning_summary.csv"
SCOPE_FILE = OUTPUT_DIR / "out_of_scope_records.csv"
AUDIT_FILE = OUTPUT_DIR / "row_level_audit.csv"

TARGET_SERIES = "AG_PRD_FIESMS"
REQUIRED = [
    "REF_AREA_ISO3", "TIME_PERIOD", "SERIES", "OBS_VALUE", "SEX", "AGE",
    "OBS_STATUS", "UNIT_MEASURE", "TIME_DETAIL", "SOURCE_DETAIL", "INDICATOR",
]


def normalize_text(series):
    """Trim strings while preserving missing values."""
    return series.astype("string").str.strip()


def write_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Raw FAO export not found at {SOURCE_FILE}. "
            "Place it beside clean_data.py and rerun."
        )

    raw = pd.read_csv(SOURCE_FILE, low_memory=False)
    raw.columns = raw.columns.str.strip()
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"Required FAO columns missing: {missing}")

    raw = raw.copy()
    raw.insert(0, "source_row_number", range(2, len(raw) + 2))  # CSV line, including header
    raw["SERIES"] = normalize_text(raw["SERIES"])
    raw["SEX"] = normalize_text(raw["SEX"])
    raw["AGE"] = normalize_text(raw["AGE"])

    # Mutually exclusive dispositions: every input row receives exactly one.
    disposition = pd.Series("retained_candidate", index=raw.index, dtype="string")
    outside_series = ~raw["SERIES"].eq(TARGET_SERIES)
    disposition.loc[outside_series] = "outside_target_series"

    in_series = ~outside_series
    not_total = in_series & ~(raw["SEX"].eq("_T") & raw["AGE"].eq("_T"))
    disposition.loc[not_total] = "not_total_population"

    target_total = in_series & ~not_total
    iso = normalize_text(raw["REF_AREA_ISO3"]).str.upper()
    year = pd.to_numeric(raw["TIME_PERIOD"], errors="coerce")
    value = pd.to_numeric(raw["OBS_VALUE"], errors="coerce")

    valid_country = iso.str.fullmatch(r"[A-Z]{3}", na=False)
    valid_year = year.notna() & year.between(1900, 2100) & (year % 1 == 0)
    valid_value = value.notna() & value.between(0, 100)
    valid = target_total & valid_country & valid_year & valid_value
    invalid_mask = target_total & ~valid
    disposition.loc[invalid_mask] = "invalid_target_total"

    invalid = raw.loc[invalid_mask].copy()
    invalid["validation_reason"] = ""
    reason_specs = [
        (~valid_country, "missing_or_invalid_iso3"),
        (~valid_year, "missing_or_invalid_year"),
        (~valid_value, "missing_or_out_of_range_percentage"),
    ]
    for mask, reason in reason_specs:
        affected = invalid.index[mask.reindex(invalid.index, fill_value=False)]
        for idx in affected:
            existing = invalid.at[idx, "validation_reason"]
            invalid.at[idx, "validation_reason"] = (
                f"{existing}; {reason}" if existing else reason
            )

    candidates = raw.loc[valid].copy()
    candidates["country_code"] = iso.loc[valid]
    candidates["year"] = year.loc[valid].astype(int)
    candidates["food_insecurity_percent"] = value.loc[valid].astype(float)

    # Exact duplicates are identical across ALL raw columns, not merely the
    # reduced dashboard schema. Keep first; log every removed row and its line.
    raw_columns = [c for c in raw.columns if c != "source_row_number"]
    exact_dup_mask = candidates.duplicated(subset=raw_columns, keep="first")
    duplicates = candidates.loc[exact_dup_mask].copy()
    duplicates["duplicate_of_source_row_number"] = (
        candidates.loc[~exact_dup_mask]
        .drop_duplicates(subset=raw_columns, keep="first")
        .set_index(raw_columns)["source_row_number"]
        .reindex(pd.MultiIndex.from_frame(duplicates[raw_columns]))
        .to_numpy()
        if not duplicates.empty else []
    )
    disposition.loc[duplicates.index] = "exact_duplicate_removed"
    retained = candidates.loc[~exact_dup_mask].copy()

    # Keep all non-identical valid rows. Do not collapse same-value metadata
    # repeats or conflicts. These flags make the audit trail explicit.
    key = ["country_code", "year"]
    value_nunique = retained.groupby(key)["food_insecurity_percent"].transform("nunique")
    retained["record_check"] = "single_observation"
    retained.loc[value_nunique.gt(1), "record_check"] = "conflicting_country_year"
    same_value_repeat = retained.duplicated(
        subset=["country_code", "year", "food_insecurity_percent"], keep=False
    ) & value_nunique.eq(1)
    retained.loc[same_value_repeat, "record_check"] = "same_value_metadata_repeat"

    metadata = {
        "country_code": "country_code",
        "year": "year",
        "food_insecurity_percent": "food_insecurity_percent",
        "observation_status": "OBS_STATUS",
        "unit_measure": "UNIT_MEASURE",
        "time_detail": "TIME_DETAIL",
        "source": "SOURCE_DETAIL",
        "indicator": "INDICATOR",
    }
    dashboard = retained[
        ["source_row_number"] + list(metadata.values()) + ["country_code", "year", "food_insecurity_percent", "record_check"]
    ].copy()
    # Build clean schema without duplicate columns and preserve row provenance.
    dashboard = pd.DataFrame({
        "source_row_number": retained["source_row_number"],
        "country_code": retained["country_code"],
        "year": retained["year"],
        "food_insecurity_percent": retained["food_insecurity_percent"],
        "observation_status": retained["OBS_STATUS"],
        "unit_measure": retained["UNIT_MEASURE"],
        "time_detail": retained["TIME_DETAIL"],
        "source": retained["SOURCE_DETAIL"],
        "indicator": retained["INDICATOR"],
        "record_check": retained["record_check"],
    }).sort_values(["country_code", "year", "source_row_number"])

    conflict_rows = retained.loc[retained["record_check"].eq("conflicting_country_year")].copy()
    conflict_rows["conflict_group_id"] = (
        conflict_rows["country_code"].astype(str) + "-" + conflict_rows["year"].astype(str)
    )
    repeat_rows = retained.loc[retained["record_check"].eq("same_value_metadata_repeat")].copy()

    # All rows get a disposition in the row-level audit.
    audit = raw[["source_row_number", "SERIES", "REF_AREA_ISO3", "TIME_PERIOD", "OBS_VALUE", "SEX", "AGE"]].copy()
    audit["disposition"] = disposition
    # Valid kept rows are split into retained/conflict/repeat subtypes for clarity.
    audit.loc[retained.index, "disposition"] = retained["record_check"].map({
        "single_observation": "retained",
        "same_value_metadata_repeat": "retained_equivalent_repeat_flagged",
        "conflicting_country_year": "retained_conflict_flagged",
    }).astype("string")
    audit.loc[invalid.index, "reason"] = invalid["validation_reason"]
    audit.loc[duplicates.index, "reason"] = "identical across all raw columns; duplicate row removed"

    out_scope = raw.loc[outside_series | not_total].copy()
    out_scope["disposition"] = disposition.loc[out_scope.index]
    write_csv(dashboard, CLEAN_FILE)
    write_csv(conflict_rows, REVIEW_FILE)
    write_csv(invalid, INVALID_FILE)
    write_csv(duplicates, DUPLICATE_FILE)
    write_csv(out_scope, SCOPE_FILE)
    write_csv(audit, AUDIT_FILE)

    counts = disposition.value_counts().to_dict()
    # The categories below partition the input rows and must sum exactly to raw rows.
    partition_total = sum(int(counts.get(k, 0)) for k in [
        "outside_target_series", "not_total_population", "invalid_target_total",
        "exact_duplicate_removed", "retained_candidate"
    ])
    if partition_total != len(raw):
        raise AssertionError(
            f"Row accounting failed: dispositions total {partition_total:,}, input {len(raw):,}"
        )

    selected_series_rows = int(raw["SERIES"].eq(TARGET_SERIES).sum())
    total_population_rows = int((raw["SERIES"].eq(TARGET_SERIES) & raw["SEX"].eq("_T") & raw["AGE"].eq("_T")).sum())
    summary = {
        "raw_rows": len(raw),
        "original_rows": len(raw),
        "selected_series_rows": selected_series_rows,
        "total_population_rows": total_population_rows,
        "valid_rows_before_deduplication": int(valid.sum()),
        "outside_target_series_rows": int(counts.get("outside_target_series", 0)),
        "target_series_not_total_population_rows": int(counts.get("not_total_population", 0)),
        "invalid_target_total_rows": int(counts.get("invalid_target_total", 0)),
        "invalid_rows": int(counts.get("invalid_target_total", 0)),
        "exact_duplicate_rows_removed": int(counts.get("exact_duplicate_removed", 0)),
        "dashboard_rows_saved": len(dashboard),
        "clean_rows_saved": len(dashboard),
        "countries_saved": dashboard["country_code"].nunique(),
        "first_year": int(dashboard["year"].min()) if len(dashboard) else None,
        "last_year": int(dashboard["year"].max()) if len(dashboard) else None,
        "conflicting_country_year_rows_flagged": len(conflict_rows),
        "conflicting_country_year_rows": len(conflict_rows),
        "conflicting_country_year_groups": conflict_rows[key].drop_duplicates().shape[0],
        "equivalent_repeat_rows_flagged": len(repeat_rows),
        "row_level_audit_rows": len(audit),
        "partition_total_check": partition_total,
    }
    write_csv(pd.DataFrame(summary.items(), columns=["metric", "value"]), SUMMARY_FILE)

    print("FOODSHIELD AI — AUDITABLE DATA CLEANING")
    for metric, number in summary.items():
        print(f"{metric}: {number}")
    print("\nCreated:")
    for path in [CLEAN_FILE, REVIEW_FILE, INVALID_FILE, DUPLICATE_FILE, SUMMARY_FILE, SCOPE_FILE, AUDIT_FILE]:
        print(f" - {path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
