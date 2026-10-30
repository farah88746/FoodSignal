"""Pure-pandas data layer for FoodShield AI.

Nothing in this module imports Streamlit or Plotly, so every function can be
unit-tested without a browser. The rules implemented here:

* Records that fail validation are returned in ``rejected`` instead of being
  dropped silently.
* Conflicting values for the same country and year are never averaged,
  medianed or picked arbitrarily. They are flagged and kept.
* Two years are only compared for countries that have an unambiguous
  observation in BOTH years.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from .countries import country_name

REQUIRED_COLUMNS = ("country_code", "year", "food_insecurity_percent")
METADATA_COLUMNS = (
    "observation_status",
    "unit_measure",
    "time_detail",
    "source",
    "indicator",
)
NOT_PROVIDED = "Not provided"

# "2023-2025", "2023 - 2025", "2023–2025" (en dash) and em dash are accepted.
WINDOW_PATTERN = r"^\s*(\d{4})\s*[-\u2013\u2014]\s*(\d{4})\s*$"

RECORD_SINGLE = "Single record"
RECORD_REPEAT = "Repeated, same value"
RECORD_CONFLICT = "UNRESOLVED CONFLICT"


class LoadedData(NamedTuple):
    data: pd.DataFrame
    rejected: pd.DataFrame
    missing_metadata: tuple


# ----------------------------------------------------------------------
# Loading and validation
# ----------------------------------------------------------------------

def _clean_text(series: pd.Series) -> pd.Series:
    """Strip whitespace; turn blanks into missing values; return plain objects."""
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.mask(cleaned.fillna("").eq(""))
    return cleaned.astype(object).where(cleaned.notna(), None)


def add_reference_windows(df: pd.DataFrame, column: str = "time_detail") -> pd.DataFrame:
    """Parse the ``time_detail`` field (for example '2023-2025').

    Adds: window_start, window_end, window_years, window_label and
    window_centered (True when the reference year is the window midpoint).
    Unparseable values give NaN and the label 'Not provided'; nothing is guessed.
    """
    out = df.copy()
    parts = out[column].astype("string").str.extract(WINDOW_PATTERN)
    start = pd.to_numeric(parts[0], errors="coerce").astype(float)
    end = pd.to_numeric(parts[1], errors="coerce").astype(float)

    out["window_start"] = start
    out["window_end"] = end
    out["window_years"] = end - start + 1

    label = pd.Series(NOT_PROVIDED, index=out.index, dtype=object)
    ok = start.notna() & end.notna()
    label.loc[ok] = (
        start[ok].astype(int).astype(str) + "\u2013" + end[ok].astype(int).astype(str)
    )
    out["window_label"] = label
    out["window_centered"] = ((start + end) / 2 == out["year"]).astype(bool)
    return out


def flag_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Flag repeated country-year keys WITHOUT resolving or removing anything.

    records_for_key         number of rows sharing the country-year key
    distinct_values_for_key number of different estimates among those rows
    conflict                True when the same key carries different estimates
    record_status           single record / repeated, same value / conflict
    """
    out = df.copy()
    if out.empty:
        for col in ("records_for_key", "distinct_values_for_key"):
            out[col] = pd.Series(dtype=int)
        out["conflict"] = pd.Series(dtype=bool)
        out["record_status"] = pd.Series(dtype=object)
        return out

    grouped = out.groupby(["country_code", "year"])["food_insecurity_percent"]
    out["records_for_key"] = grouped.transform("size").astype(int)
    out["distinct_values_for_key"] = grouped.transform("nunique").astype(int)
    out["conflict"] = out["distinct_values_for_key"] > 1
    repeat = (out["records_for_key"] > 1) & ~out["conflict"]
    out["record_status"] = np.select(
        [out["conflict"], repeat],
        [RECORD_CONFLICT, RECORD_REPEAT],
        default=RECORD_SINGLE,
    ).astype(object)
    return out


def build_observations(raw: pd.DataFrame) -> LoadedData:
    """Validate a raw table and return (data, rejected, missing_metadata).

    Every input row ends up in exactly one of ``data`` or ``rejected``.
    """
    df = raw.copy()
    df.columns = df.columns.astype(str).str.strip()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    df["file_line"] = np.arange(2, len(df) + 2)  # line 1 is the CSV header
    df["country_code"] = _clean_text(df["country_code"]).str.upper()

    year_num = pd.to_numeric(df["year"], errors="coerce")
    value_num = pd.to_numeric(df["food_insecurity_percent"], errors="coerce")
    df["year"] = year_num
    df["food_insecurity_percent"] = value_num

    missing_metadata = []
    for col in METADATA_COLUMNS:
        if col in df.columns:
            df[col] = _clean_text(df[col])
        else:
            df[col] = None
            missing_metadata.append(col)

    checks = pd.DataFrame(
        {
            "invalid country code (expected three capital letters)": ~df["country_code"]
            .fillna("")
            .str.fullmatch(r"[A-Z]{3}"),
            "year missing, non-numeric, non-integer or outside 1900-2100": ~(
                year_num.notna() & (year_num % 1 == 0) & year_num.between(1900, 2100)
            ),
            "value missing, non-numeric or outside 0-100": ~(
                value_num.notna() & value_num.between(0, 100)
            ),
        },
        index=df.index,
    )
    invalid_mask = checks.any(axis=1)

    rejected = df.loc[invalid_mask].copy()
    if rejected.empty:
        rejected["reject_reason"] = pd.Series(dtype=object)
    else:
        rejected["reject_reason"] = checks.loc[invalid_mask].apply(
            lambda row: "; ".join(name for name, bad in row.items() if bad), axis=1
        )

    valid = df.loc[~invalid_mask].drop(columns="file_line").copy()
    valid["year"] = valid["year"].astype(int)
    valid["country_name"] = valid["country_code"].map(country_name).astype(object)

    valid = add_reference_windows(valid)
    for col in METADATA_COLUMNS:
        valid[col] = valid[col].where(valid[col].notna(), NOT_PROVIDED)
    valid = flag_duplicates(valid)

    valid = valid.sort_values(
        ["country_code", "year", "source", "food_insecurity_percent"]
    ).reset_index(drop=True)
    return LoadedData(valid, rejected.reset_index(drop=True), tuple(missing_metadata))


def load_observations(path) -> LoadedData:
    """Read the cleaned CSV and validate it."""
    raw = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    return build_observations(raw)


# ----------------------------------------------------------------------
# Reference-window helpers
# ----------------------------------------------------------------------

def window_label_for_year(df: pd.DataFrame, year: int) -> str:
    """Most common window label for a reference year, or 'Not provided'."""
    labels = df.loc[df["year"] == year, "window_label"]
    labels = labels[labels != NOT_PROVIDED]
    if labels.empty:
        return NOT_PROVIDED
    return str(labels.mode().iloc[0])


def window_example(df: pd.DataFrame):
    """(year, window) for the latest year with a parseable window, else None."""
    if df.empty:
        return None
    for year in sorted(df["year"].unique(), reverse=True):
        label = window_label_for_year(df, int(year))
        if label != NOT_PROVIDED:
            return int(year), label
    return None


# ----------------------------------------------------------------------
# Single-year views (no averaging, no silent resolution)
# ----------------------------------------------------------------------

def observations_for_year(df: pd.DataFrame, year: int):
    """Return (comparable, withheld) for one reference year.

    comparable: one row per country. Rows that merely repeat the same value
                are reduced to the first one (no value is changed).
    withheld:   ALL rows of country-years whose estimates conflict. They are
                excluded from rankings and shown for review, never merged.
    """
    rows = df[df["year"] == year]
    withheld = rows[rows["conflict"]].copy()
    comparable = (
        rows[~rows["conflict"]]
        .sort_values(["country_code", "source", "observation_status"])
        .drop_duplicates("country_code", keep="first")
        .copy()
    )
    return comparable, withheld


# ----------------------------------------------------------------------
# Coverage
# ----------------------------------------------------------------------

COVERAGE_COLUMNS = [
    "country_code",
    "country_name",
    "first_year",
    "latest_year",
    "latest_window",
    "n_years_observed",
    "years_in_dataset",
    "coverage_share",
    "gap_years",
    "n_conflict_years",
    "latest_value",
    "latest_source",
    "latest_status",
    "latest_year_conflict",
    "latest_is_current",
]


def coverage_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per country: which years exist and how current the data is."""
    if df.empty:
        return pd.DataFrame(columns=COVERAGE_COLUMNS)

    dataset_years = sorted(int(y) for y in df["year"].unique())
    newest = dataset_years[-1]
    records = []
    for code, group in df.groupby("country_code", sort=True):
        years = sorted(int(y) for y in group["year"].unique())
        first, latest = years[0], years[-1]
        span_years = [y for y in dataset_years if first <= y <= latest]
        latest_rows = group[group["year"] == latest].sort_values(
            ["source", "observation_status"]
        )
        reference = latest_rows.iloc[0]
        latest_conflict = bool(latest_rows["conflict"].any())
        records.append(
            {
                "country_code": code,
                "country_name": reference["country_name"],
                "first_year": first,
                "latest_year": latest,
                "latest_window": reference["window_label"],
                "n_years_observed": len(years),
                "years_in_dataset": len(dataset_years),
                "coverage_share": len(years) / len(dataset_years),
                "gap_years": len(span_years) - len(years),
                "n_conflict_years": int(group.loc[group["conflict"], "year"].nunique()),
                "latest_value": np.nan
                if latest_conflict
                else float(reference["food_insecurity_percent"]),
                "latest_source": reference["source"],
                "latest_status": reference["observation_status"],
                "latest_year_conflict": latest_conflict,
                "latest_is_current": latest == newest,
            }
        )
    return pd.DataFrame(records, columns=COVERAGE_COLUMNS)


# ----------------------------------------------------------------------
# Like-for-like change between two reference years
# ----------------------------------------------------------------------

def _slice(observations: pd.DataFrame, suffix: str) -> pd.DataFrame:
    rename = {
        "food_insecurity_percent": f"value_{suffix}",
        "window_label": f"label_{suffix}",
        "window_start": f"wstart_{suffix}",
        "window_end": f"wend_{suffix}",
        "source": f"source_{suffix}",
        "observation_status": f"status_{suffix}",
    }
    return observations[["country_code", "country_name", *rename]].rename(columns=rename)


def compute_changes(df: pd.DataFrame, start_year: int, end_year: int):
    """Compare two reference years for countries observed in BOTH.

    Returns (eligible, excluded). A country is eligible only when it has an
    unambiguous value in the start year and in the end year, so every
    comparison covers the same two reference years. Everything else goes to
    ``excluded`` with an explicit reason.
    """
    if start_year >= end_year:
        raise ValueError("start_year must be earlier than end_year")

    start_obs, start_conflicts = observations_for_year(df, start_year)
    end_obs, end_conflicts = observations_for_year(df, end_year)

    merged = _slice(start_obs, "start").merge(
        _slice(end_obs, "end").drop(columns="country_name"),
        on="country_code",
        how="inner",
    )
    merged["start_year"] = start_year
    merged["end_year"] = end_year
    merged["change_pp"] = (merged["value_end"] - merged["value_start"]).round(6)
    merged["direction"] = np.select(
        [merged["change_pp"] > 0, merged["change_pp"] < 0],
        ["Increase", "Decrease"],
        default="No change",
    )
    overlap = (
        np.minimum(merged["wend_start"], merged["wend_end"])
        - np.maximum(merged["wstart_start"], merged["wstart_end"])
        + 1
    )
    merged["windows_overlap_years"] = overlap.clip(lower=0)
    merged["source_changed"] = merged["source_start"] != merged["source_end"]
    eligible = merged.sort_values(
        ["change_pp", "country_name"], ascending=[False, True]
    ).reset_index(drop=True)

    names = df.drop_duplicates("country_code").set_index("country_code")["country_name"]
    start_ok = set(start_obs["country_code"])
    end_ok = set(end_obs["country_code"])
    start_bad = set(start_conflicts["country_code"])
    end_bad = set(end_conflicts["country_code"])

    excluded_rows = []
    for code in names.index:
        if code in start_ok and code in end_ok:
            continue
        reasons = []
        for year, ok, bad in (
            (start_year, start_ok, start_bad),
            (end_year, end_ok, end_bad),
        ):
            if code in bad:
                reasons.append(f"conflicting records in {year}")
            elif code not in ok:
                reasons.append(f"no observation for {year}")
        excluded_rows.append(
            {
                "country_code": code,
                "country_name": names[code],
                "reason": "; ".join(reasons).capitalize(),
            }
        )
    excluded = pd.DataFrame(
        excluded_rows, columns=["country_code", "country_name", "reason"]
    ).sort_values("country_name").reset_index(drop=True)
    return eligible, excluded


def select_changes(eligible: pd.DataFrame, mode: str, limit: int | None = None):
    """Pick rows to display. mode: 'increases', 'decreases' or 'all'.

    'all' returns EVERY eligible country; ``limit`` is ignored for it.
    """
    if mode == "increases":
        picked = eligible.sort_values(["change_pp", "country_name"], ascending=[False, True])
    elif mode == "decreases":
        picked = eligible.sort_values(["change_pp", "country_name"], ascending=[True, True])
    elif mode == "all":
        return eligible.sort_values(
            ["change_pp", "country_name"], ascending=[False, True]
        ).reset_index(drop=True)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    if limit is not None:
        picked = picked.head(limit)
    return picked.reset_index(drop=True)


def country_change(country_df: pd.DataFrame):
    """First-to-latest change for ONE country using unambiguous years only.

    Returns a Series (same columns as ``compute_changes``) or None when the
    country has fewer than two unambiguous reference years.
    """
    years = sorted(int(y) for y in country_df.loc[~country_df["conflict"], "year"].unique())
    if len(years) < 2:
        return None
    eligible, _ = compute_changes(country_df, years[0], years[-1])
    if eligible.empty:
        return None
    return eligible.iloc[0]


# ----------------------------------------------------------------------
# Display and export helpers
# ----------------------------------------------------------------------

def to_display_table(df: pd.DataFrame) -> pd.DataFrame:
    """Readable column names; keeps the ISO code next to the country name."""
    table = df[
        [
            "country_name",
            "country_code",
            "year",
            "window_label",
            "food_insecurity_percent",
            "source",
            "observation_status",
            "record_status",
        ]
    ].rename(
        columns={
            "country_name": "Country",
            "country_code": "ISO code",
            "year": "Reference year",
            "window_label": "Reference window",
            "food_insecurity_percent": "Estimate (%)",
            "source": "Source",
            "observation_status": "Status code",
            "record_status": "Record check",
        }
    )
    return table.reset_index(drop=True)


def to_export_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Export with country name AND ISO code, original metadata, and the check flag."""
    columns = [
        "country_code",
        "country_name",
        "year",
        "time_detail",
        "food_insecurity_percent",
        "observation_status",
        "unit_measure",
        "source",
        "indicator",
        "record_status",
    ]
    return df[columns].reset_index(drop=True)


def coverage_display_table(coverage: pd.DataFrame) -> pd.DataFrame:
    table = coverage[
        [
            "country_name",
            "country_code",
            "first_year",
            "latest_year",
            "latest_window",
            "n_years_observed",
            "years_in_dataset",
            "gap_years",
            "latest_value",
            "latest_source",
        ]
    ].rename(
        columns={
            "country_name": "Country",
            "country_code": "ISO code",
            "first_year": "First year",
            "latest_year": "Latest year",
            "latest_window": "Latest window",
            "n_years_observed": "Years observed",
            "years_in_dataset": "Years in dataset",
            "gap_years": "Missing years inside span",
            "latest_value": "Latest estimate (%)",
            "latest_source": "Latest source",
        }
    )
    return table.reset_index(drop=True)


# ----------------------------------------------------------------------
# Cleaning-pipeline statistics (data/processed/cleaning_summary.csv)
# ----------------------------------------------------------------------

CLEANING_LABELS = {
    "original_rows": "Rows in the raw FAO export",
    "selected_series_rows": "Rows in the selected series (AG_PRD_FIESMS)",
    "total_population_rows": "Rows for the total population (SEX and AGE = _T)",
    "valid_rows_before_deduplication": "Rows passing validation",
    "invalid_rows": "Rows failing validation (saved to invalid_records.csv)",
    "exact_duplicate_rows_removed": "Exact duplicate rows removed",
    "conflicting_country_year_rows": (
        "Rows with conflicting country-year values "
        "(left out of the dashboard file; saved to records_for_review.csv)"
    ),
    "clean_rows_saved": "Rows in the dashboard file",
    "countries_saved": "Countries in the dashboard file",
    "first_year": "First year in the dashboard file",
    "last_year": "Last year in the dashboard file",
}


def load_optional_csv(path) -> pd.DataFrame | None:
    """Read a CSV if it exists. Returns None when missing; empty frame if blank."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_cleaning_summary(path) -> pd.DataFrame | None:
    """Load cleaning_summary.csv (columns: metric, value). None if absent."""
    summary = load_optional_csv(path)
    if summary is None:
        return None
    summary.columns = summary.columns.astype(str).str.strip()
    if not {"metric", "value"}.issubset(summary.columns):
        raise ValueError("cleaning_summary.csv must have the columns 'metric' and 'value'")
    summary["metric"] = summary["metric"].astype(str).str.strip()
    return summary


def summary_value(summary: pd.DataFrame, metric: str):
    """Numeric value of one metric, or None if absent or not numeric."""
    row = summary.loc[summary["metric"] == metric, "value"]
    if row.empty:
        return None
    number = pd.to_numeric(row.iloc[0], errors="coerce")
    return None if pd.isna(number) else float(number)


def summary_for_display(summary: pd.DataFrame) -> pd.DataFrame:
    """Readable labels for known metrics; unknown metrics are shown as they are."""
    table = summary.copy()
    table["Measure"] = table["metric"].map(lambda m: CLEANING_LABELS.get(m, m))
    return table[["Measure", "value"]].rename(columns={"value": "Value"}).reset_index(drop=True)


def reconcile_cleaning_counts(summary: pd.DataFrame):
    """Rows that the pipeline summary does not explain.

    clean_data.py keeps one row per country-year. Rows that share a key AND a
    value but differ in metadata are collapsed in its last step and are not
    logged anywhere. This function derives that number from the summary:

        valid rows - exact duplicates - conflicting rows - rows saved

    Returns None when a needed metric is missing.
    """
    needed = [
        "valid_rows_before_deduplication",
        "exact_duplicate_rows_removed",
        "conflicting_country_year_rows",
        "clean_rows_saved",
    ]
    values = [summary_value(summary, m) for m in needed]
    if any(v is None for v in values):
        return None
    valid, exact, conflicts, saved = values
    return int(valid - exact - conflicts - saved)


def check_summary_consistency(summary: pd.DataFrame, rows_loaded: int, countries_loaded: int):
    """Warnings when cleaning_summary.csv does not match the loaded dataset."""
    messages = []
    saved = summary_value(summary, "clean_rows_saved")
    if saved is not None and int(saved) != rows_loaded:
        messages.append(
            f"cleaning_summary.csv reports {int(saved):,} saved rows but the loaded "
            f"file has {rows_loaded:,}. The summary may be out of date; re-run clean_data.py."
        )
    countries = summary_value(summary, "countries_saved")
    if countries is not None and int(countries) != countries_loaded:
        messages.append(
            f"cleaning_summary.csv reports {int(countries):,} countries but the loaded "
            f"file has {countries_loaded:,}."
        )
    leftover = reconcile_cleaning_counts(summary)
    if leftover is not None and leftover < 0:
        messages.append(
            "The row counts in cleaning_summary.csv do not add up "
            f"(difference {leftover:,}). Re-run clean_data.py."
        )
    return messages
