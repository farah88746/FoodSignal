"""Priority screening and decision-brief logic (pure pandas, no Streamlit).

This is a faithful port of the screening block that lived inside app.py, so it
can be unit-tested and evaluated. With ``stale_standing=False`` it reproduces
the previous behaviour exactly.

The one deliberate improvement (``stale_standing=True``, the default):

    A country whose latest usable estimate is older than the dataset's latest
    reference year used to be reported only as "Data update needed", whatever
    its burden. It is now also compared with the other countries observed in
    ITS OWN last reference year (the same like-for-like rule used for current
    estimates). If it was in the higher quarter of that year, it is reported
    as "Data update needed: higher last-known burden" and sorted above generic
    stale or routine cases. Stale estimates are still never ranked against
    current-year estimates.

Nothing here forecasts anything. "Higher quarter" is a relative position among
observed estimates for one reference year, not a policy threshold.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .countries import country_label

HIGH_QUANTILE = 0.75
MIN_PEER_GROUP = 4
TREND_TOLERANCE_PP = 0.05  # display tolerance, not a significance test

BURDEN_HIGH = "Higher quarter of current-year estimates"
BURDEN_BELOW = "Below higher quarter"
BURDEN_SMALL_GROUP = "Comparable-year estimate (small peer group)"
BURDEN_PENDING = "Pending peer comparison"
BURDEN_NOT_ASSESSABLE = "Not assessable"

CAT_HIGHER = "Higher review priority"
CAT_REVIEW_TREND = "Review burden and trend"
CAT_HIGH_TREND_UNCERTAIN = "Higher burden; trend uncertain"
CAT_STALE_HIGH = "Data update needed: higher last-known burden"
CAT_MONITOR_INCREASE = "Monitor increasing signal"
CAT_EVIDENCE = "Evidence review needed"
CAT_DATA_UPDATE = "Data update needed"
CAT_ROUTINE = "Routine monitoring"

SORT_ORDER = {
    CAT_HIGHER: 0,
    CAT_REVIEW_TREND: 1,
    CAT_HIGH_TREND_UNCERTAIN: 2,
    CAT_STALE_HIGH: 2.5,
    CAT_MONITOR_INCREASE: 3,
    CAT_EVIDENCE: 4,
    CAT_DATA_UPDATE: 5,
    CAT_ROUTINE: 6,
}

PUBLIC_COLUMNS = [
    "Country",
    "Country code",
    "Latest estimate (%)",
    "Reference year",
    "Burden relative to peers",
    "Historical signal",
    "Evidence availability",
    "Priority screen",
    "Reason",
    "Unambiguous years",
    "Sources",
    "Conflict years",
]
DISPLAY_COLUMNS = [
    "Country",
    "Latest estimate (%)",
    "Reference year",
    "Burden relative to peers",
    "Historical signal",
    "Evidence availability",
    "Priority screen",
    "Reason",
    "Unambiguous years",
    "Sources",
    "Conflict years",
]


# ----------------------------------------------------------------------
# Reference windows
# ----------------------------------------------------------------------

def parse_window_bounds(value):
    """Return (start, end) for a 'YYYY-YYYY' reference window, or None."""
    match = re.search(r"(\d{4})\s*[-\u2013]\s*(\d{4})", str(value))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def window_overlap_years(left, right):
    """Years shared by two reference windows (0 if none, None if unparseable)."""
    left_bounds = parse_window_bounds(left)
    right_bounds = parse_window_bounds(right)
    if not left_bounds or not right_bounds:
        return None
    return max(0, min(left_bounds[1], right_bounds[1]) - max(left_bounds[0], right_bounds[0]) + 1)


def _window_of(row):
    return str(row.get("window_label", row.get("time_detail", "")))


def historical_signal(clean_country: pd.DataFrame, latest_row: pd.Series):
    """Latest minus the most recent earlier estimate with a NON-overlapping window.

    Returns (delta_or_None, label).
    """
    latest_window = _window_of(latest_row)
    delta = None
    earlier = clean_country[clean_country["year"] < latest_row["year"]]
    for _, candidate in earlier.sort_values("year", ascending=False).iterrows():
        if window_overlap_years(_window_of(candidate), latest_window) == 0:
            delta = float(latest_row["food_insecurity_percent"]) - float(
                candidate["food_insecurity_percent"]
            )
            break
    if delta is None:
        return None, "Insufficient comparable periods"
    if delta > TREND_TOLERANCE_PP:
        return delta, "Increasing"
    if delta < -TREND_TOLERANCE_PP:
        return delta, "Decreasing"
    return delta, "Little/no numerical change"


def evidence_label(age: int, n_years: int, n_sources: int) -> str:
    if age == 0 and n_years >= 4 and n_sources >= 1:
        return "More available"
    if age <= 1 and n_years >= 2:
        return "Some available"
    return "Limited"


# ----------------------------------------------------------------------
# Peer standing (like-for-like reference year)
# ----------------------------------------------------------------------

def year_peer_cutoffs(clean_by_country: dict):
    """{year: 75th percentile of that year's unambiguous estimates, or None if < 4 peers}.

    ``clean_by_country`` maps country code -> one-row-per-year unambiguous frame.
    """
    values_by_year = {}
    for frame in clean_by_country.values():
        for year, value in zip(frame["year"], frame["food_insecurity_percent"]):
            values_by_year.setdefault(int(year), []).append(float(value))
    cutoffs = {}
    for year, values in values_by_year.items():
        cutoffs[year] = (
            float(pd.Series(values).quantile(HIGH_QUANTILE)) if len(values) >= MIN_PEER_GROUP else None
        )
    return cutoffs, {year: len(v) for year, v in values_by_year.items()}


# ----------------------------------------------------------------------
# Priority table
# ----------------------------------------------------------------------

def build_priority_table(
    observations: pd.DataFrame,
    latest_year: int,
    stale_standing: bool = True,
    country_order=None,
):
    """Build the screening table (one row per country) and report the cutoff used.

    ``country_order`` (optional list of ISO codes) fixes the row order before
    sorting, so ties keep the order the dashboard already used.
    Returns (table, info) where info has: current_cutoff, n_comparable.
    The table is sorted by priority category, then by latest estimate.
    """
    clean_by_country = {}
    meta = {}
    for code, country in observations.groupby("country_code", sort=False):
        country = country.sort_values(["year", "source"])
        clean_by_country[code] = (
            country[~country["conflict"]].drop_duplicates("year").sort_values("year")
        )
        meta[code] = {
            "conflict_years": int(country.loc[country["conflict"], "year"].nunique()),
            "sources_all": int(country["source"].nunique()) if "source" in country else 0,
        }

    cutoffs, peer_counts = year_peer_cutoffs(clean_by_country)

    rows = []
    codes = list(country_order) if country_order is not None else list(dict.fromkeys(observations["country_code"]))
    for code in codes:
        clean_country = clean_by_country[code]
        base = {
            "Country": country_label(code),
            "Country code": code,
            "Conflict years": meta[code]["conflict_years"],
        }
        if clean_country.empty:
            rows.append(
                {
                    **base,
                    "Latest estimate (%)": None,
                    "Reference year": None,
                    "Burden relative to peers": BURDEN_NOT_ASSESSABLE,
                    "Historical signal": "Not assessable",
                    "Evidence availability": "Limited",
                    "Priority screen": CAT_EVIDENCE,
                    "Reason": "No unambiguous estimate available",
                    "Unambiguous years": 0,
                    "Sources": meta[code]["sources_all"],
                }
            )
            continue

        latest = clean_country.iloc[-1]
        latest_value = float(latest["food_insecurity_percent"])
        latest_ref_year = int(latest["year"])
        n_years = int(clean_country["year"].nunique())
        n_sources = int(clean_country["source"].nunique()) if "source" in clean_country else 0
        age = max(0, int(latest_year) - latest_ref_year)
        _, signal = historical_signal(clean_country, latest)

        rows.append(
            {
                **base,
                "Latest estimate (%)": latest_value,
                "Reference year": latest_ref_year,
                "Burden relative to peers": BURDEN_PENDING,
                "Historical signal": signal,
                "Evidence availability": evidence_label(age, n_years, n_sources),
                "Priority screen": BURDEN_PENDING,
                "Reason": "",
                "Unambiguous years": n_years,
                "Sources": n_sources,
            }
        )

    table = pd.DataFrame(rows, columns=PUBLIC_COLUMNS)
    info = {"current_cutoff": None, "n_comparable": 0}
    if table.empty:
        return table, info

    current = table["Reference year"].notna() & (table["Reference year"] == int(latest_year))
    current &= table["Latest estimate (%)"].notna()
    info["n_comparable"] = int(current.sum())
    peers_now = int(current.sum())

    # Burden for current-year estimates (identical to the previous behaviour).
    if peers_now >= MIN_PEER_GROUP:
        info["current_cutoff"] = float(table.loc[current, "Latest estimate (%)"].quantile(HIGH_QUANTILE))
        table.loc[current, "Burden relative to peers"] = table.loc[current, "Latest estimate (%)"].apply(
            lambda v: BURDEN_HIGH if v >= info["current_cutoff"] else BURDEN_BELOW
        )
    elif peers_now > 0:
        table.loc[current, "Burden relative to peers"] = BURDEN_SMALL_GROUP

    stale_standing_label = {}
    if stale_standing:
        stale = (
            table["Reference year"].notna()
            & (table["Reference year"] != int(latest_year))
            & table["Latest estimate (%)"].notna()
        )
        for idx in table.index[stale]:
            year = int(table.at[idx, "Reference year"])
            cutoff = cutoffs.get(year)
            if cutoff is None:
                continue
            high = float(table.at[idx, "Latest estimate (%)"]) >= cutoff
            stale_standing_label[idx] = (high, year, cutoff, peer_counts.get(year, 0))
            table.at[idx, "Burden relative to peers"] = (
                f"Higher quarter of {year} estimates (last known)"
                if high
                else f"Below higher quarter of {year} estimates (last known)"
            )

    def screen(idx, row):
        if row["Reference year"] is None or pd.isna(row["Reference year"]):
            return CAT_EVIDENCE, "No usable estimate"
        if int(row["Reference year"]) != int(latest_year):
            year = int(row["Reference year"])
            if idx in stale_standing_label:
                high, _, cutoff, n_peers = stale_standing_label[idx]
                if high:
                    return (
                        CAT_STALE_HIGH,
                        f"Latest usable estimate ({row['Latest estimate (%)']:.1f}%) is from {year} "
                        f"and was in the higher quarter of {n_peers} countries observed in {year} "
                        f"(cutoff {cutoff:.1f}%); dataset ends at {latest_year}",
                    )
            return (
                CAT_DATA_UPDATE,
                f"Latest usable estimate is from {year}; dataset ends at {latest_year}",
            )
        if row["Burden relative to peers"] == BURDEN_HIGH:
            if row["Historical signal"] == "Increasing":
                return CAT_HIGHER, "Higher relative burden and increasing historical signal"
            if row["Historical signal"] == "Decreasing":
                return CAT_REVIEW_TREND, "Higher relative burden, although the historical signal is decreasing"
            if row["Historical signal"] == "Little/no numerical change":
                return CAT_REVIEW_TREND, "Higher relative burden with little numerical change"
            return (
                CAT_HIGH_TREND_UNCERTAIN,
                "Higher relative burden; insufficient non-overlapping periods for trend",
            )
        if row["Historical signal"] == "Increasing":
            return (
                CAT_MONITOR_INCREASE,
                "Increasing historical signal, but not in the higher relative-burden group",
            )
        if row["Evidence availability"] == "Limited":
            return CAT_EVIDENCE, "Limited historical coverage or stale estimate"
        return CAT_ROUTINE, "No higher relative-burden signal in this comparison; continue monitoring"

    screened = [screen(idx, row) for idx, row in table.iterrows()]
    table["Priority screen"] = [s[0] for s in screened]
    table["Reason"] = [s[1] for s in screened]

    table["_sort"] = table["Priority screen"].map(SORT_ORDER).fillna(9)
    table = table.sort_values(
        ["_sort", "Latest estimate (%)"], ascending=[True, False], na_position="last"
    ).drop(columns="_sort")
    return table, info


# ----------------------------------------------------------------------
# Decision brief
# ----------------------------------------------------------------------

def follow_up_steps(row) -> list:
    """Recommended verification steps for one screening row (a Series or dict)."""
    steps = []
    category = row["Priority screen"]
    if category == CAT_STALE_HIGH:
        steps.append(
            "Prioritise obtaining a newer official estimate: the last known estimate was in the "
            "higher quarter of its reference year, and no estimate exists for the latest year."
        )
    if category in (CAT_EVIDENCE, CAT_DATA_UPDATE):
        steps.append("Verify whether newer official estimates or country-specific data are available.")
    if row["Historical signal"] == "Increasing":
        steps.append("Review the historical increase alongside recent local context and other trusted indicators.")
    elif row["Historical signal"] == "Decreasing":
        steps.append(
            "Check whether the decrease is sustained and consistent with other sources before drawing conclusions."
        )
    elif row["Historical signal"] == "Insufficient comparable periods":
        steps.append("Locate an earlier non-overlapping reference period before interpreting direction.")
    if int(row["Conflict years"]) > 0:
        steps.append("Resolve conflicting country-year observations before using the signal operationally.")
    if int(row["Unambiguous years"]) < 4:
        steps.append("Treat the historical pattern cautiously because the number of unambiguous years is limited.")
    if not steps:
        steps.append("Cross-check the estimate against recent official reporting and local assessments before taking action.")
    return steps


def build_brief_markdown(row) -> str:
    """Markdown brief for one screening row. Bullets are separated by real newlines."""
    latest = row["Latest estimate (%)"]
    latest_text = "No unambiguous estimate available" if pd.isna(latest) else f"{float(latest):.1f}%"
    year = row["Reference year"]
    year_text = "Not available" if pd.isna(year) else str(int(year))
    steps = "\n".join(f"- {step}" for step in follow_up_steps(row))
    return (
        "# FoodShield AI \u2014 Country Evidence Brief\n\n"
        f"- **Country / area:** {row['Country']}\n"
        f"- **Latest observed estimate:** {latest_text}\n"
        f"- **Reference year:** {year_text}\n"
        f"- **Relative burden:** {row['Burden relative to peers']}\n"
        f"- **Historical signal:** {row['Historical signal']}\n"
        f"- **Evidence availability:** {row['Evidence availability']}\n"
        f"- **Priority screening category:** {row['Priority screen']}\n"
        f"- **Reason shown by screening:** {row['Reason']}\n"
        f"- **Unambiguous reference years:** {int(row['Unambiguous years'])}\n"
        f"- **Distinct sources recorded:** {int(row['Sources'])}\n"
        f"- **Conflict years flagged:** {int(row['Conflict years'])}\n\n"
        "## Recommended follow-up\n"
        f"{steps}\n\n"
        "## Interpretation limits\n"
        "This is a transparent historical screening aid, not a forecast, validated risk score, "
        "causal analysis, or substitute for local assessment. Relative burden is defined against "
        "the observed peer group for the same reference year. Evidence-availability labels describe "
        "coverage, not certainty. This brief should not be used alone to allocate aid.\n"
    )
