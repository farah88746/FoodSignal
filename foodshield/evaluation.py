"""Retrospective evaluation of the 'last-known higher-quarter standing' rule.

QUESTION (fixed before looking at results)
    The stale-standing improvement treats a country whose latest estimate is old
    as "higher last-known burden" when that estimate was in the higher quarter
    of the countries observed in the SAME reference year. That is only sensible
    if relative standing tends to persist. For countries that were observed
    again later, does higher-quarter standing in year T predict higher-quarter
    standing in year T + h?

DESIGN
    * Standing: for every reference year, the 75th percentile of that year's
      unambiguous country estimates (years with fewer than 4 peers are skipped).
      A country is "high" in a year if its estimate is at or above the cutoff.
    * Pairs: (country, origin T, target T + h) with an unambiguous estimate in both
      years. Primary horizon is h = 3, the shortest horizon at which the three-year
      reference windows do not overlap. Horizons 4 to 6 are reported as secondary.
    * Rule: predict "high at T + h" if and only if "high at T".
    * Baselines:
        base rate    share of pairs that are high at T + h (what an uninformed flag
                     would achieve; the previous app gave stale countries no burden
                     information at all, i.e. flagged none of them)
        cross-year   the stale estimate compared with the TARGET year's cutoff
                     instead of its own year's (the comparison the app avoids)
    * Uncertainty: Wilson intervals per origin; a country-clustered bootstrap
      (fixed seed) for the pooled h = 3 result, because the same countries recur
      across origins.

LIMITS (also printed in the report)
    * No ground truth for 'priority' exists. This tests persistence of relative
      standing between observed years, nothing more.
    * Countries that dropped out of the data (the truly stale ones) cannot be
      evaluated on their own missing future values. Whether they resemble the
      countries that were observed again is shown only as a diagnostic.
    * Few origins and heavily dependent pairs: treat intervals as approximate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as fsdata
from .screening import HIGH_QUANTILE, MIN_PEER_GROUP

DEFAULT_HORIZONS = (3, 4, 5, 6)
PRIMARY_HORIZON = 3
DEFAULT_SEED = 20261031
DEFAULT_BOOTSTRAPS = 2000


def wilson_interval(successes: int, n: int, z: float = 1.96):
    """Wilson score interval for a proportion; (nan, nan) when n == 0."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def yearly_standing(observations: pd.DataFrame) -> pd.DataFrame:
    """One row per country-year: value, that year's cutoff and the 'high' flag.

    Conflicting country-years are excluded (never averaged), exactly like the app.
    """
    frames = []
    for year in sorted(int(y) for y in observations["year"].unique()):
        comparable, _ = fsdata.observations_for_year(observations, year)
        if len(comparable) < MIN_PEER_GROUP:
            continue
        values = comparable["food_insecurity_percent"].astype(float)
        cutoff = float(values.quantile(HIGH_QUANTILE))
        frames.append(
            pd.DataFrame(
                {
                    "country_code": comparable["country_code"].to_numpy(),
                    "year": year,
                    "value": values.to_numpy(),
                    "cutoff": cutoff,
                    "high": (values >= cutoff).to_numpy(),
                    "peers": len(comparable),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["country_code", "year", "value", "cutoff", "high", "peers"])
    return pd.concat(frames, ignore_index=True)


def build_pairs(standing: pd.DataFrame, horizons=DEFAULT_HORIZONS) -> pd.DataFrame:
    """All (country, origin, origin + h) pairs observed in both years."""
    origin = standing.rename(
        columns={"year": "origin", "value": "value_origin", "cutoff": "cutoff_origin", "high": "high_origin"}
    )[["country_code", "origin", "value_origin", "cutoff_origin", "high_origin"]]
    target = standing.rename(
        columns={"year": "target", "value": "value_target", "cutoff": "cutoff_target", "high": "high_target"}
    )[["country_code", "target", "value_target", "cutoff_target", "high_target"]]

    pieces = []
    for horizon in horizons:
        shifted = origin.assign(target=origin["origin"] + horizon, horizon=horizon)
        pieces.append(shifted.merge(target, on=["country_code", "target"], how="inner"))
    if not pieces:
        return pd.DataFrame()
    pairs = pd.concat(pieces, ignore_index=True)
    pairs["high_origin_vs_target_cutoff"] = pairs["value_origin"] >= pairs["cutoff_target"]
    return pairs


def classification_summary(pairs: pd.DataFrame, flag_col: str) -> dict:
    """Confusion counts and rates for a boolean flag against 'high at target'."""
    flag = pairs[flag_col].astype(bool)
    outcome = pairs["high_target"].astype(bool)
    n = len(pairs)
    tp = int((flag & outcome).sum())
    fp = int((flag & ~outcome).sum())
    fn = int((~flag & outcome).sum())
    tn = int((~flag & ~outcome).sum())
    flagged = tp + fp
    positives = tp + fn
    base_rate = positives / n if n else float("nan")
    precision = tp / flagged if flagged else float("nan")
    recall = tp / positives if positives else float("nan")
    miss_rate = fn / (fn + tn) if (fn + tn) else float("nan")
    return {
        "pairs": n,
        "flagged": flagged,
        "high_at_target": positives,
        "true_pos": tp,
        "false_pos": fp,
        "false_neg": fn,
        "true_neg": tn,
        "base_rate": base_rate,
        "precision": precision,
        "precision_ci_low": wilson_interval(tp, flagged)[0],
        "precision_ci_high": wilson_interval(tp, flagged)[1],
        "lift_over_base_rate": precision / base_rate if base_rate and not np.isnan(precision) else float("nan"),
        "recall": recall,
        "recall_ci_low": wilson_interval(tp, positives)[0],
        "recall_ci_high": wilson_interval(tp, positives)[1],
        "high_rate_when_not_flagged": miss_rate,
    }


def _spearman(pairs: pd.DataFrame) -> float:
    if len(pairs) < 3:
        return float("nan")
    return float(pairs["value_origin"].rank().corr(pairs["value_target"].rank()))


def clustered_bootstrap(
    pairs: pd.DataFrame,
    flag_col: str = "high_origin",
    n_boot: int = DEFAULT_BOOTSTRAPS,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Percentile 95% intervals for precision, recall and lift, resampling countries."""
    countries = pairs["country_code"].unique()
    groups = {c: g for c, g in pairs.groupby("country_code")}
    rng = np.random.default_rng(seed)
    precision, recall, lift = [], [], []
    for _ in range(n_boot):
        drawn = rng.choice(countries, size=len(countries), replace=True)
        sample = pd.concat([groups[c] for c in drawn], ignore_index=True)
        stats = classification_summary(sample, flag_col)
        precision.append(stats["precision"])
        recall.append(stats["recall"])
        lift.append(stats["lift_over_base_rate"])

    def interval(values):
        values = np.asarray(values, dtype=float)
        values = values[~np.isnan(values)]
        if values.size == 0:
            return float("nan"), float("nan")
        return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))

    return {
        "precision_boot_low": interval(precision)[0],
        "precision_boot_high": interval(precision)[1],
        "recall_boot_low": interval(recall)[0],
        "recall_boot_high": interval(recall)[1],
        "lift_boot_low": interval(lift)[0],
        "lift_boot_high": interval(lift)[1],
        "n_countries": int(len(countries)),
        "n_boot": n_boot,
        "seed": seed,
    }


def attrition_diagnostic(standing: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """For each origin: are countries missing at origin + h different from those observed again?

    Shows whether the persistence result could transfer to countries that stopped
    being observed (the stale ones). It cannot prove it does.
    """
    rows = []
    by_year = {y: g for y, g in standing.groupby("year")}
    for origin, group in sorted(by_year.items()):
        target = origin + horizon
        if target not in by_year:
            continue
        seen_again = set(by_year[target]["country_code"])
        retained = group[group["country_code"].isin(seen_again)]
        lost = group[~group["country_code"].isin(seen_again)]
        rows.append(
            {
                "origin": origin,
                "target": target,
                "observed_at_origin": len(group),
                "retained": len(retained),
                "lost": len(lost),
                "share_high_retained": float(retained["high"].mean()) if len(retained) else float("nan"),
                "share_high_lost": float(lost["high"].mean()) if len(lost) else float("nan"),
                "high_lost_count": int(lost["high"].sum()),
            }
        )
    return pd.DataFrame(rows)


def run_evaluation(
    observations: pd.DataFrame,
    horizons=DEFAULT_HORIZONS,
    n_boot: int = DEFAULT_BOOTSTRAPS,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Run the full, pre-specified evaluation. Returns a dict of DataFrames."""
    standing = yearly_standing(observations)
    pairs = build_pairs(standing, horizons)

    by_horizon, by_origin = [], []
    for horizon, group in pairs.groupby("horizon"):
        for rule, flag_col in (
            ("own-year standing (new rule)", "high_origin"),
            ("cross-year cutoff (baseline)", "high_origin_vs_target_cutoff"),
        ):
            stats = classification_summary(group, flag_col)
            by_horizon.append({"horizon": horizon, "rule": rule, **stats, "spearman_levels": _spearman(group)})
        for origin, sub in group.groupby("origin"):
            by_origin.append(
                {"horizon": horizon, "origin": origin, "target": origin + horizon,
                 **classification_summary(sub, "high_origin")}
            )

    primary = pairs[pairs["horizon"] == PRIMARY_HORIZON]
    boot = (
        clustered_bootstrap(primary, "high_origin", n_boot=n_boot, seed=seed)
        if len(primary)
        else {}
    )
    return {
        "standing": standing,
        "pairs": pairs,
        "by_horizon": pd.DataFrame(by_horizon),
        "by_origin": pd.DataFrame(by_origin),
        "primary_bootstrap": pd.DataFrame([boot]) if boot else pd.DataFrame(),
        "attrition": attrition_diagnostic(standing, PRIMARY_HORIZON),
    }
