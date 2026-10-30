"""Tests for foodshield.screening (priority screening and decision brief).

Synthetic fixtures use made-up numbers purely to exercise the rules. The last
class compares the ported logic with the output captured from the app that was
running before this change, on the real data file (skipped if it differs).
"""

import hashlib
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foodshield import data as fsdata  # noqa: E402
from foodshield import screening as sc  # noqa: E402
from foodshield.countries import country_name  # noqa: E402

FAO = "FAO"
OTHER = "National Entity"


def raw(rows):
    """rows: (code, year, value[, source]) with the 3-year window centred on year."""
    records = []
    for r in rows:
        code, year, value = r[:3]
        source = r[3] if len(r) > 3 else FAO
        records.append(
            {
                "country_code": code,
                "year": year,
                "food_insecurity_percent": value,
                "time_detail": f"{year - 1}-{year + 1}",
                "source": source,
                "observation_status": "E",
                "unit_measure": "PT",
                "indicator": "2.1.2",
            }
        )
    return pd.DataFrame(records)


def obs(rows):
    return fsdata.build_observations(raw(rows)).data


def years(code, values, start=2018, source=FAO):
    return [(code, start + i, v, source) for i, v in enumerate(values)]


def row_for(table, code):
    return table[table["Country code"] == code].iloc[0]


class WindowTests(unittest.TestCase):
    def test_overlap_of_adjacent_and_separated_windows(self):
        self.assertEqual(sc.window_overlap_years("2014-2016", "2015-2017"), 2)
        self.assertEqual(sc.window_overlap_years("2014-2016", "2017-2019"), 0)
        self.assertEqual(sc.window_overlap_years("2014\u20132016", "2016-2018"), 1)
        self.assertIsNone(sc.window_overlap_years("n/a", "2016-2018"))

    def test_parse_window_bounds(self):
        self.assertEqual(sc.parse_window_bounds("2023-2025"), (2023, 2025))
        self.assertIsNone(sc.parse_window_bounds("unknown"))


class HistoricalSignalTests(unittest.TestCase):
    def signal(self, values_by_year, tolerance_check=None):
        data = obs([("AAA", y, v) for y, v in values_by_year.items()])
        clean = data.sort_values("year")
        return sc.historical_signal(clean, clean.iloc[-1])

    def test_uses_non_overlapping_earlier_window_not_the_previous_year(self):
        # 2024 vs 2023/2022 overlap; the first non-overlapping earlier year is 2021.
        delta, label = self.signal({2021: 10.0, 2022: 40.0, 2023: 40.0, 2024: 12.0})
        self.assertAlmostEqual(delta, 2.0)
        self.assertEqual(label, "Increasing")

    def test_decreasing_and_no_change_tolerance(self):
        self.assertEqual(self.signal({2018: 20.0, 2021: 15.0})[1], "Decreasing")
        self.assertEqual(self.signal({2018: 20.0, 2021: 20.04})[1], "Little/no numerical change")
        self.assertEqual(self.signal({2018: 20.0, 2021: 20.06})[1], "Increasing")

    def test_insufficient_when_only_overlapping_years(self):
        delta, label = self.signal({2023: 10.0, 2024: 12.0})
        self.assertIsNone(delta)
        self.assertEqual(label, "Insufficient comparable periods")


class EvidenceLabelTests(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(sc.evidence_label(0, 4, 1), "More available")
        self.assertEqual(sc.evidence_label(0, 3, 1), "Some available")
        self.assertEqual(sc.evidence_label(1, 2, 1), "Some available")
        self.assertEqual(sc.evidence_label(2, 9, 1), "Limited")
        self.assertEqual(sc.evidence_label(0, 1, 1), "Limited")


def scenario():
    """Latest year 2024. Ten current countries (values 10..100 step 10, cutoff 77.5),
    plus stale and special cases."""
    rows = []
    for i, code in enumerate(["CAA", "CAB", "CAC", "CAD", "CAE", "CAF", "CAG", "CAH", "CAI", "CAJ"]):
        latest = 10.0 * (i + 1)
        # earlier non-overlapping point 3 years before: higher burden countries CAI and CAJ increased
        earlier = latest - 5.0 if code in ("CAI", "CAJ") else latest + 5.0
        rows += [(code, 2021, earlier), (code, 2022, earlier), (code, 2023, earlier), (code, 2024, latest)]
    # Stale, high in its own year (2022): peers observed in 2022 are the ten countries above
    # (their 2022 values are 'earlier'), so 2022 cutoff is computed from those.
    rows += [("STH", 2020, 90.0), ("STH", 2022, 95.0)]
    # Stale, low in its own year
    rows += [("STL", 2020, 5.0), ("STL", 2022, 6.0)]
    # Stale but too few peers in its own year (2015 only one other country)
    rows += [("OLD", 2015, 99.0), ("OLE", 2015, 1.0)]
    return obs(rows)


class PriorityTableTests(unittest.TestCase):
    def setUp(self):
        self.data = scenario()
        self.new, self.info = sc.build_priority_table(self.data, 2024, stale_standing=True)
        self.old, _ = sc.build_priority_table(self.data, 2024, stale_standing=False)

    def test_current_year_cutoff_uses_only_current_year_estimates(self):
        self.assertAlmostEqual(self.info["current_cutoff"], 77.5)
        self.assertEqual(self.info["n_comparable"], 10)
        self.assertEqual(row_for(self.new, "CAJ")["Burden relative to peers"], sc.BURDEN_HIGH)
        self.assertEqual(row_for(self.new, "CAG")["Burden relative to peers"], sc.BURDEN_BELOW)

    def test_high_burden_with_increase_is_highest_priority(self):
        top = row_for(self.new, "CAJ")
        self.assertEqual(top["Priority screen"], sc.CAT_HIGHER)
        self.assertEqual(top["Historical signal"], "Increasing")

    def test_stale_high_is_flagged_only_when_improvement_is_on(self):
        self.assertEqual(row_for(self.new, "STH")["Priority screen"], sc.CAT_STALE_HIGH)
        self.assertEqual(row_for(self.old, "STH")["Priority screen"], sc.CAT_DATA_UPDATE)
        self.assertEqual(row_for(self.old, "STH")["Burden relative to peers"], sc.BURDEN_PENDING)

    def test_stale_low_stays_generic_data_update(self):
        self.assertEqual(row_for(self.new, "STL")["Priority screen"], sc.CAT_DATA_UPDATE)
        self.assertIn("Below higher quarter of 2022", row_for(self.new, "STL")["Burden relative to peers"])

    def test_stale_standing_uses_its_own_year_not_the_current_cutoff(self):
        # 2022 values of the ten current countries are 'earlier' (15..105 style); STH=95 is high
        # against those, and 95 is also above the 2024 cutoff, so also check a value that is
        # high in 2022 but below the 2024 cutoff.
        data = obs(
            [(f"PA{'ABCD'[i]}", 2022, v) for i, v in enumerate([10.0, 20.0, 30.0, 40.0])]
            + [(f"PA{'ABCD'[i]}", 2024, v) for i, v in enumerate([60.0, 70.0, 80.0, 90.0])]
            + [("MID", 2022, 40.0)]
        )
        table, info = sc.build_priority_table(data, 2024)
        self.assertGreater(info["current_cutoff"], 40.0)  # 40 would NOT be high in 2024
        mid = row_for(table, "MID")
        self.assertEqual(mid["Priority screen"], sc.CAT_STALE_HIGH)  # 40 is high among 2022 peers

    def test_too_few_peers_in_stale_year_falls_back_to_generic(self):
        self.assertEqual(row_for(self.new, "OLD")["Priority screen"], sc.CAT_DATA_UPDATE)
        self.assertEqual(row_for(self.new, "OLD")["Burden relative to peers"], sc.BURDEN_PENDING)

    def test_improvement_changes_only_stale_countries(self):
        merged = self.new.merge(self.old, on="Country code", suffixes=("_new", "_old"))
        changed = merged[merged["Priority screen_new"] != merged["Priority screen_old"]]
        self.assertEqual(set(changed["Country code"]), {"STH"})
        current = merged[merged["Reference year_new"] == 2024]
        for column in ("Priority screen", "Reason", "Burden relative to peers", "Historical signal"):
            self.assertTrue((current[f"{column}_new"] == current[f"{column}_old"]).all(), column)

    def test_sort_order_places_stale_high_between_trend_uncertain_and_monitor_increasing(self):
        self.assertLess(sc.SORT_ORDER[sc.CAT_HIGH_TREND_UNCERTAIN], sc.SORT_ORDER[sc.CAT_STALE_HIGH])
        self.assertLess(sc.SORT_ORDER[sc.CAT_STALE_HIGH], sc.SORT_ORDER[sc.CAT_MONITOR_INCREASE])
        order = self.new["Priority screen"].map(sc.SORT_ORDER).tolist()
        self.assertEqual(order, sorted(order))

    def test_reason_for_stale_high_states_year_value_and_cutoff(self):
        reason = row_for(self.new, "STH")["Reason"]
        self.assertIn("2022", reason)
        self.assertIn("95.0%", reason)
        self.assertIn("cutoff", reason)
        self.assertIn("dataset ends at 2024", reason)

    def test_every_country_appears_exactly_once(self):
        self.assertEqual(len(self.new), self.data["country_code"].nunique())
        self.assertEqual(self.new["Country code"].is_unique, True)


class ConflictAndMissingTests(unittest.TestCase):
    def test_conflicting_latest_year_is_excluded_not_averaged(self):
        rows = (
            years("CCC", [10.0, 12.0, 14.0, 20.0, 22.0, 24.0, 26.0])
            + [("CCC", 2024, 99.0, OTHER)]  # conflicts with FAO 2024 value
        )
        table, _ = sc.build_priority_table(obs(rows), 2024)
        row = row_for(table, "CCC")
        self.assertEqual(int(row["Conflict years"]), 1)
        self.assertEqual(int(row["Reference year"]), 2023)  # latest unambiguous year
        self.assertEqual(float(row["Latest estimate (%)"]), 24.0)

    def test_country_with_only_conflicts_needs_evidence_review(self):
        data = obs([("ZZZ", 2024, 10.0, FAO), ("ZZZ", 2024, 30.0, OTHER), ("AAA", 2024, 5.0)])
        table, _ = sc.build_priority_table(data, 2024)
        row = row_for(table, "ZZZ")
        self.assertEqual(row["Priority screen"], sc.CAT_EVIDENCE)
        self.assertEqual(row["Reason"], "No usable estimate")  # same wording the previous app showed
        self.assertTrue(pd.isna(row["Latest estimate (%)"]))

    def test_small_peer_group_is_not_ranked(self):
        data = obs([("AAA", 2024, 10.0), ("BBB", 2024, 20.0), ("CCC", 2024, 30.0)])
        table, info = sc.build_priority_table(data, 2024)
        self.assertIsNone(info["current_cutoff"])
        self.assertTrue((table["Burden relative to peers"] == sc.BURDEN_SMALL_GROUP).all())


class BriefTests(unittest.TestCase):
    def setUp(self):
        data = scenario()
        self.table, _ = sc.build_priority_table(data, 2024)

    def test_brief_uses_real_newlines_between_follow_up_steps(self):
        brief = sc.build_brief_markdown(row_for(self.table, "STH"))
        self.assertNotIn("\\n", brief)  # no literal backslash-n
        follow = brief.split("## Recommended follow-up\n")[1].split("\n\n## Interpretation limits")[0]
        steps = follow.split("\n")
        self.assertGreaterEqual(len(steps), 2)
        self.assertTrue(all(s.startswith("- ") for s in steps))

    def test_country_label_is_not_duplicated(self):
        brief = sc.build_brief_markdown(row_for(self.table, "CAJ"))
        self.assertEqual(brief.count("(CAJ)"), 1)

    def test_stale_high_gets_a_specific_first_step(self):
        steps = sc.follow_up_steps(row_for(self.table, "STH"))
        self.assertIn("newer official estimate", steps[0])

    def test_brief_for_country_without_estimate(self):
        data = obs([("ZZZ", 2024, 10.0, FAO), ("ZZZ", 2024, 30.0, OTHER), ("AAA", 2024, 5.0)])
        table, _ = sc.build_priority_table(data, 2024)
        brief = sc.build_brief_markdown(row_for(table, "ZZZ"))
        self.assertIn("No unambiguous estimate available", brief)
        self.assertIn("Not available", brief)
        self.assertIn("Resolve conflicting country-year observations", brief)

    def test_brief_states_interpretation_limits(self):
        brief = sc.build_brief_markdown(row_for(self.table, "CAA"))
        self.assertIn("not a forecast", brief)
        self.assertIn("should not be used alone to allocate aid", brief)


REAL_DATA = ROOT / "data" / "processed" / "food_insecurity_percentages.csv"
GOLDEN = ROOT / "tests" / "fixtures" / "legacy_priority_screening.csv"
GOLDEN_HASH = ROOT / "tests" / "fixtures" / "legacy_priority_input.sha256"


def _data_matches_golden():
    return (
        REAL_DATA.exists()
        and GOLDEN.exists()
        and GOLDEN_HASH.exists()
        and hashlib.sha256(REAL_DATA.read_bytes()).hexdigest() == GOLDEN_HASH.read_text().strip()
    )


@unittest.skipUnless(_data_matches_golden(), "real data differs from the data the golden file was captured on")
class RegressionAgainstPreviousAppTests(unittest.TestCase):
    """The golden file is the priority table the previous app.py produced on this exact data."""

    @classmethod
    def setUpClass(cls):
        cls.data = fsdata.load_observations(REAL_DATA).data
        cls.order = sorted(cls.data["country_code"].unique(), key=lambda c: country_name(c).lower())
        cls.golden = pd.read_csv(GOLDEN)
        cls.latest = int(cls.data["year"].max())

    def test_improvement_off_reproduces_previous_output_exactly(self):
        table, _ = sc.build_priority_table(self.data, self.latest, stale_standing=False, country_order=self.order)
        self.assertEqual(table.reset_index(drop=True).to_csv(index=False), self.golden.to_csv(index=False))

    def test_improvement_on_changes_only_previously_generic_stale_rows(self):
        new, _ = sc.build_priority_table(self.data, self.latest, stale_standing=True, country_order=self.order)
        merged = new.merge(self.golden, on="Country code", suffixes=("_new", "_old"))
        changed = merged[merged["Priority screen_new"] != merged["Priority screen_old"]]
        self.assertGreater(len(changed), 0)
        self.assertTrue((changed["Priority screen_old"] == sc.CAT_DATA_UPDATE).all())
        self.assertTrue((changed["Priority screen_new"] == sc.CAT_STALE_HIGH).all())
        unchanged = merged[~merged["Country code"].isin(changed["Country code"])]
        self.assertTrue((unchanged["Priority screen_new"] == unchanged["Priority screen_old"]).all())

    def test_every_country_is_screened_once(self):
        new, _ = sc.build_priority_table(self.data, self.latest, country_order=self.order)
        self.assertEqual(len(new), self.data["country_code"].nunique())


if __name__ == "__main__":
    unittest.main()
