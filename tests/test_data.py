"""Tests for foodshield.data and foodshield.countries.

Run from the project root:
    python -m unittest discover -s tests -v

The synthetic fixtures below are made-up numbers used ONLY to test the logic.
The final class checks invariants of the real cleaned CSV when it is present.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foodshield import data as fs  # noqa: E402
from foodshield.countries import COUNTRY_NAMES, country_label, country_name  # noqa: E402

FAO = "Food and Agriculture Organization of the United Nations (FAO)"
NATIONAL = "National Entity"


def row(code, year, value, source=FAO, status="E", window=None):
    """One synthetic raw row; window defaults to the 3-year window centred on year."""
    if window is None:
        window = f"{year - 1}-{year + 1}"
    return {
        "country_code": code,
        "year": year,
        "food_insecurity_percent": value,
        "observation_status": status,
        "unit_measure": "PT",
        "time_detail": window,
        "source": source,
        "indicator": "2.1.2",
    }


def build(rows):
    return fs.build_observations(pd.DataFrame(rows)).data


class CountryNameTests(unittest.TestCase):
    def test_known_code_gets_name_and_label_keeps_iso(self):
        self.assertEqual(country_name("dza"), "Algeria")
        self.assertEqual(country_label("DZA"), "Algeria (DZA)")

    def test_unknown_code_falls_back_to_code(self):
        self.assertEqual(country_name("ZZZ"), "ZZZ")
        self.assertEqual(country_label("ZZZ"), "ZZZ (ZZZ)")

    def test_codes_are_three_capital_letters(self):
        for code in COUNTRY_NAMES:
            self.assertRegex(code, r"^[A-Z]{3}$")


class LoadingAndValidationTests(unittest.TestCase):
    def test_every_input_row_is_kept_or_rejected_with_reason(self):
        raw = pd.DataFrame(
            [
                row("AAA", 2020, 10.0),
                row("bbb", 2020, 20.0),  # lower-case code is normalised, not rejected
                row("XX", 2020, 30.0),  # bad code
                row("CCC", 1800, 30.0),  # bad year
                row("DDD", 2020, 120.0),  # out of range
                row("EEE", 2020, "n/a"),  # not numeric
                row("FFF", 2020.5, 5.0),  # non-integer year
            ]
        )
        loaded = fs.build_observations(raw)
        self.assertEqual(len(loaded.data) + len(loaded.rejected), len(raw))
        self.assertEqual(set(loaded.data["country_code"]), {"AAA", "BBB"})
        self.assertEqual(len(loaded.rejected), 5)
        self.assertTrue((loaded.rejected["reject_reason"] != "").all())
        self.assertIn("file_line", loaded.rejected.columns)

    def test_missing_required_column_raises(self):
        raw = pd.DataFrame([{"country_code": "AAA", "year": 2020}])
        with self.assertRaises(ValueError):
            fs.build_observations(raw)

    def test_missing_metadata_columns_are_reported_not_invented(self):
        raw = pd.DataFrame([{"country_code": "AAA", "year": 2020, "food_insecurity_percent": 5}])
        loaded = fs.build_observations(raw)
        self.assertIn("time_detail", loaded.missing_metadata)
        self.assertEqual(loaded.data.loc[0, "window_label"], fs.NOT_PROVIDED)
        self.assertEqual(loaded.data.loc[0, "source"], fs.NOT_PROVIDED)

    def test_names_added_and_iso_code_preserved(self):
        data = build([row("DZA", 2020, 10.0), row("ZZZ", 2020, 11.0)])
        names = dict(zip(data["country_code"], data["country_name"]))
        self.assertEqual(names["DZA"], "Algeria")
        self.assertEqual(names["ZZZ"], "ZZZ")


class ReferenceWindowTests(unittest.TestCase):
    def test_window_parsed_from_time_detail(self):
        data = build([row("AAA", 2024, 10.0, window="2023-2025")])
        record = data.iloc[0]
        self.assertEqual(record["window_start"], 2023)
        self.assertEqual(record["window_end"], 2025)
        self.assertEqual(record["window_years"], 3)
        self.assertEqual(record["window_label"], "2023\u20132025")
        self.assertTrue(record["window_centered"])

    def test_en_dash_and_spaces_are_accepted(self):
        data = build([row("AAA", 2020, 10.0, window=" 2019 \u2013 2021 ")])
        self.assertEqual(data.iloc[0]["window_label"], "2019\u20132021")

    def test_unparseable_window_is_not_guessed(self):
        data = build([row("AAA", 2020, 10.0, window="around 2020")])
        self.assertEqual(data.iloc[0]["window_label"], fs.NOT_PROVIDED)
        self.assertTrue(np.isnan(data.iloc[0]["window_start"]))
        self.assertFalse(data.iloc[0]["window_centered"])

    def test_example_uses_latest_year_with_a_window(self):
        data = build([row("AAA", 2023, 10.0), row("AAA", 2024, 11.0)])
        self.assertEqual(fs.window_example(data), (2024, "2023\u20132025"))

    def test_example_none_when_no_windows(self):
        data = build([row("AAA", 2023, 10.0, window="n/a")])
        self.assertIsNone(fs.window_example(data))


class DuplicateHandlingTests(unittest.TestCase):
    def setUp(self):
        self.data = build(
            [
                row("AAA", 2020, 10.0, source=FAO),
                row("AAA", 2020, 18.0, source=NATIONAL),  # conflict
                row("BBB", 2020, 5.0, source=FAO),
                row("BBB", 2020, 5.0, source=NATIONAL),  # same value, other source
                row("CCC", 2020, 7.0),
            ]
        )

    def test_flags_do_not_remove_or_change_rows(self):
        self.assertEqual(len(self.data), 5)
        aaa = self.data[self.data["country_code"] == "AAA"]
        self.assertEqual(sorted(aaa["food_insecurity_percent"]), [10.0, 18.0])
        self.assertTrue(aaa["conflict"].all())
        self.assertTrue((aaa["record_status"] == fs.RECORD_CONFLICT).all())

    def test_same_value_repeat_is_not_a_conflict(self):
        bbb = self.data[self.data["country_code"] == "BBB"]
        self.assertFalse(bbb["conflict"].any())
        self.assertTrue((bbb["record_status"] == fs.RECORD_REPEAT).all())

    def test_comparison_withholds_conflicts_and_never_averages(self):
        comparable, withheld = fs.observations_for_year(self.data, 2020)
        self.assertNotIn("AAA", set(comparable["country_code"]))
        self.assertEqual(len(withheld), 2)
        self.assertEqual(sorted(withheld["food_insecurity_percent"]), [10.0, 18.0])
        # 14.0 would be the mean of the conflict; it must not appear anywhere.
        self.assertNotIn(14.0, set(comparable["food_insecurity_percent"]))

    def test_same_value_repeat_appears_once_with_unchanged_value(self):
        comparable, _ = fs.observations_for_year(self.data, 2020)
        bbb = comparable[comparable["country_code"] == "BBB"]
        self.assertEqual(len(bbb), 1)
        self.assertEqual(bbb.iloc[0]["food_insecurity_percent"], 5.0)

    def test_empty_input_does_not_crash(self):
        empty = fs.build_observations(
            pd.DataFrame(columns=["country_code", "year", "food_insecurity_percent"])
        ).data
        self.assertTrue(empty.empty)
        self.assertTrue(fs.coverage_summary(empty).empty)


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.data = build(
            [row("FUL", y, 10.0 + y - 2018) for y in range(2018, 2023)]
            + [
                row("OLD", 2018, 30.0),
                row("OLD", 2019, 31.0),
                row("GAP", 2018, 20.0),
                row("GAP", 2022, 25.0),
                row("CON", 2022, 1.0),
                row("CON", 2022, 2.0),
            ]
        )
        self.coverage = fs.coverage_summary(self.data).set_index("country_code")

    def test_latest_year_and_currentness(self):
        self.assertEqual(self.coverage.loc["FUL", "latest_year"], 2022)
        self.assertTrue(self.coverage.loc["FUL", "latest_is_current"])
        self.assertEqual(self.coverage.loc["OLD", "latest_year"], 2019)
        self.assertFalse(self.coverage.loc["OLD", "latest_is_current"])
        self.assertEqual(self.coverage.loc["OLD", "latest_window"], "2018\u20132020")

    def test_observation_count_and_gap_years(self):
        self.assertEqual(self.coverage.loc["FUL", "n_years_observed"], 5)
        self.assertEqual(self.coverage.loc["FUL", "gap_years"], 0)
        # dataset years are 2018-2022; GAP has 2018 and 2022 only -> 3 missing inside span
        self.assertEqual(self.coverage.loc["GAP", "gap_years"], 3)
        self.assertAlmostEqual(self.coverage.loc["GAP", "coverage_share"], 2 / 5)

    def test_conflicting_latest_year_has_no_latest_value(self):
        self.assertTrue(self.coverage.loc["CON", "latest_year_conflict"])
        self.assertTrue(np.isnan(self.coverage.loc["CON", "latest_value"]))
        self.assertEqual(self.coverage.loc["CON", "n_conflict_years"], 1)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.data = build(
            [
                # observed in both 2018 and 2022
                row("AAA", 2018, 10.0),
                row("AAA", 2022, 16.0),
                # observed 2018 and 2019 only (different coverage period)
                row("BBB", 2018, 40.0),
                row("BBB", 2019, 50.0),
                # starts in 2019, so no 2018 value
                row("CCC", 2019, 5.0),
                row("CCC", 2022, 4.0),
                # conflict in the end year
                row("DDD", 2018, 8.0),
                row("DDD", 2022, 9.0),
                row("DDD", 2022, 12.0, source=NATIONAL),
                # source switch between the two years
                row("EEE", 2018, 20.0, source=FAO),
                row("EEE", 2022, 20.0, source=NATIONAL),
            ]
        )
        self.eligible, self.excluded = fs.compute_changes(self.data, 2018, 2022)

    def test_only_countries_with_both_years_are_compared(self):
        self.assertEqual(set(self.eligible["country_code"]), {"AAA", "EEE"})

    def test_different_coverage_is_excluded_with_reason(self):
        reasons = dict(zip(self.excluded["country_code"], self.excluded["reason"]))
        self.assertEqual(reasons["BBB"], "No observation for 2022")
        self.assertEqual(reasons["CCC"], "No observation for 2018")
        self.assertEqual(reasons["DDD"], "Conflicting records in 2022")

    def test_every_country_is_eligible_or_excluded_exactly_once(self):
        all_codes = set(self.data["country_code"])
        eligible = set(self.eligible["country_code"])
        excluded = set(self.excluded["country_code"])
        self.assertEqual(eligible | excluded, all_codes)
        self.assertEqual(eligible & excluded, set())

    def test_change_values_and_direction(self):
        aaa = self.eligible.set_index("country_code").loc["AAA"]
        self.assertEqual(aaa["value_start"], 10.0)
        self.assertEqual(aaa["value_end"], 16.0)
        self.assertAlmostEqual(aaa["change_pp"], 6.0)
        self.assertEqual(aaa["direction"], "Increase")
        eee = self.eligible.set_index("country_code").loc["EEE"]
        self.assertEqual(eee["direction"], "No change")

    def test_source_change_is_flagged(self):
        flags = self.eligible.set_index("country_code")["source_changed"]
        self.assertFalse(flags["AAA"])
        self.assertTrue(flags["EEE"])

    def test_window_overlap_between_adjacent_years(self):
        data = build([row("AAA", 2018, 1.0), row("AAA", 2019, 2.0), row("AAA", 2021, 3.0)])
        adjacent, _ = fs.compute_changes(data, 2018, 2019)
        self.assertEqual(adjacent.iloc[0]["windows_overlap_years"], 2)
        apart, _ = fs.compute_changes(data, 2018, 2021)
        self.assertEqual(apart.iloc[0]["windows_overlap_years"], 0)

    def test_start_must_precede_end(self):
        with self.assertRaises(ValueError):
            fs.compute_changes(self.data, 2022, 2018)
        with self.assertRaises(ValueError):
            fs.compute_changes(self.data, 2018, 2018)


class SelectChangesTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for i, delta in enumerate([5, -3, 0, 12, -8, 1]):
            code = "AB" + "CDEFGH"[i]
            rows += [row(code, 2018, 20.0), row(code, 2022, 20.0 + delta)]
        self.eligible, _ = fs.compute_changes(build(rows), 2018, 2022)

    def test_all_returns_every_eligible_country(self):
        picked = fs.select_changes(self.eligible, "all", limit=2)
        self.assertEqual(len(picked), len(self.eligible))
        self.assertEqual(set(picked["country_code"]), set(self.eligible["country_code"]))

    def test_increases_are_largest_first_and_limited(self):
        picked = fs.select_changes(self.eligible, "increases", 3)
        self.assertEqual(list(picked["change_pp"]), [12, 5, 1])

    def test_decreases_are_most_negative_first(self):
        picked = fs.select_changes(self.eligible, "decreases", 2)
        self.assertEqual(list(picked["change_pp"]), [-8, -3])

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            fs.select_changes(self.eligible, "median")


class CountryChangeTests(unittest.TestCase):
    def test_uses_first_and_last_unambiguous_years(self):
        data = build(
            [
                row("AAA", 2018, 10.0),
                row("AAA", 2020, 99.0),
                row("AAA", 2020, 1.0, source=NATIONAL),  # conflict, must be ignored
                row("AAA", 2022, 16.0),
            ]
        )
        change = fs.country_change(data)
        self.assertEqual((change["start_year"], change["end_year"]), (2018, 2022))
        self.assertAlmostEqual(change["change_pp"], 6.0)

    def test_none_with_fewer_than_two_unambiguous_years(self):
        self.assertIsNone(fs.country_change(build([row("AAA", 2018, 10.0)])))


class TableHelperTests(unittest.TestCase):
    def setUp(self):
        self.data = build([row("DZA", 2024, 10.0, window="2023-2025")])

    def test_display_table_has_name_iso_and_window(self):
        table = fs.to_display_table(self.data)
        self.assertEqual(table.loc[0, "Country"], "Algeria")
        self.assertEqual(table.loc[0, "ISO code"], "DZA")
        self.assertEqual(table.loc[0, "Reference window"], "2023\u20132025")

    def test_export_keeps_iso_name_and_metadata(self):
        export = fs.to_export_frame(self.data)
        for col in ("country_code", "country_name", "time_detail", "source", "observation_status"):
            self.assertIn(col, export.columns)
        self.assertEqual(export.loc[0, "time_detail"], "2023-2025")  # original text kept


class CleaningSummaryTests(unittest.TestCase):
    def write(self, text):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "cleaning_summary.csv"
        path.write_text(text, encoding="utf-8")
        return path

    SUMMARY = (
        "metric,value\n"
        "original_rows,5000\n"
        "valid_rows_before_deduplication,1300\n"
        "invalid_rows,4\n"
        "exact_duplicate_rows_removed,6\n"
        "conflicting_country_year_rows,10\n"
        "clean_rows_saved,1280\n"
        "countries_saved,157\n"
        "first_year,2015\n"
    )

    def test_missing_file_returns_none(self):
        self.assertIsNone(fs.load_cleaning_summary(Path("/no/such/cleaning_summary.csv")))

    def test_malformed_file_raises(self):
        with self.assertRaises(ValueError):
            fs.load_cleaning_summary(self.write("a,b\n1,2\n"))

    def test_values_and_labels(self):
        summary = fs.load_cleaning_summary(self.write(self.SUMMARY))
        self.assertEqual(fs.summary_value(summary, "invalid_rows"), 4)
        self.assertIsNone(fs.summary_value(summary, "not_a_metric"))
        shown = fs.summary_for_display(summary)
        self.assertIn("Rows in the dashboard file", set(shown["Measure"]))

    def test_unknown_metric_is_shown_unchanged(self):
        summary = fs.load_cleaning_summary(self.write("metric,value\nnew_metric,3\n"))
        self.assertEqual(fs.summary_for_display(summary).loc[0, "Measure"], "new_metric")

    def test_reconciliation_reveals_unlogged_collapsed_rows(self):
        summary = fs.load_cleaning_summary(self.write(self.SUMMARY))
        # 1300 - 6 - 10 - 1280 = 4 rows the pipeline summary does not explain
        self.assertEqual(fs.reconcile_cleaning_counts(summary), 4)

    def test_reconciliation_none_when_metrics_missing(self):
        summary = fs.load_cleaning_summary(self.write("metric,value\ninvalid_rows,1\n"))
        self.assertIsNone(fs.reconcile_cleaning_counts(summary))

    def test_consistency_warnings(self):
        summary = fs.load_cleaning_summary(self.write(self.SUMMARY))
        self.assertEqual(fs.check_summary_consistency(summary, 1280, 157), [])
        stale = fs.check_summary_consistency(summary, 1200, 150)
        self.assertEqual(len(stale), 2)

    def test_impossible_counts_are_flagged(self):
        broken = self.SUMMARY.replace("clean_rows_saved,1280", "clean_rows_saved,1299")
        summary = fs.load_cleaning_summary(self.write(broken))
        messages = fs.check_summary_consistency(summary, 1299, 157)
        self.assertTrue(any("do not add up" in m for m in messages))


REAL_FILE = ROOT / "data" / "processed" / "food_insecurity_percentages.csv"


@unittest.skipUnless(REAL_FILE.exists(), "cleaned FAO file not found")
class RealDatasetInvariantTests(unittest.TestCase):
    """Properties of the shipped file. No counts are hard-coded, so these keep
    passing after the pipeline is re-run on a newer FAO export."""

    @classmethod
    def setUpClass(cls):
        cls.loaded = fs.load_observations(REAL_FILE)
        cls.data = cls.loaded.data

    def test_no_rows_rejected_at_load(self):
        self.assertEqual(len(self.loaded.rejected), 0)

    def test_all_countries_have_readable_names(self):
        unmapped = sorted(set(self.data["country_code"]) - set(COUNTRY_NAMES))
        self.assertEqual(unmapped, [])

    def test_windows_are_three_years_and_centred_on_the_reference_year(self):
        self.assertFalse((self.data["window_label"] == fs.NOT_PROVIDED).any())
        self.assertTrue((self.data["window_years"] == 3).all())
        self.assertTrue(self.data["window_centered"].all())

    def test_no_unresolved_conflicts_in_dashboard_file(self):
        self.assertEqual(int(self.data["conflict"].sum()), 0)

    def test_coverage_and_comparison_are_consistent(self):
        coverage = fs.coverage_summary(self.data)
        self.assertEqual(len(coverage), self.data["country_code"].nunique())
        years = sorted(self.data["year"].unique())
        eligible, excluded = fs.compute_changes(self.data, int(years[0]), int(years[-1]))
        self.assertEqual(len(eligible) + len(excluded), self.data["country_code"].nunique())


if __name__ == "__main__":
    unittest.main()
