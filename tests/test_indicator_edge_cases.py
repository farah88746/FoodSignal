import unittest
import pandas as pd

from foodshield.data import (
    build_observations,
    compute_changes,
    country_change,
    coverage_summary,
    observations_for_year,
)

def raw_rows(rows):
    return pd.DataFrame(rows, columns=[
        "country_code", "year", "food_insecurity_percent", "time_detail", "source"
    ])

class IndicatorEdgeCaseTests(unittest.TestCase):
    def test_old_latest_year_is_not_marked_current(self):
        raw = raw_rows([
            ("LCA", 2015, 20.0, "2014-2016", "FAO"),
            ("LCA", 2017, 25.0, "2016-2018", "FAO"),
            ("DZA", 2023, 10.0, "2022-2024", "FAO"),
            ("DZA", 2024, 11.0, "2023-2025", "FAO"),
        ])
        loaded = build_observations(raw)
        coverage = coverage_summary(loaded.data).set_index("country_code")
        self.assertFalse(bool(coverage.loc["LCA", "latest_is_current"]))
        self.assertTrue(bool(coverage.loc["DZA", "latest_is_current"]))
        self.assertEqual(int(coverage.loc["LCA", "latest_year"]), 2017)

    def test_comparison_excludes_country_missing_one_reference_year(self):
        raw = raw_rows([
            ("LCA", 2017, 25.0, "2016-2018", "FAO"),
            ("DZA", 2017, 10.0, "2016-2018", "FAO"),
            ("DZA", 2024, 11.0, "2023-2025", "FAO"),
        ])
        loaded = build_observations(raw)
        eligible, excluded = compute_changes(loaded.data, 2017, 2024)
        self.assertEqual(eligible["country_code"].tolist(), ["DZA"])
        excluded_lca = excluded.loc[excluded["country_code"] == "LCA", "reason"].iloc[0]
        self.assertIn("no observation for 2024", excluded_lca.lower())

    def test_conflicting_year_is_withheld_not_averaged(self):
        raw = raw_rows([
            ("DZA", 2024, 11.0, "2023-2025", "FAO"),
            ("DZA", 2024, 13.0, "2023-2025", "Other"),
        ])
        loaded = build_observations(raw)
        comparable, withheld = observations_for_year(loaded.data, 2024)
        self.assertTrue(comparable.empty)
        self.assertEqual(len(withheld), 2)

    def test_country_change_reports_actual_first_and_last_year(self):
        raw = raw_rows([
            ("LCA", 2015, 20.0, "2014-2016", "FAO"),
            ("LCA", 2017, 25.0, "2016-2018", "FAO"),
        ])
        loaded = build_observations(raw)
        change = country_change(loaded.data[loaded.data["country_code"] == "LCA"])
        self.assertIsNotNone(change)
        self.assertEqual(int(change["start_year"]), 2015)
        self.assertEqual(int(change["end_year"]), 2017)
        self.assertEqual(float(change["change_pp"]), 5.0)

    def test_one_unambiguous_year_has_no_country_change(self):
        raw = raw_rows([
            ("LCA", 2017, 25.0, "2016-2018", "FAO"),
            ("LCA", 2019, 20.0, "2018-2020", "FAO"),
            ("LCA", 2019, 22.0, "2018-2020", "Other"),
        ])
        loaded = build_observations(raw)
        change = country_change(loaded.data[loaded.data["country_code"] == "LCA"])
        self.assertIsNone(change)

if __name__ == "__main__":
    unittest.main()
