"""Tests for foodshield.evaluation. Synthetic data only, except the final class
which checks structural invariants on the real data file (no numeric results are
hard-coded)."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foodshield import data as fsdata  # noqa: E402
from foodshield import evaluation as ev  # noqa: E402
from foodshield.screening import window_overlap_years  # noqa: E402


def obs(rows):
    records = [
        {
            "country_code": code,
            "year": year,
            "food_insecurity_percent": value,
            "time_detail": f"{year - 1}-{year + 1}",
            "source": src if src else "FAO",
            "observation_status": "E",
            "unit_measure": "PT",
            "indicator": "2.1.2",
        }
        for code, year, value, *rest in rows
        for src in [rest[0] if rest else None]
    ]
    return fsdata.build_observations(pd.DataFrame(records)).data


def eight_countries(years, value_of):
    codes = [f"CA{chr(65 + i)}" for i in range(8)]
    return [(c, y, value_of(i, y)) for i, c in enumerate(codes) for y in years]


class WilsonTests(unittest.TestCase):
    def test_zero_n_is_nan(self):
        low, high = ev.wilson_interval(0, 0)
        self.assertTrue(np.isnan(low) and np.isnan(high))

    def test_interval_contains_estimate_and_stays_in_unit_range(self):
        for k, n in [(0, 10), (10, 10), (5, 10), (92, 100)]:
            low, high = ev.wilson_interval(k, n)
            self.assertGreaterEqual(low, 0.0)
            self.assertLessEqual(high, 1.0)
            self.assertLessEqual(low, k / n + 1e-12)
            self.assertGreaterEqual(high, k / n - 1e-12)

    def test_known_value(self):
        low, high = ev.wilson_interval(5, 10)
        self.assertAlmostEqual(low, 0.2366, places=3)
        self.assertAlmostEqual(high, 0.7634, places=3)


class StandingTests(unittest.TestCase):
    def test_cutoff_is_75th_percentile_and_high_flag(self):
        data = obs([(f"CA{chr(65 + i)}", 2020, 10.0 * (i + 1)) for i in range(8)])
        standing = ev.yearly_standing(data)
        self.assertEqual(set(standing["year"]), {2020})
        self.assertAlmostEqual(standing["cutoff"].iloc[0], 62.5)
        self.assertEqual(int(standing["high"].sum()), 2)  # 70 and 80

    def test_years_with_fewer_than_four_peers_are_skipped(self):
        data = obs([("AAA", 2020, 1.0), ("BBB", 2020, 2.0), ("CCC", 2020, 3.0)])
        self.assertTrue(ev.yearly_standing(data).empty)

    def test_conflicting_country_year_is_excluded_not_averaged(self):
        rows = [(f"CA{chr(65 + i)}", 2020, 10.0 * (i + 1)) for i in range(8)]
        rows.append(("CAA", 2020, 55.0, "Other"))  # conflicts with CAA's 10.0
        standing = ev.yearly_standing(obs(rows))
        self.assertNotIn("CAA", set(standing["country_code"]))
        self.assertEqual(len(standing), 7)
        self.assertNotIn(32.5, set(standing["value"]))  # the mean of 10 and 55


class PairTests(unittest.TestCase):
    def setUp(self):
        rows = eight_countries([2018, 2021], lambda i, y: 10.0 * (i + 1))
        rows += [("LST", 2018, 100.0)]  # observed at origin, never again
        self.standing = ev.yearly_standing(obs(rows))
        self.pairs = ev.build_pairs(self.standing, horizons=(3,))

    def test_pairs_only_for_countries_observed_in_both_years(self):
        self.assertEqual(len(self.pairs), 8)
        self.assertNotIn("LST", set(self.pairs["country_code"]))
        self.assertTrue((self.pairs["target"] - self.pairs["origin"] == 3).all())

    def test_cross_year_flag_uses_target_cutoff(self):
        self.assertIn("high_origin_vs_target_cutoff", self.pairs.columns)

    def test_horizon_three_windows_do_not_overlap(self):
        for _, r in self.pairs.iterrows():
            self.assertEqual(
                window_overlap_years(f"{r['origin'] - 1}-{r['origin'] + 1}", f"{r['target'] - 1}-{r['target'] + 1}"), 0
            )

    def test_attrition_counts_lost_countries(self):
        attrition = ev.attrition_diagnostic(self.standing, 3)
        row = attrition[attrition["origin"] == 2018].iloc[0]
        self.assertEqual(int(row["lost"]), 1)
        self.assertEqual(int(row["high_lost_count"]), 1)  # LST had the top value
        self.assertEqual(int(row["retained"]), 8)


class ClassificationTests(unittest.TestCase):
    def test_confusion_counts_and_rates(self):
        pairs = pd.DataFrame(
            {
                "high_origin": [True, True, True, False, False, False, False, False],
                "high_target": [True, True, False, True, False, False, False, False],
            }
        )
        s = ev.classification_summary(pairs, "high_origin")
        self.assertEqual((s["true_pos"], s["false_pos"], s["false_neg"], s["true_neg"]), (2, 1, 1, 4))
        self.assertAlmostEqual(s["precision"], 2 / 3)
        self.assertAlmostEqual(s["recall"], 2 / 3)
        self.assertAlmostEqual(s["base_rate"], 3 / 8)
        self.assertAlmostEqual(s["lift_over_base_rate"], (2 / 3) / (3 / 8))

    def test_no_flags_gives_nan_precision_and_zero_recall(self):
        pairs = pd.DataFrame({"high_origin": [False, False], "high_target": [True, False]})
        s = ev.classification_summary(pairs, "high_origin")
        self.assertTrue(np.isnan(s["precision"]))
        self.assertEqual(s["recall"], 0.0)

    def test_perfect_persistence_gives_perfect_scores(self):
        data = obs(eight_countries([2018, 2021], lambda i, y: 10.0 * (i + 1)))
        result = ev.run_evaluation(data, horizons=(3,), n_boot=50, seed=1)
        new = result["by_horizon"].query("rule == 'own-year standing (new rule)'").iloc[0]
        self.assertEqual(new["precision"], 1.0)
        self.assertEqual(new["recall"], 1.0)
        self.assertAlmostEqual(new["lift_over_base_rate"], 1 / new["base_rate"])

    def test_complete_reshuffle_is_not_rewarded(self):
        # ranks reverse between origin and target, so nobody high at T is high at T+3
        data = obs(
            eight_countries([2018], lambda i, y: 10.0 * (i + 1))
            + eight_countries([2021], lambda i, y: 10.0 * (8 - i))
        )
        result = ev.run_evaluation(data, horizons=(3,), n_boot=50, seed=1)
        new = result["by_horizon"].query("rule == 'own-year standing (new rule)'").iloc[0]
        self.assertEqual(new["precision"], 0.0)


class DeterminismTests(unittest.TestCase):
    def test_same_seed_gives_identical_results(self):
        rng = np.random.default_rng(0)
        rows = []
        for i in range(30):
            base = float(rng.uniform(5, 80))
            for y in (2018, 2021, 2024):
                rows.append((f"K{chr(65 + i // 26)}{chr(65 + i % 26)}", y, float(np.clip(base + rng.normal(0, 3), 0, 100))))
        data = obs(rows)
        a = ev.run_evaluation(data, horizons=(3,), n_boot=200, seed=7)
        b = ev.run_evaluation(data, horizons=(3,), n_boot=200, seed=7)
        pd.testing.assert_frame_equal(a["primary_bootstrap"], b["primary_bootstrap"])
        pd.testing.assert_frame_equal(a["by_horizon"], b["by_horizon"])

    def test_bootstrap_interval_brackets_point_estimate_for_typical_data(self):
        rng = np.random.default_rng(1)
        rows = []
        for i in range(40):
            base = float(rng.uniform(5, 80))
            for y in (2018, 2021):
                rows.append((f"K{chr(65 + i // 26)}{chr(65 + i % 26)}", y, float(np.clip(base + rng.normal(0, 4), 0, 100))))
        data = obs(rows)
        result = ev.run_evaluation(data, horizons=(3,), n_boot=300, seed=3)
        point = result["by_horizon"].query("rule == 'own-year standing (new rule)'").iloc[0]["precision"]
        boot = result["primary_bootstrap"].iloc[0]
        self.assertLessEqual(boot["precision_boot_low"], point + 1e-9)
        self.assertGreaterEqual(boot["precision_boot_high"], point - 1e-9)


REAL_DATA = ROOT / "data" / "processed" / "food_insecurity_percentages.csv"


@unittest.skipUnless(REAL_DATA.exists(), "processed data file not found")
class RealDataStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.observations = fsdata.load_observations(REAL_DATA).data
        cls.result = ev.run_evaluation(cls.observations, n_boot=50, seed=1)

    def test_primary_horizon_pairs_never_have_overlapping_windows(self):
        windows = (
            self.observations.drop_duplicates("year").set_index("year")["window_label"].to_dict()
        )
        primary = self.result["pairs"].query("horizon == 3")
        self.assertGreater(len(primary), 0)
        for _, r in primary.drop_duplicates(["origin", "target"]).iterrows():
            self.assertEqual(window_overlap_years(windows[r["origin"]], windows[r["target"]]), 0)

    def test_rates_are_valid_probabilities(self):
        by_h = self.result["by_horizon"]
        for column in ("precision", "recall", "base_rate"):
            values = by_h[column].dropna()
            self.assertTrue(((values >= 0) & (values <= 1)).all(), column)

    def test_pair_counts_match_confusion_matrix(self):
        by_h = self.result["by_horizon"]
        sums = by_h["true_pos"] + by_h["false_pos"] + by_h["false_neg"] + by_h["true_neg"]
        self.assertTrue((sums == by_h["pairs"]).all())


if __name__ == "__main__":
    unittest.main()
