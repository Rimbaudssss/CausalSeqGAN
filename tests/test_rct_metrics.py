import math
import unittest

import numpy as np
import pandas as pd

from src.eval.rct_benchmark import FAMILY_ORDER, aggregate, global_family_contributions


class RCTAggregationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        specification = [
            ("A", "A1", "A11", 0.1, FAMILY_ORDER[0]),
            ("A", "A1", "A12", 0.3, FAMILY_ORDER[1]),
            ("A", "A2", "A21", 0.4, FAMILY_ORDER[0]),
            ("B", "B1", "B11", 0.5, FAMILY_ORDER[0]),
            ("B", "B1", "B12", 0.7, FAMILY_ORDER[0]),
            ("C", "C1", "C11", 0.9, FAMILY_ORDER[1]),
        ]
        rows = []
        for seed, offset in [(42, 0.0), (43, 0.01), (44, 0.02)]:
            for source, experiment, task, error, family in specification:
                error += offset
                rows.append(
                    dict(
                        dataset=source,
                        experiment_id=experiment,
                        task_key=task,
                        family=family,
                        seed=seed,
                        model="scm_causal",
                        label="CausalSeqGAN",
                        ate_mae=error,
                        ate_mse=error**2,
                        policy_value_error=error / 2,
                        policy_regret=error / 4,
                        sign_accuracy=1.0,
                        sign_accuracy_clear=1.0,
                        pooled_ate_abs_error=error,
                    )
                )
        cls.tasks = pd.DataFrame(rows)
        cls.results = aggregate(cls.tasks)

    def test_global_mae_balances_experiments_then_sources(self):
        by_seed = self.results["global_by_seed"].set_index("seed")
        for seed, offset in [(42, 0.0), (43, 0.01), (44, 0.02)]:
            self.assertAlmostEqual(by_seed.loc[seed, "ate_mae"], 0.6 + offset)
        source = self.results["source_by_seed"]
        source_a = source[source.dataset.eq("A") & source.seed.eq(42)].iloc[0]
        self.assertAlmostEqual(source_a.ate_mae, 0.3)
        self.assertNotAlmostEqual(
            by_seed.loc[42, "ate_mae"], self.tasks[self.tasks.seed.eq(42)].ate_mae.mean()
        )

    def test_rmse_takes_square_root_before_source_averaging(self):
        expected = []
        by_seed = self.results["global_by_seed"].set_index("seed")
        for seed, offset in [(42, 0.0), (43, 0.01), (44, 0.02)]:
            source_mse = [
                (((0.1 + offset) ** 2 + (0.3 + offset) ** 2) / 2 + (0.4 + offset) ** 2) / 2,
                ((0.5 + offset) ** 2 + (0.7 + offset) ** 2) / 2,
                (0.9 + offset) ** 2,
            ]
            rmse = sum(math.sqrt(value) for value in source_mse) / 3
            expected.append(rmse)
            self.assertAlmostEqual(by_seed.loc[seed, "ate_rmse"], rmse)
            self.assertNotAlmostEqual(rmse, math.sqrt(sum(source_mse) / 3))
        overall = self.results["overall"].iloc[0]
        self.assertAlmostEqual(overall.ate_rmse_mean, float(np.mean(expected)))
        self.assertAlmostEqual(overall.ate_rmse_sd, float(np.std(expected, ddof=1)))

    def test_family_means_use_experiments_without_source_reweighting(self):
        family = self.results["family_by_seed"]
        generic = family[family.family.eq(FAMILY_ORDER[0]) & family.seed.eq(42)].iloc[0]
        self.assertAlmostEqual(generic.ate_mae, (0.1 + 0.4 + (0.5 + 0.7) / 2) / 3)
        self.assertAlmostEqual(
            generic.ate_rmse, math.sqrt((0.1**2 + 0.4**2 + (0.5**2 + 0.7**2) / 2) / 3)
        )
        source_balanced = self.results["family_source_balanced"]
        generic_balanced = source_balanced[source_balanced.family.eq(FAMILY_ORDER[0])].iloc[0]
        self.assertAlmostEqual(generic_balanced.ate_mae_mean, (0.25 + 0.6) / 2 + 0.01)
        self.assertNotAlmostEqual(generic.ate_mae + 0.01, generic_balanced.ate_mae_mean)

    def test_seed_standard_deviation_and_family_contributions(self):
        overall = self.results["overall"].iloc[0]
        self.assertAlmostEqual(overall.ate_mae_mean, 0.61)
        self.assertAlmostEqual(overall.ate_mae_sd, 0.01)
        self.assertEqual(
            (overall.n_seeds, overall.n_tasks, overall.n_experiments, overall.n_sources),
            (3, 6, 4, 3),
        )
        contributions = global_family_contributions(self.tasks, self.results["global_by_seed"])
        reconstructed = contributions.groupby("seed").global_mae_contribution.sum()
        np.testing.assert_allclose(reconstructed.to_numpy(), [0.6, 0.61, 0.62], atol=1e-12, rtol=0)


if __name__ == "__main__":
    unittest.main()
