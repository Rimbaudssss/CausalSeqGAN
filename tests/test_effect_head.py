import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.model.effect_head import CausalSeqGANEffectHead, EffectHeadConfig


class EffectHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        contrasts = [
            "Control vs. treatment",
            "Adaptive vs. fixed practice",
            "Hint vs. example",
            "PTM 3 vs. PTM 5",
            "Motivational vs. standard",
            "Video vs. text feedback",
        ]
        random = np.random.default_rng(24)
        tasks, training, evaluation, priors = [], [], [], []
        for index, contrast in enumerate(contrasts):
            task = dict(
                task_key=f"task_{index}",
                experiment_id=f"experiment_{index // 2}",
                independent_measure_pair=contrast,
                dependent_measure="accuracy",
            )
            tasks.append(task)
            for split, size, rows in [("train", 64, training), ("test", 12, evaluation)]:
                for participant in range(size):
                    condition = participant % 2
                    prior_score = (participant + index) % 11 / 10
                    probability = 0.25 + 0.2 * prior_score + 0.08 * condition
                    rows.append(
                        {
                            **task,
                            "student_id": f"{split}_{participant}",
                            "condition": condition,
                            "prior_score": prior_score,
                            "y_binary": int(random.random() < probability),
                            "normalized_student_learning": float(random.random()),
                            "assignment_completed": 1,
                        }
                    )
            priors.append(
                dict(
                    task_key=task["task_key"],
                    model_value_t0=0.35 + index * 0.01,
                    model_value_t1=0.41 + index * 0.02,
                )
            )
        cls.tasks = pd.DataFrame(tasks)
        cls.training = pd.DataFrame(training)
        cls.evaluation = pd.DataFrame(evaluation)
        cls.priors = pd.DataFrame(priors)
        cls.head = CausalSeqGANEffectHead(EffectHeadConfig(inner_seeds=(17,), inner_folds=2))
        cls.head.fit(cls.tasks, cls.training, source="synthetic", outer_fold=0)

    def predict(self, evaluation=None):
        return self.head.predict(
            self.tasks,
            self.training,
            self.evaluation if evaluation is None else evaluation,
            self.priors,
        )

    def test_heldout_outcomes_do_not_change_predictions(self):
        expected = self.predict()
        replaced = self.evaluation.copy()
        replaced["y_binary"] = 1 - replaced.y_binary
        replaced["normalized_student_learning"] = 1000.0
        replaced["assignment_completed"] = 0
        pd.testing.assert_frame_equal(self.predict(replaced), expected, check_exact=True)
        without_outcomes = self.evaluation.drop(
            columns=["y_binary", "normalized_student_learning", "assignment_completed"]
        )
        pd.testing.assert_frame_equal(self.predict(without_outcomes), expected, check_exact=True)

    def test_fitted_state_roundtrip_preserves_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "effect_head.json"
            self.head.save(path)
            restored = CausalSeqGANEffectHead.load(path)
            self.assertEqual(restored.config, self.head.config)
            actual = restored.predict(self.tasks, self.training, self.evaluation, self.priors)
            pd.testing.assert_frame_equal(actual, self.predict(), check_exact=True)

    def test_changed_training_outcomes_require_refitting(self):
        changed = self.training.copy()
        changed.loc[0, "y_binary"] = 1 - changed.loc[0, "y_binary"]
        with self.assertRaisesRegex(ValueError, "Training rows differ"):
            self.head.predict(self.tasks, changed, self.evaluation, self.priors)


if __name__ == "__main__":
    unittest.main()
