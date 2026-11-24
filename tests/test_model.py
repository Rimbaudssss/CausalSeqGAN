import tempfile
import unittest
from pathlib import Path

import torch

from src.model import CausalSeqGAN, CausalSeqGANConfig


class CausalSeqGANTests(unittest.TestCase):
    def make_model(self, dynamics="gru"):
        torch.manual_seed(42)
        config = CausalSeqGANConfig(
            d_x=3,
            d_k=8,
            d_eps=2,
            a_vocab_size=5,
            a_emb_dim=3,
            t_vocab_size=2,
            t_emb_dim=2,
            mlp_hidden=12,
            dropout=0.0,
            dynamics=dynamics,
            tf_n_heads=2,
            tf_n_layers=1,
            tf_ffn_hidden=16,
        )
        return CausalSeqGAN(config).eval()

    def inputs(self):
        torch.manual_seed(7)
        return dict(
            x=torch.randn(2, 3),
            a=torch.randint(5, (2, 4)),
            t_obs=torch.zeros(2, 4, dtype=torch.long),
            eps=torch.randn(2, 4, 2),
            stochastic_y=False,
        )

    def test_checkpoint_preserves_rollout(self):
        for dynamics in ("gru", "mlp", "transformer"):
            with self.subTest(dynamics=dynamics), tempfile.TemporaryDirectory() as folder:
                model = self.make_model(dynamics)
                inputs = self.inputs()
                with torch.no_grad():
                    expected = model.rollout(**inputs)
                path = Path(folder) / "model.pt"
                model.save(path)
                restored = CausalSeqGAN.load(path)
                with torch.no_grad():
                    actual = restored.rollout(**inputs)
                self.assertEqual(model.cfg, restored.cfg)
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_intervention_preserves_shared_prefix(self):
        for dynamics in ("gru", "mlp", "transformer"):
            with self.subTest(dynamics=dynamics):
                model = self.make_model(dynamics)
                inputs = self.inputs()
                with torch.no_grad():
                    factual = model.rollout(**inputs)
                    counterfactual = model.rollout(
                        **inputs, do_t={2: torch.ones(2, dtype=torch.long)}
                    )
                torch.testing.assert_close(
                    factual["k"][:, :3], counterfactual["k"][:, :3], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    factual["y_logits"][:, :2],
                    counterfactual["y_logits"][:, :2],
                    rtol=0,
                    atol=0,
                )
                self.assertFalse(torch.equal(factual["k"][:, 3:], counterfactual["k"][:, 3:]))


if __name__ == "__main__":
    unittest.main()
