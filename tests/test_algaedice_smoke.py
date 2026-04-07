"""Sanity: AlgaeDICE train_step accepts (s, a, s2, r, mask)."""

from __future__ import annotations

import unittest

import torch

from park_train.algaedice import DiscreteAlgaeDICE


class TestAlgaeDICESmoke(unittest.TestCase):
    def test_train_step_runs(self) -> None:
        B, d = 32, 75
        algo = DiscreteAlgaeDICE(d, learn_alpha=False)
        s = torch.randn(B, d)
        a = torch.randint(0, 9, (B,))
        s2 = torch.randn(B, d)
        r = torch.zeros(B)
        m = torch.ones(B)
        out = algo.train_step(s, a, s2, r, m)
        self.assertIn("critic_loss", out)


if __name__ == "__main__":
    unittest.main()
