"""Population chart helpers for Park MARL."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from park_visual.population_charts import save_park_population_csv, save_park_population_png


class TestPopulationCharts(unittest.TestCase):
    def test_save_png_and_csv(self) -> None:
        preds = [10, 12, 9]
        preys = [100, 95, 102]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "out.png"
            save_park_population_png(preds, preys, p)
            self.assertTrue(p.is_file())
            c = Path(td) / "out.csv"
            save_park_population_csv(preds, preys, c)
            rows = list(csv.reader(c.read_text(encoding="utf-8").splitlines()))
            self.assertEqual(rows[0], ["outer_iteration", "n_predators", "n_prey"])
            self.assertEqual(len(rows), 4)


if __name__ == "__main__":
    unittest.main()
