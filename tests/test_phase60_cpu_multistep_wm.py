"""Deterministic Phase60 invariants; no separate smoke job required."""
import unittest
from scripts.phase60_cpu_multistep_wm import CategoricalMixtureWM, simulate, split_train, select_models

ITEMS = tuple(f"visible_{i}" for i in range(8))

def people(n=240):
    return [{"pid": f"synthetic_{i:04d}", "label": int(i % 3 == 0),
             "features": {name: int((i + j + i // 7) % 9 == 0)
                          for j, name in enumerate(ITEMS)}}
            for i in range(n)]

class Phase60Tests(unittest.TestCase):
    def test_probability_normalization(self):
        fit, dev, outer = split_train(people())
        model = CategoricalMixtureWM(ITEMS, components=2, epochs=3).fit(fit)
        state = model.initial_state()
        self.assertAlmostEqual(sum(state), 1)
        self.assertAlmostEqual(sum(model.predict(state, 1)), 1)
        self.assertAlmostEqual(sum(model.imagined_transition(state, 0, 4)), 1)

    def test_real_vs_imagined(self):
        fit, dev, outer = split_train(people())
        model = CategoricalMixtureWM(ITEMS, epochs=2).fit(fit)
        for lookahead in (False, True):
            risk, nll = simulate(outer[0], model, 3, lookahead)
            self.assertTrue(0 <= risk <= 1)
            self.assertGreaterEqual(nll, 0)

    def test_dev_selection(self):
        fit, dev, outer = split_train(people())
        naive, mixture, k, scores = select_models(fit, dev, ITEMS, 2)
        self.assertIn(k, (1, 2, 3))
        self.assertEqual(len(scores), 3)
