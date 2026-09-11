"""Scientific interface checks and parity with the published fitted models."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from binding_dynamics.clean_tpr import CleanTPR, clean_inputs
from binding_dynamics.episodes import generate_episode
from binding_dynamics.evaluation import make_probe_inputs, model_edits, variant_data
from binding_dynamics.features import history_record


class TPRContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        with np.load(ROOT / "tests/fixture.npz", allow_pickle=False) as data:
            cls.fixture = dict(data)
        cls.data = {key: cls.fixture[key] for key in ["labels", "positions", "mask", "q", "states", "scene_ids", "edit_nodes"]}

    def model(self, activation):
        return CleanTPR.load(ROOT / "models" / f"f26r16_{activation}_s51.pt")

    def test_frozen_forward_and_edit_parity(self):
        factual = variant_data(self.data, 0)
        for label, activation in [("linear", "linear"), ("tanh", "tanh")]:
            model = self.model(activation)
            np.testing.assert_allclose(model.predict(factual), self.fixture[label + "_predictions"], atol=2e-5, rtol=2e-4)
            np.testing.assert_allclose(model_edits(model, self.data, "clipped_difference"),
                                       self.fixture[label + "_edited_states"], atol=3e-5, rtol=2e-4)

    def test_tanh_is_after_complete_affine_sum(self):
        model = self.model("tanh")
        data = variant_data(self.data, 0)
        z = model.predict(data, preactivation=True)
        np.testing.assert_allclose(model.predict(data), np.tanh(z), atol=2e-7)
        z_tensor = torch.tensor(z)
        torch.testing.assert_close(torch.tanh(z_tensor), 2*torch.sigmoid(2*z_tensor)-1, atol=3e-7, rtol=1e-5)

    def test_linear_edit_equals_one_binding_difference(self):
        model = self.model("linear")
        factual, changed = [variant_data(self.data, i) for i in [0, 1]]
        x = clean_inputs(factual, "cpu")
        nodes = self.data["edit_nodes"][:, 0]
        row = torch.arange(len(nodes))
        with torch.no_grad():
            old = model.fillers(torch.tensor(factual["labels"][np.arange(len(nodes)), nodes]))
            new = model.fillers(torch.tensor(changed["labels"][np.arange(len(nodes)), nodes]))
            role = model.roles(x["phi"])[row, nodes]
            formula = model.output(((new-old)[:, :, None] * role[:, None, :]).flatten(1)).numpy()
        np.testing.assert_allclose(model.predict(changed)-model.predict(factual), formula, atol=4e-5, rtol=2e-4)

    def test_noop_and_bounds(self):
        factual = variant_data(self.data, 0)
        for name in ["linear", "tanh"]:
            model = self.model(name)
            np.testing.assert_array_equal(model.edit(factual["states"], factual, factual, "clipped_difference"), factual["states"])
            self.assertLessEqual(np.abs(model_edits(model, self.data, "clipped_difference")).max(), 1)

    def test_answer_and_history_extras_are_not_inputs(self):
        model = self.model("tanh")
        factual = variant_data(self.data, 0)
        poisoned = {**factual, "query": np.full((len(factual["labels"]), 26), np.nan), "ages": None, "counts": None, "targets": None}
        np.testing.assert_array_equal(model.predict(factual), model.predict(poisoned))

    def test_prefix_contains_no_unobserved_label(self):
        episode = generate_episode(5, 413001, unique_labels=True)
        unobserved = np.setdiff1d(np.arange(6), episode.visits[:2])
        labels = episode.labels.copy()
        labels[unobserved] = (labels[unobserved]+7) % 26
        a, b = history_record(episode, 1), history_record(replace(episode, labels=labels), 1)
        for key in ["labels", "positions", "mask", "q"]:
            np.testing.assert_array_equal(a[key], b[key])

    def test_probe_regeneration_matches_archive_and_never_refreshes(self):
        probe = make_probe_inputs(self.data, 413003, 5)
        with np.load(ROOT / "results/evidence.npz", allow_pickle=False) as evidence:
            for key in probe:
                np.testing.assert_array_equal(probe[key], evidence["probe_" + key][:8])
        for scene, pair in enumerate(self.data["edit_nodes"]):
            for variant in range(2):
                for qindex, query in enumerate(probe["queries"][scene]):
                    blocked = [*pair[:variant+1], query]
                    self.assertFalse(np.isin(probe["nodes"][scene, variant, qindex], blocked).any())
        self.assertEqual(probe["labels"].shape, (8, 2, 5, 6))

    def test_swap_preserves_inventory_and_all_variants_preserve_context(self):
        np.testing.assert_array_equal(np.sort(self.data["labels"][:, 0], axis=-1), np.sort(self.data["labels"][:, 2], axis=-1))
        for key in ["q", "positions", "mask"]:
            np.testing.assert_array_equal(self.data[key], np.repeat(self.data[key][:, :1], 3, axis=1))


if __name__ == "__main__":
    unittest.main()
