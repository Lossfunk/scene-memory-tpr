"""Static scenes and explicitly observed histories.

Intervention schedules live in interventions.py; restricted-training generation is
future work. Ground truth scene metadata must not enter representation fitting.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Episode:
    scene_id: int
    positions: np.ndarray  # [N,2]; starts at origin, privileged evaluator metadata
    labels: np.ndarray     # [N]; evaluator metadata
    visits: np.ndarray     # [T+1]; no repeated adjacent node
    actions: np.ndarray    # [T,2]

    @property
    def input_labels(self):
        return self.labels[self.visits[:-1]].copy()

    @property
    def targets(self):
        return self.labels[self.visits[1:]].copy()

    def observations_through(self, step: int):
        """Only labels seen by post-update step t; never the label at t+1.

        Return event-level labels and integrated positions, preserving repeated
        exposures. No access to unvisited coordinates is needed for this history.
        """
        if not 0 <= step < len(self.actions):
            raise IndexError("Step outside episode")
        positions = np.concatenate((np.zeros((1, 2)), np.cumsum(self.actions[:step], axis=0)))
        labels = self.input_labels[:step + 1].copy()
        return labels, positions.astype(np.float32)


def generate_episode(scene_id: int, seed: int, steps=100, n_tokens=6, unique_labels=False):
    if not 2 <= n_tokens <= 26 or steps < 1 or seed < 0 or scene_id < 0:
        raise ValueError("Invalid scene size, length, seed, or scene ID")
    rng = np.random.default_rng(np.random.SeedSequence([seed, scene_id]))
    positions = [np.zeros(2)]
    for _ in range(n_tokens - 1):
        for _attempt in range(10_000):
            candidate = rng.uniform(-4, 4, size=2)
            if min(np.linalg.norm(candidate - p) for p in positions) >= 0.25:
                positions.append(candidate)
                break
        else:
            raise RuntimeError("Unable to place a separated token")
    positions = np.asarray(positions, dtype=np.float32)
    labels = rng.choice(26, size=n_tokens, replace=not unique_labels).astype(np.int64)
    visits = [0]
    for _ in range(steps):
        choices = np.delete(np.arange(n_tokens), visits[-1])
        visits.append(int(rng.choice(choices)))
    visits = np.asarray(visits, dtype=np.int64)
    actions = positions[visits[1:]] - positions[visits[:-1]]
    return Episode(scene_id, positions, labels, visits, actions)
