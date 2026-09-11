"""Exact archived probe timing and independent-state conventions."""
import numpy as np
import torch

def probe_path(case, delay, query_node, seed):
    """d neutral updates, then one query update. No target label is consumed.

    The first input is the destination already encoded by the outgoing action at
    the edit boundary. The final source differs from the query node, preserving
    the generator's no-zero-saccade convention. Probes may have different neutral
    paths, but every intervention uses the same path for a given scene/probe/delay.
    """
    if delay < 1:
        raise ValueError("Pilot delays must be at least one")
    e = case.episode
    rng = np.random.default_rng(np.random.SeedSequence([seed, e.scene_id, delay, query_node, 811]))
    nodes = [int(e.visits[case.boundary+1])]
    for j in range(delay):
        exclude = [*case.blocked_nodes, nodes[-1]]
        if j == delay-1:
            exclude.append(query_node)
        nodes.append(int(rng.choice(np.setdiff1d(np.arange(len(e.labels)), exclude))))
    nodes = np.asarray(nodes, np.int64)
    destinations = np.concatenate((nodes[1:], [query_node]))
    if np.isin(nodes, case.blocked_nodes).any() or np.any(nodes == destinations):
        raise AssertionError("Leaked edited-node observation or zero saccade")
    actions = e.positions[destinations] - e.positions[nodes]
    return e.labels[nodes], actions, nodes


def flat_state(state):
    return state.permute(1, 0, 2).reshape(state.shape[1], -1).cpu().numpy()


def recurrent_state(flat, device):
    return torch.as_tensor(flat, device=device).reshape(len(flat), 3, 512).permute(1, 0, 2).contiguous()


def bootstrap_mean(x, indices):
    values = x[indices].mean(axis=1)
    return {"mean": float(x.mean()), "ci95": np.quantile(values, [0.025, 0.975]).tolist()}
