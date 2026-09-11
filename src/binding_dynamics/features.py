"""Fixed smooth position features and observation-only explanatory designs."""
import numpy as np
import torch

WAVES = np.asarray([(0.5, 0), (0, 0.5), (1, 0), (0, 1), (1, 1), (1, -1),
                    (2, 0), (0, 2), (2, 2), (2, -2), (4, 0), (0, 4)], dtype=np.float32)
POSITION_DIM = 3 + 2 * len(WAVES)
Q_DIM = 26 + 2 * POSITION_DIM + 4


def position_features(p):
    p = np.asarray(p, dtype=np.float32)
    angle = (p / 4) @ WAVES.T * np.pi
    return np.concatenate((np.ones((*p.shape[:-1], 1), dtype=np.float32), p / 4,
                           np.sin(angle), np.cos(angle)), axis=-1).astype(np.float32)


def random_features(x, out_dim=64, seed=821):
    rng = np.random.default_rng(seed)
    weight = rng.normal(size=(x.shape[-1], out_dim)).astype(np.float32) / np.sqrt(x.shape[-1])
    phase = rng.uniform(-np.pi, np.pi, size=out_dim).astype(np.float32)
    return (np.sqrt(2 / out_dim) * np.cos(x @ weight + phase)).astype(np.float32)


def history_record(episode, t):
    """Six stable bookkeeping slots; all fit features come from observed events.

    Node IDs identify slots but never become explanatory features. Unvisited slots
    have label/position zero and mask zero, so no future scene content is exposed.
    """
    obs_labels, obs_positions = episode.observations_through(t)
    nodes = episode.visits[:t+1]
    n = len(episode.labels)
    labels = np.zeros(n, dtype=np.int64)
    positions = np.zeros((n, 2), dtype=np.float32)
    counts = np.zeros(n, dtype=np.int64)
    for node, label, pos in zip(nodes, obs_labels, obs_positions):
        labels[node], positions[node] = label, pos
        counts[node] += 1
    mask = (counts > 0).astype(np.float32)
    current = obs_positions[-1]
    nxt = current + episode.actions[t]
    q = np.concatenate((np.eye(26, dtype=np.float32)[obs_labels[-1]],
                        position_features(current), position_features(nxt),
                        episode.actions[t] / 8, np.asarray([t/100, mask.sum()/6], dtype=np.float32)))
    # A transition becomes observed only when its destination label has arrived.
    edges = {}
    for j in range(t):
        edges[(int(nodes[j]), int(nodes[j+1]))] = np.concatenate((
            np.eye(26, dtype=np.float32)[obs_labels[j]],
            np.eye(26, dtype=np.float32)[obs_labels[j+1]],
            position_features(obs_positions[j+1]-obs_positions[j])))
    transition = np.zeros(64, dtype=np.float32)
    if edges:
        transition = random_features(np.stack(list(edges.values())), seed=823).sum(axis=0)
    return dict(labels=labels, positions=positions, mask=mask, counts=counts,
                current=current, next_position=nxt, q=q.astype(np.float32), transition=transition)


def role_features(data, frame):
    p = data["positions"]
    if frame == "next_relative":
        p = p - data["next_position"][:, None, :]
    elif frame != "absolute":
        raise ValueError(frame)
    return position_features(p)


def design(data, kind, frame="absolute"):
    q = data["q"]
    if kind == "nuisance":
        return q
    if kind == "transition":
        return np.concatenate((q, data["transition"]), axis=1)
    phi = role_features(data, frame)
    onehot = np.eye(26, dtype=np.float32)[data["labels"]]
    mask = data["mask"]
    if kind == "full_pair":
        memory = np.einsum("bnf,bnr,bn->bfr", onehot, phi, mask).reshape(len(q), -1)
    elif kind == "unbound":
        memory = np.concatenate(((onehot * mask[..., None]).sum(axis=1),
                                 (phi * mask[..., None]).sum(axis=1)), axis=1)
    elif kind == "joint_random":
        joined = np.concatenate((onehot * 2, phi), axis=-1)
        memory = (random_features(joined) * mask[..., None]).sum(axis=1)
    else:
        raise ValueError(kind)
    return np.concatenate((q, memory), axis=1).astype(np.float32)


def tensor_inputs(data, frame, device):
    return {"labels": torch.as_tensor(data["labels"], dtype=torch.long, device=device),
            "roles": torch.as_tensor(role_features(data, frame), device=device),
            "mask": torch.as_tensor(data["mask"], device=device),
            "q": torch.as_tensor(data["q"], device=device)}

