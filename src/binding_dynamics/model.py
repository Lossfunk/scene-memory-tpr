"""Thin inference adapter around the pinned public GP_model; no retraining.

State convention: [layer, batch, hidden], AFTER observing the current label and
processing its outgoing displacement. That state predicts the next label and is
carried into the next observation. Intervention deltas use this same convention.
"""
from dataclasses import dataclass
import os
import hashlib
import json
import importlib.util
from pathlib import Path

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_ROOT = Path(os.environ.get("SCENE_MODEL_ROOT", ROOT / "assets/minimal_world_model_interp"))
DEFAULT_CHECKPOINT = PUBLIC_ROOT / "paper_data/model/final_epoch_79.pth"


def choose_device(requested: str = "auto") -> torch.device:
    if requested not in {"auto", "cpu", "mps"}:
        raise ValueError("Expected auto, cpu, or mps")
    available = torch.backends.mps.is_available()
    if requested == "mps" and not available:
        raise RuntimeError("MPS requested but unavailable to this process; check host permissions.")
    return torch.device("mps" if available and requested != "cpu" else "cpu")


@dataclass(frozen=True)
class Run:
    logits: torch.Tensor  # [batch, time, label]
    state: torch.Tensor   # [layer, batch, hidden], final post-update state


class SceneModel:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device="auto"):
        self.device = choose_device(device)
        lock = json.loads((ROOT / "configs/upstream.json").read_text())
        for item in lock["assets"]:
            asset = Path(checkpoint) if item["path"].endswith(".pth") else PUBLIC_ROOT / item["path"]
            if not asset.exists():
                raise FileNotFoundError(f"Missing {asset}; run python scripts/fetch_assets.py")
            if hashlib.sha256(asset.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Pinned asset hash mismatch: {asset}")
        spec = importlib.util.spec_from_file_location("public_scene_model", PUBLIC_ROOT / "setup/model.py")
        if spec is None or spec.loader is None:
            raise ImportError("Public model source unavailable; run fetch_public_assets.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        weights = saved["model_state_dict"]
        hidden = weights["rnn.weight_hh_l0"].shape[1]
        layers = sum(key.startswith("rnn.weight_hh_l") for key in weights)
        self.config = dict(
            tokens_size=weights["token_proj.linear.weight"].shape[1],
            directions_size=weights["coord_proj.linear.weight"].shape[1],
            embedding_size=weights["embedding.0.weight"].shape[1],
            hidden_size=hidden, n_layers=layers, dropout=0.0,
            output_size=weights["readout.linear.weight"].shape[0],
            layer_norm="rnn_norm.weight" in weights,
        )
        self.model = module.GP_model(**self.config)
        self.model.load_state_dict(weights, strict=True)
        self.model.eval().requires_grad_(False).to(self.device)
        self.metadata = {
            "checkpoint": str(Path(checkpoint).resolve()),
            "epoch": saved.get("epoch"),
            "recorded_loss": saved.get("loss"),
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "config_inferred_from_weights": self.config,
        }

    def inputs(self, labels, displacements):
        labels = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        displacements = torch.as_tensor(displacements, dtype=torch.float32, device=self.device)
        if labels.ndim != 2 or displacements.shape != (*labels.shape, 2):
            raise ValueError("Expected labels [B,T] and displacements [B,T,2]")
        if labels.shape[1] == 0:
            raise ValueError("An inference segment must contain at least one step")
        return F.one_hot(labels, num_classes=self.config["tokens_size"]).float(), displacements

    def _state(self, state, batch):
        if state is None:
            return None
        expected = (self.config["n_layers"], batch, self.config["hidden_size"])
        if tuple(state.shape) != expected:
            raise ValueError(f"State shape {tuple(state.shape)} != {expected}")
        return state.detach().clone().to(device=self.device, dtype=torch.float32)

    @torch.inference_mode()
    def run(self, labels, displacements, initial_state=None) -> Run:
        tokens, actions = self.inputs(labels, displacements)
        state = self._state(initial_state, tokens.shape[0])
        m = self.model
        projected = torch.cat((m.token_proj(tokens), m.coord_proj(actions)), dim=2)
        embedded = m.embedding(m.dropout(projected))
        top, final = m.rnn(embedded, state)
        logits = m.readout(F.relu(m.fc_proj(m.rnn_norm(top))), transpose=True)
        return Run(logits, final.detach().clone())

    @torch.inference_mode()
    def public_forward(self, labels, displacements, initial_state=None):
        tokens, actions = self.inputs(labels, displacements)
        return self.model(tokens, actions, initial_hidden=self._state(initial_state, tokens.shape[0]))

    @torch.inference_mode()
    def trace(self, labels, displacements, initial_state=None):
        """Public manual unroll; trace shape [B,T,L,H], raw states before norm.

        Gate collection is for small validation sequences only. Use chunked native
        recurrence for large data collection to avoid retaining every gate tensor.
        """
        tokens, actions = self.inputs(labels, displacements)
        logits, layers, gates = self.model(
            tokens, actions, return_all_activations=True,
            initial_hidden=self._state(initial_state, tokens.shape[0]),
        )
        return logits, torch.stack(layers, dim=2), gates


def edit_state(state: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Apply one detached delta to a cloned state; never mutate another branch."""
    if state.shape != delta.shape:
        raise ValueError("State and delta must have exactly equal shapes")
    if state.device != delta.device or state.dtype != delta.dtype:
        raise ValueError("State and delta must use the same device and dtype")
    if not bool(torch.isfinite(delta).all()):
        raise ValueError("Nonfinite intervention delta")
    return state.detach().clone() + delta.detach()

