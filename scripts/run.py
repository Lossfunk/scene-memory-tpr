"""Collect, refit, or replay only the two static TPR models in the report."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from binding_dynamics.binding import fit_ridge_family
from binding_dynamics.clean_tpr import CleanTPR
from binding_dynamics.clean_tpr_fit import train_clean
from binding_dynamics.data import collect_counterfactuals, flatten_groups, matched_binding_records
from binding_dynamics.evaluation import boundary_data, make_probe_inputs, model_edits, run_probe, score_logits, variant_data
from binding_dynamics.metrics import intervals, output_metrics, reconstruction_metrics, retrieval_scores
from binding_dynamics.model import SceneModel


def read(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def scores(logits, probe):
    result, arrays = score_logits(logits, probe)
    correct = logits.argmax(-1) == probe["answers"]
    for index, kind in enumerate(["single", "swap"]):
        values = np.where(probe["target_mask"][:, index], correct[:, index], True).all(-1).astype(float)
        result[kind]["all_targets_correct"] = intervals(values)
    return result


def collect(target, config, directory, splits):
    for split in splits:
        path = directory / f"{split}.npz"
        if path.exists():
            recorded = json.loads(path.with_suffix(".json").read_text())
            for key in ["seed", "scenes"]:
                if recorded[key] != config["splits"][split][key]:
                    raise ValueError(f"Cached {split} {key} mismatch; use a new --work directory")
            print(f"Using existing {split} data")
        else:
            collect_counterfactuals(target, path, **config["splits"][split],
                                   snapshot_steps=config["snapshot_steps"],
                                   minimum_observations=config["minimum_observations"])


def fit(target, config, work):
    collect(target, config, work / "data", ["train", "validation"])
    train, validation = [read(work / "data" / f"{split}.npz") for split in ["train", "validation"]]
    initial, _ = fit_ridge_family(flatten_groups(train), flatten_groups(validation), "full_pair", "absolute", [.01])
    output = work / "models"
    output.mkdir(parents=True, exist_ok=True)
    initial.save(output / "initialization.npz")
    np.save(output / "training_mean.npy", train["states"][:, 0].mean(0))
    for spec in config["candidates"]:
        path = output / f"{spec['name']}_s{config['training']['seed']}.pt"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite fitted model: {path}")
        model, info = train_clean(train, validation, initial, spec, config,
                                  config["training"]["seed"], device=str(target.device))
        model.save(path, info)
    print(f"Two fits saved to {output}; no model selection against the old test is performed.")


def evaluate(target, config, work, model_dir):
    output = work / "evaluation"
    if (output / "results.json").exists():
        raise FileExistsError("Evaluation already exists; choose a new --work directory")
    collect(target, config, work / "data", ["test"])
    data = read(work / "data/test.npz")
    matched_path = work / "data/matched.npz"
    if matched_path.exists():
        matched = read(matched_path)
    else:
        matched, rejected = matched_binding_records(target, **config["matched_test"])
        np.savez_compressed(matched_path, **matched)
        matched_path.with_suffix(".json").write_text(json.dumps({"rejected_scene_ids": rejected}, indent=2) + "\n")
    factual = variant_data(data, 0)
    boundary = boundary_data(data)
    probe = make_probe_inputs(boundary, config["splits"]["test"]["seed"], 5)
    training_mean = np.load(model_dir / "training_mean.npy", allow_pickle=False)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "probes.npz", **probe)
    report = {"status": "replay/refit evaluation; not a new confirmatory cohort", "device": str(target.device),
              "models": {}, "controls": {}, "model_sha256": {}, "matched_families": len(matched["states"]) // 6}
    for spec in config["candidates"]:
        name = f"{spec['name']}_s{config['training']['seed']}"
        path = model_dir / f"{name}.pt"
        model = CleanTPR.load(path, str(target.device))
        predicted = model.predict(factual)
        matched_prediction = model.predict(matched)
        retrieval = retrieval_scores(matched["states"].reshape(-1, 6, 1536), matched_prediction.reshape(-1, 6, 1536))
        states = model_edits(model, boundary, "clipped_difference")
        logits = run_probe(target, states, probe)
        row = {"representation": {**reconstruction_metrics(factual["states"], predicted, training_mean),
                                  **output_metrics(target, predicted, factual["logits"], factual["targets"])},
               "retrieval": {key: intervals(retrieval[key]) for key in ["top1", "pairwise"]},
               "behavior": scores(logits, probe)}
        report["models"][name] = row
        report["model_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        np.savez_compressed(output / f"{name}.npz", logits=logits, states=states,
                            factual_predictions=predicted, matched_predictions=matched_prediction)
        print(name, "replacement", row["behavior"]["single"]["target_accuracy"]["mean"],
              "swap", row["behavior"]["swap"]["target_accuracy"]["mean"], flush=True)
    for name, states in [("original", np.repeat(boundary["states"][:, :1], 2, axis=1)), ("donor", boundary["states"][:, 1:])]:
        logits = run_probe(target, states, probe)
        report["controls"][name] = scores(logits, probe)
        np.savez_compressed(output / f"{name}.npz", logits=logits)
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Evaluation complete: {output / 'results.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["collect", "fit", "evaluate"])
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--work", type=Path, default=ROOT / "runs/reproduction")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--smoke", action="store_true", help="Tiny engineering check; never a scientific result")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/experiment.json").read_text())
    if args.smoke:
        for key, count in [("train", 12), ("validation", 6), ("test", 8)]:
            config["splits"][key]["scenes"] = count
        config["matched_test"]["scenes"] = 8
        config["training"].update(epochs=2, patience=2)
    args.work.mkdir(parents=True, exist_ok=True)
    config_path = args.work / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Work-directory configuration differs; use a new --work directory")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    torch.set_num_threads(4)
    target = SceneModel(device=args.device)
    if args.command == "collect":
        collect(target, config, args.work / "data", ["train", "validation"])
    elif args.command == "fit":
        fit(target, config, args.work)
    else:
        evaluate(target, config, args.work, args.model_dir)


if __name__ == "__main__":
    main()
