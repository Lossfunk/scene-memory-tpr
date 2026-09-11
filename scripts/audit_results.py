"""Recompute reported endpoints from compact, bundled evidence (no target needed)."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def interval(values):
    values = np.asarray(values, dtype=float)
    indices = np.random.default_rng(82319).integers(len(values), size=(4000, len(values)))
    samples = values[indices].mean(1)
    return {"mean": float(values.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist()}


def per_scene(logits, evidence):
    correct = logits.argmax(-1) == evidence["probe_answers"]
    mask = evidence["probe_target_mask"]
    result = {}
    for i, name in enumerate(["single", "swap"]):
        c, target = correct[:, i], mask[:, i]
        result[name + "_target"] = (c * target).sum(-1) / target.sum(-1)
        result[name + "_unchanged"] = (c * ~target).sum(-1) / (~target).sum(-1)
        result[name + "_all_targets"] = np.where(target, c, True).all(-1).astype(float)
        result[name + "_all_five"] = c.all(-1).astype(float)
    result["joint"] = .5 * (result["single_all_five"] + result["swap_all_five"])
    return result


def reference_behavior(row):
    result = {}
    for kind in ["single", "swap"]:
        for key, original in [("target", "target_accuracy"), ("unchanged", "offtarget_accuracy"),
                              ("all_targets", "all_targets_correct"), ("all_five", "all_five_correct")]:
            result[kind + "_" + key] = row[kind][original]
    result["joint"] = row["selection_joint_score"]
    return result


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    metrics = [("single_target", "Replacement letter retrieved"),
               ("swap_target", "Swapped letter retrieved (per location)"),
               ("swap_all_targets", "Both swapped letters recalled correctly"),
               ("single_unchanged", "Unchanged locations after replacement"),
               ("swap_unchanged", "Unchanged locations after swap"),
               ("joint", "All five correct (mean of the two tasks)")]
    colors = {"linear": "#405e88", "tanh": "#b24a24", "original": "#787878"}
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for index, (key, label) in enumerate(metrics):
        for model, shift in [("linear", -.14), ("tanh", .14)]:
            v = rows[model]["behavior"][key]
            mean = 100 * v["mean"]
            lo, hi = 100 * np.asarray(v["ci95"])
            ax.errorbar(mean, index + shift, xerr=[[mean-lo], [hi-mean]], fmt="o",
                        markersize=4.5, color=colors[model], linewidth=1.2, capsize=0)
            ax.text(103, index + shift, f"{mean:.2f}%", va="center", color=colors[model], fontsize=9)
        if "unchanged" in key:
            v = rows["original"]["behavior"][key]
            mean = 100 * v["mean"]
            lo, hi = 100 * np.asarray(v["ci95"])
            ax.errorbar(mean, index, xerr=[[mean-lo], [hi-mean]], fmt="D", markersize=3,
                        color=colors["original"], linewidth=.8)
    ax.set(yticks=range(len(metrics)), yticklabels=[label for _, label in metrics],
           xlim=(0, 100), ylim=(len(metrics)-.5, -.65), xlabel="Accuracy (%; farther right means more correct)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.grid(axis="x", color="#e6e6e6", linewidth=.6)
    ax.set_axisbelow(True)
    legend = [Line2D([], [], color=colors[name], marker="D" if name == "original" else "o",
                     linestyle="", label=label) for name, label in
              [("linear", "Linear TPR"), ("tanh", "TPR + tanh"), ("original", "Without editing")]]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.02, .93), frameon=False, ncol=3, fontsize=9)
    fig.suptitle("Recall after TPR edits: intended changes and unchanged letters",
                 x=.025, ha="left", fontsize=12.5, y=.985)
    fig.text(.025, .025, "400 scenes · 5 intervening updates + 1 query update · edited and queried locations not re-observed\n"
             "95% intervals resample scenes. Each location is tested on a separate copy of the edited state.",
             fontsize=8.5, color="#444444")
    fig.subplots_adjust(left=.40, right=.89, top=.80, bottom=.19)
    (ROOT / "figures").mkdir(exist_ok=True)
    for extension in ["png", "pdf"]:
        fig.savefig(ROOT / "figures" / f"results.{extension}", dpi=180)
    plt.close(fig)


def main(make_plot=False):
    with np.load(ROOT / "results/evidence.npz", allow_pickle=False) as data:
        evidence = dict(data)
    reported = json.loads((ROOT / "results/reported.json").read_text())
    result = {}
    values = {}
    for model in ["linear", "tanh", "original", "donor"]:
        values[model] = per_scene(evidence[model + "_logits"], evidence)
        behavior = {key: interval(value) for key, value in values[model].items()}
        old = (reported["models"][model]["modes"]["clipped_difference"]["delays"]["5"]
               if model in ["linear", "tanh"] else reported["controls"][model]["5"])
        for key, reference in reference_behavior(old).items():
            np.testing.assert_allclose(behavior[key]["mean"], reference["mean"], atol=1e-12)
            np.testing.assert_allclose(behavior[key]["ci95"], reference["ci95"], atol=1e-12)
        result[model] = {"behavior": behavior}
        if model in ["linear", "tanh"]:
            distances = evidence[model + "_matched_distances"]
            tied = np.isclose(distances, distances.min(-1, keepdims=True), atol=1e-8, rtol=1e-7)
            top1 = (np.diagonal(tied, axis1=1, axis2=2) / tied.sum(-1)).mean(1)
            r2 = float(1-evidence[model + "_factual_squared_error_sums"].sum()/evidence["factual_baseline_sums"].sum())
            result[model]["representation"] = {"state_r2": r2, "assignment_top1": interval(top1)}
            np.testing.assert_allclose(r2, reported["models"][model]["representation_factual"]["r2_training_mean_reference"], atol=1e-6)
            np.testing.assert_allclose(top1.mean(), reported["models"][model]["retrieval"]["raw"]["top1"]["mean"], atol=1e-12)
            if model + "_factual_output_agreement" in evidence:
                agreement = float(evidence[model + "_factual_output_agreement"].mean())
                result[model]["representation"]["immediate_output_agreement"] = agreement
                np.testing.assert_allclose(agreement, reported["models"][model]["representation_factual"]["original_output_agreement"], atol=1e-6)
    result["tanh_minus_linear"] = {key: interval(values["tanh"][key]-values["linear"][key]) for key in values["tanh"]}
    result["audit"] = {"scene_bootstrap_seed": 82319, "replicates": 4000,
                       "causal_families": len(evidence["probe_answers"]), "matched_families": len(evidence["matched_scene_ids"]),
                       "passed": True}
    (ROOT / "results/audited.json").write_text(json.dumps(result, indent=2) + "\n")
    with (ROOT / "results/metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "metric", "mean", "ci95_low", "ci95_high", "units"])
        for model in ["linear", "tanh", "original", "donor"]:
            for key, row in result[model]["behavior"].items():
                writer.writerow([model, key, row["mean"], *row["ci95"], "proportion"])
    if make_plot:
        plot(result)
    print(json.dumps(result["audit"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true")
    main(parser.parse_args().plot)
