import numpy as np
import torch
from torch.nn import functional as F
from .interventions import bootstrap_mean

def retrieval_scores(truth, prediction, center=False, cosine=False):
    """Rows are predictions, columns actual candidates; ties receive fractional credit.

    truth/prediction have shape [scene family, candidate identity, hidden unit].
    Family centering is an explicitly transductive diagnostic: it removes each
    set's common component using all candidates, and never fits model parameters.
    """
    a, b = truth.astype(np.float64), prediction.astype(np.float64)
    if center:
        a, b = a-a.mean(1, keepdims=True), b-b.mean(1, keepdims=True)
    if cosine:
        a = a/np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
        b = b/np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    distance = (b*b).sum(-1)[..., None]+(a*a).sum(-1)[:, None, :] - 2*np.einsum('fid,fjd->fij', b, a)
    k = distance.shape[-1]
    diagonal = np.diagonal(distance, axis1=1, axis2=2)
    tied = np.isclose(distance, distance.min(-1, keepdims=True), atol=1e-8, rtol=1e-7)
    top1 = np.diagonal(tied, axis1=1, axis2=2)/tied.sum(-1)
    tolerance = 1e-8+1e-7*np.abs(diagonal[..., None])
    pair = ((distance>diagonal[..., None]+tolerance).sum(-1) +
            .5*((np.abs(distance-diagonal[..., None])<=tolerance).sum(-1)-1))/(k-1)
    return {"top1": top1.mean(1), "pairwise": pair.mean(1), "top1_by_candidate": top1,
            "mean_squared_matching_error": np.mean((a-b)**2, axis=(1,2))}


def intervals(values, seed=82319, n_bootstrap=4000):
    values = np.asarray(values)
    ix = np.random.default_rng(seed).integers(len(values), size=(n_bootstrap, len(values)))
    return bootstrap_mean(values, ix)


def delta_geometry(actual, donor, edited):
    truth, pred = (donor-actual).astype(np.float64), (edited-actual).astype(np.float64)
    tnorm, pnorm = np.linalg.norm(truth, axis=-1), np.linalg.norm(pred, axis=-1)
    cos = (truth*pred).sum(-1)/np.maximum(tnorm*pnorm, 1e-12)
    return {"mean_cosine_to_donor_difference": intervals(cos),
            "mean_norm_ratio_to_donor_difference": intervals(pnorm/np.maximum(tnorm,1e-12)),
            "pooled_delta_r2_vs_zero": float(1-np.sum((truth-pred)**2)/np.sum(truth**2)),
            "mean_per_scene_delta_r2_vs_zero": intervals(1-np.sum((truth-pred)**2,axis=-1)/np.maximum(tnorm**2,1e-12)),
            "optimistic_direction_only_explained_energy": intervals(np.maximum(cos,0)**2),
            "direction_only_note": "Per-case oracle positive rescaling bound; diagnostic only, not an implemented intervention"}



def reconstruction_metrics(target, prediction, training_mean):
    error = (prediction-target)**2
    baseline = (target-training_mean)**2
    return {"mse": float(error.mean()), "r2_training_mean_reference": float(1-error.sum()/baseline.sum()),
            "layer_r2": [float(1-error[:, i*512:(i+1)*512].sum()/baseline[:, i*512:(i+1)*512].sum()) for i in range(3)]}


@torch.inference_mode()
def output_metrics(target_model, states, expected_logits, target_labels):
    m = target_model.model
    states = torch.as_tensor(states, device=target_model.device)
    logits = m.readout(F.relu(m.fc_proj(m.rnn_norm(states[:, -512:]))), transpose=True)
    reference = torch.as_tensor(expected_logits, device=logits.device)
    probabilities = reference.softmax(-1)
    kl = (probabilities * (reference.log_softmax(-1)-logits.log_softmax(-1))).sum(-1)
    return {"original_output_agreement": float((logits.argmax(-1)==reference.argmax(-1)).float().mean()),
            "correct_answer_accuracy": float((logits.argmax(-1)==torch.as_tensor(target_labels, device=logits.device)).float().mean()),
            "mean_KL_original_to_reconstruction": float(kl.mean())}
