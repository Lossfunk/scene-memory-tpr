from dataclasses import dataclass
import numpy as np
from .features import design

@dataclass
class LinearApprox:
    kind: str
    frame: str
    weight: np.ndarray
    bias: np.ndarray
    ridge_lambda: float

    def predict(self, data):
        return (design(data, self.kind, self.frame) @ self.weight + self.bias).astype(np.float32)

    def save(self, path):
        np.savez_compressed(path, kind=self.kind, frame=self.frame, weight=self.weight,
                            bias=self.bias, ridge_lambda=self.ridge_lambda)

    @classmethod
    def load(cls, path):
        with np.load(path) as z:
            return cls(str(z["kind"]), str(z["frame"]), z["weight"], z["bias"], float(z["ridge_lambda"]))


def fit_ridge_family(train, validation, kind, frame, lambdas):
    x = design(train, kind, frame).astype(np.float64)
    y = train["states"].astype(np.float64)
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-4)
    center = y.mean(axis=0)
    z = (x-mean)/scale
    gram = z.T @ z / len(x)
    cross = z.T @ (y-center) / len(x)
    val_x = design(validation, kind, frame)
    best, best_loss, rows = None, np.inf, []
    for penalty in lambdas:
        beta = np.linalg.solve(gram+penalty*np.eye(len(gram)), cross)
        weight = (beta/scale[:, None]).astype(np.float32)
        bias = (center-mean@weight).astype(np.float32)
        loss = float(np.mean((val_x@weight+bias-validation["states"])**2))
        rows.append({"lambda": penalty, "validation_mse": loss})
        if loss < best_loss:
            best_loss = loss
            best = LinearApprox(kind, frame, weight, bias, penalty)
    return best, rows
