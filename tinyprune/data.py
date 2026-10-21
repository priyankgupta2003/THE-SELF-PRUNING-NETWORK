"""Dataset loading. scikit-learn is used ONLY to load data, nothing else."""
import numpy as np


def load(name: str = "digits", seed: int = 0, test_frac: float = 0.2):
    if name == "digits":
        from sklearn.datasets import load_digits  # bundled with sklearn, no download
        d = load_digits()
        X, y = d.data.astype(np.float64) / 16.0, d.target.astype(np.int64)
    elif name == "fashion":
        from sklearn.datasets import fetch_openml  # needs internet
        X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True, as_frame=False, parser="auto")
        X, y = X.astype(np.float64) / 255.0, y.astype(np.int64)
    else:
        raise ValueError(name)
    # stratified split with our own RNG (no sklearn model_selection)
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.nonzero(y == c)[0])
        k = int(round(test_frac * len(idx)))
        te.extend(idx[:k]); tr.extend(idx[k:])
    tr, te = np.array(tr), np.array(te)
    # centre per feature, scale by ONE global std (TRAIN statistics only).
    # Per-feature std is a trap here: border pixels are almost always 0 in the
    # training set, so dividing by their tiny std turns a rare test-time ink
    # pixel into a |x| ~ 40 outlier.
    mu = X[tr].mean(0)
    sd = (X[tr] - mu).std() + 1e-8
    X = (X - mu) / sd
    return X[tr], y[tr], X[te], y[te]
