"""LightGBM models on the shared feature frame (Stage 5).

Two variants (rule 6):
  independent           no assessed values
  assessment_informed   + log roll Y-1 assessed total and land value

Target log(price). Prediction at a trend point t (Dec of the last training
year) with a NEUTRAL month-of-year: log predictions averaged over the 12
months. Deterministic: fixed seed, deterministic=True, fixed thread count.
Monotone constraints (increasing) on sq ft, lot and baths when enabled.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

SEED = 20261018
THREADS = 4
NUMERIC = ["l_area", "l_lot", "lot_missing", "age", "beds", "full_baths", "half_baths", "baths_missing",
           "stories_num", "duplex", "t", "moy"]
CATEGORICAL = ["style", "loc"]
INFORMED = ["l_prior_total", "l_prior_land"]
MONOTONE_UP = {"l_area", "l_lot", "full_baths", "half_baths"}


def columns(variant: str) -> list[str]:
    if variant not in ("independent", "assessment_informed"):
        raise ValueError(variant)
    return NUMERIC + CATEGORICAL + (INFORMED if variant == "assessment_informed" else [])


def _matrix(f: pd.DataFrame, cols: list[str], categories: dict | None) -> tuple[pd.DataFrame, dict]:
    X = f[cols].copy()
    cats = categories or {c: sorted(X[c].astype(str).unique()) for c in CATEGORICAL}
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c].astype(str), categories=cats[c])
    return X, cats


def params(cfg: dict, cols: list[str], objective: str = "regression", alpha: float | None = None,
           monotone: bool = False) -> dict:
    p = {"objective": objective, "learning_rate": cfg["learning_rate"], "num_leaves": cfg["num_leaves"],
         "min_data_in_leaf": cfg["min_data_in_leaf"], "feature_fraction": cfg["feature_fraction"],
         "bagging_fraction": cfg["bagging_fraction"], "bagging_freq": 1, "lambda_l2": cfg["lambda_l2"],
         "cat_smooth": cfg["cat_smooth"], "min_data_per_group": 30, "seed": SEED, "deterministic": True,
         "force_row_wise": True, "num_threads": THREADS, "verbosity": -1}
    if alpha is not None:
        p["alpha"] = alpha
    if monotone:
        p["monotone_constraints"] = [1 if c in MONOTONE_UP else 0 for c in cols]
        p["monotone_constraints_method"] = "advanced"
    return p


class GBM:
    def __init__(self, cfg: dict, variant: str, monotone: bool = False, objective: str = "regression",
                 alpha: float | None = None):
        self.cfg, self.variant, self.monotone = cfg, variant, monotone
        self.objective, self.alpha = objective, alpha
        self.cols = columns(variant)

    def fit(self, train: pd.DataFrame) -> "GBM":
        X, self.cats = _matrix(train, self.cols, None)
        ds = lgb.Dataset(X, label=train.lp.to_numpy(), categorical_feature=CATEGORICAL, free_raw_data=False)
        self.booster = lgb.train(params(self.cfg, self.cols, self.objective, self.alpha, self.monotone), ds,
                                 num_boost_round=self.cfg["num_boost_round"])
        return self

    def predict_log(self, X: pd.DataFrame, t_point: float) -> np.ndarray:
        Xm, _ = _matrix(X, self.cols, self.cats)
        logs = []
        for mo in range(1, 13):
            logs.append(self.booster.predict(Xm.assign(t=t_point, moy=mo)))
        return np.mean(logs, axis=0)


class GBMX:
    """LightGBM on an explicit column list and target (Stage 6+): the target
    is already detrended, so there are no time features and no month
    averaging; predict() returns log price at the reference trend level."""

    def __init__(self, cfg: dict, cols: list[str], monotone: bool = False, objective: str = "regression",
                 alpha: float | None = None):
        self.cfg, self.cols, self.objective, self.alpha = cfg, list(cols), objective, alpha
        # LightGBM forbids monotone constraints with the quantile objective.
        self.monotone = monotone and objective != "quantile"
        self.cat = [c for c in self.cols if c in CATEGORICAL]

    def _X(self, f: pd.DataFrame, fit: bool) -> pd.DataFrame:
        X = f[self.cols].copy()
        if fit:
            self.cats = {c: sorted(X[c].astype(str).unique()) for c in self.cat}
        for c in self.cat:
            X[c] = pd.Categorical(X[c].astype(str), categories=self.cats[c])
        return X

    def fit(self, f: pd.DataFrame, y: np.ndarray) -> "GBMX":
        ds = lgb.Dataset(self._X(f, True), label=np.asarray(y), categorical_feature=self.cat, free_raw_data=False)
        self.booster = lgb.train(params(self.cfg, self.cols, self.objective, self.alpha, self.monotone), ds,
                                 num_boost_round=self.cfg["num_boost_round"])
        return self

    def predict(self, f: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(self._X(f, False))


def sample_configs(n: int, seed: int = SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        out.append({"config_id": i,
                    "learning_rate": float(rng.choice([0.02, 0.03, 0.05, 0.08])),
                    "num_boost_round": int(rng.choice([300, 600, 1000, 1500])),
                    "num_leaves": int(rng.choice([15, 31, 63, 127])),
                    "min_data_in_leaf": int(rng.choice([10, 20, 50, 100])),
                    "feature_fraction": float(rng.choice([0.6, 0.8, 1.0])),
                    "bagging_fraction": float(rng.choice([0.7, 0.85, 1.0])),
                    "lambda_l2": float(rng.choice([0.0, 1.0, 5.0, 10.0])),
                    "cat_smooth": float(rng.choice([10.0, 30.0])),
                    "train_start": int(rng.choice([2014, 2017, 2020])),
                    "loc": str(rng.choice(["neighborhood", "tract"]))})
    return out
