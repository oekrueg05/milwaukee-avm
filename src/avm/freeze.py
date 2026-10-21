"""Re-tune on the Stage 6 feature set (CV 2021-2024 ONLY), then freeze.

Usage:
    python -m avm.freeze

Per variant (primary = independent + all groups; secondary = informed with
prior assessed value relative to the neighborhood median), 20 random configs
(seed FREEZE_SEED, distinct from Stage 5) on the detrended target, selected
by the Stage 5 rule: among configs within 0.2 pts of the best mean CV
level-neutral MdAPE, the smallest mean |within-tier PRB|. NO 2025 RESULT
ENTERS SELECTION (Owen, 2026-10-18). Fold frames are built once per location
unit with training from 2014; a config's later train_start drops earlier
training rows.

Writes reports/frozen/model_spec.json (configs, seed, features, windows,
code commit, input hashes), the CV table, and frozen boosters fit through
2024 to interim/models/. The 2025 evaluation of the frozen models is
reported, never used to choose.
"""

import json
import subprocess

import numpy as np
import pandas as pd

from avm.acquire import sha256_of
from avm.config import INTERIM_DIR, REPORTS_DIR
from avm.gbm import SEED, GBMX, sample_configs
from avm.progress import log, step
from avm.stage6_geo import GROUPS, cols_for, metrics
from avm import frames

OUT = REPORTS_DIR / "frozen"
MODELS = INTERIM_DIR / "models"
CV_YEARS = [2021, 2022, 2023, 2024]
TEST_YEAR = 2025
N_CONFIGS = 20
FREEZE_SEED = SEED + 7
MDAPE_TOL = 0.002
BASE_START = 2014
VARIANTS = {"primary": ("independent", False), "secondary": ("assessment_informed", True)}


def fit_predict(fold: dict, cfg: dict, cols: list[str]) -> pd.DataFrame:
    tr = fold["train"][fold["train"].sale_year >= cfg["train_start"]]
    m = GBMX(cfg, cols, monotone=True).fit(tr, tr.y_detr)
    sc = fold["scored"]
    return sc[["sale_id", "target_price", "tier", "neighborhood"]].assign(pred=np.exp(m.predict(sc)))


def select(summ: pd.DataFrame) -> pd.Series:
    cand = summ[summ.mdape_ln <= summ.mdape_ln.min() + MDAPE_TOL]
    return cand.sort_values(["prb_tier_abs", "mdape_ln"]).iloc[0]


def spec_diffs(old: dict, new: dict) -> list[str]:
    """Top-level spec fields that differ, ignoring the commit stamp."""
    return [k for k in sorted(set(old) | set(new)) if k != "frozen_at_commit" and old.get(k) != new.get(k)]


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=REPORTS_DIR.parent).stdout.strip()


def main(verify: bool = False) -> None:
    """verify=True (used by the pipeline): re-run the deterministic tuning and
    freeze, then compare with the committed model_spec.json (all fields but
    frozen_at_commit) and raise on any difference. The committed spec is never
    rewritten in verify mode."""
    out = OUT / "verify" if verify else OUT  # verify never touches the committed frozen reports
    out.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    s = frames.sales()
    glob = frames.global_table(s)
    configs = sample_configs(N_CONFIGS, seed=FREEZE_SEED)
    folds = {}
    for loc in sorted({c["loc"] for c in configs}):
        for Y in CV_YEARS + [TEST_YEAR]:
            with step(f"fold frame {loc} {Y}", 3600):
                folds[(loc, Y)] = frames.fold(s, glob, Y, BASE_START, loc)

    rows = []
    for role, (variant, rel) in VARIANTS.items():
        cols = cols_for(variant, list(GROUPS), informed_rel=rel)
        with step(f"tuning {role} ({variant}): {N_CONFIGS} configs x {len(CV_YEARS)} folds", 4 * 3600):
            for cfg in configs:
                ms = [metrics(fit_predict(folds[(cfg["loc"], Y)], cfg, cols)) for Y in CV_YEARS]
                r = {"role": role, "variant": variant, **cfg,
                     **{k: float(np.mean([m[k] for m in ms])) for k in ["mdape_ln", "cod", "median_ratio",
                                                                         "prd", "prb", "mki", "prb_tier_abs"]}}
                rows.append(r)
                log(f"      {role} cfg {cfg['config_id']:>2}: MdAPE_ln {r['mdape_ln']:.4f} COD {r['cod']:.2f} "
                    f"|tierPRB| {r['prb_tier_abs']:.3f}")
    cv = pd.DataFrame(rows)
    cv.to_csv(out / "tuning_cv_2021_2024.csv", index=False)

    spec = {"frozen_at_commit": git_head(), "selection": "CV 2021-2024 only; within 0.2 pts of best mean "
            "level-neutral MdAPE, then smallest mean |within-tier PRB|", "lightgbm_seed": SEED,
            "config_sampler_seed": FREEZE_SEED, "target": "log price minus training-years-only hedonic time "
            "effect (Dec Y-1 trend level, neutral season)", "monotone_up": ["l_area", "l_lot", "full_baths",
                                                                          "half_baths"],
            "inputs": {"sales_clean_sha256": sha256_of(INTERIM_DIR / "sales_clean.parquet"),
                       "target_prices_sha256": sha256_of(INTERIM_DIR / "target_prices.parquet")},
            "models": {}}
    evals = []
    for role, (variant, rel) in VARIANTS.items():
        pick = select(cv[cv.role == role])
        cfg = {k: (pick[k].item() if hasattr(pick[k], "item") else pick[k]) for k in configs[0]}
        cols = cols_for(variant, list(GROUPS), informed_rel=rel)
        spec["models"][role] = {"variant": variant, "features": cols, "config": cfg,
                                "cv_2021_2024": {k: float(pick[k]) for k in ["mdape_ln", "cod", "median_ratio",
                                                                              "prd", "prb", "mki", "prb_tier_abs"]}}
        with step(f"frozen {role}: fit through 2024, report 2025 (not used to select)", 3600):
            fold = folds[(cfg["loc"], TEST_YEAR)]
            tr = fold["train"][fold["train"].sale_year >= cfg["train_start"]]
            m = GBMX(cfg, cols, monotone=True).fit(tr, tr.y_detr)
            path = MODELS / f"frozen_{role}_through_2024.txt"
            m.booster.save_model(str(path))
            spec["models"][role]["booster_through_2024_sha256"] = sha256_of(path)
            o = fold["scored"][["sale_id", "target_price", "tier", "neighborhood"]].assign(
                pred=np.exp(m.predict(fold["scored"])))
            o.to_parquet(INTERIM_DIR / f"frozen_{role}_predictions_2025.parquet", index=False)
            evals.append({"role": role, **metrics(o.dropna(subset=["target_price"]))})
    pd.DataFrame(evals).to_csv(out / "frozen_eval_2025.csv", index=False)
    if verify:
        old = json.loads((OUT / "model_spec.json").read_text())
        diffs = spec_diffs(old, json.loads(json.dumps(spec, default=str)))
        if diffs:
            raise AssertionError(f"rebuild does not reproduce the frozen spec; differing fields: {diffs}")
        log("frozen spec reproduced exactly (all fields but frozen_at_commit); committed spec left unchanged")
        return
    (OUT / "model_spec.json").write_text(json.dumps(spec, indent=2, default=str), encoding="utf-8")
    log(f"wrote {OUT / 'model_spec.json'}")


if __name__ == "__main__":
    main()
