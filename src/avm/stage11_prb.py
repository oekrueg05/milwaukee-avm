"""Stage 11 (issue #4): is the frozen model's within-tier PRB real compression,
or mostly the sale-price-noise bias in PRB?

The verdict rule was pre-registered in reports/stage11_prb/verdict.json
("preregistered", committed before any diagnostic ran) and amended once
("amendments", 2026-10-22, after the 2021-fold sample and before any full CV
result: Q is judged against Q_null). This module never edits either; the full
run adds "results" next to them. Verdict from CV 2021-2024 pooled; 2025 is
reported only.

Methods: frozen primary (out-of-fold refits of the frozen spec per CV fold,
checked against Stage 8's predictions; the frozen 2025 predictions for 2025)
and the assessor (roll Y). Common set: scored sales with a benchmark in a
rated neighborhood value tier (evaluate.value_tiers). Ratios are level-
normalized by each method's median ratio within year and tier.

Diagnostics
 1. Calibration slopes: log target on log value (+ year effects in CV) within
    tiers; HC1 and neighborhood-clustered 95% CIs. Slope 1 = calibrated,
    > 1 = compressed values.
 2. Proxy PRB (Q): the PRB formula with log2(proxy) as the value term. Proxy
    = the home's most recent clean prior sale (not the first sale of a flip)
    at least PROXY_MIN_MONTHS before the target sale, detrended with the
    fold's training-only time effect: the same quantity as the model's
    prior_lp_detr. The model uses its own prior sale as a feature, so where
    the proxy is that sale its noise leaks into the value: corrected with
    beta (+/-DELTA finite difference on the refit fold model). Proxy noise
    (sigma_q^2 = sigma_p^2 + drift * gap) attenuates Q: disattenuated.
    Co-primary leak-free check: the frozen spec refit without the
    own-prior-sale group (Q_noown).
 3. Calibrated null: sale noise sigma_p^2 and drift from repeat sales (Case-
    Shiller: squared pair residual on years between, per tier; parcel-
    clustered bootstrap). Simulated perfectly calibrated valuations keep each
    method's actual values V: true log value log V + u, price P* = V exp(u +
    e), proxy = true log value + e_q. PRB_null and Q_null come from the same
    draws; E = PRB - PRB_null, D = Q - Q_null. A calibrated valuation of
    finite accuracy has negative Q, so D (not Q) measures compression.
 4. Within-tier PRB with tercile-specific time adjustment of the targets.
 Tradeoff: log values scaled around the fold-tier mean by k = 1 .. 1/b (b, the
 within-tier calibration slope, fit on earlier CV folds only), plus k = b. No
 fold before 2021 can be built (the time-effect hedonic is rank-deficient on
 2014-2019 training data), so the curve covers folds 2022-2024.

Usage: python -m avm.stage11_prb [--sample]   (--sample: 2021 fold only)
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from avm import frames  # noqa: E402
from avm.config import INTERIM_DIR, REPORTS_DIR  # noqa: E402
from avm.evaluate import value_tiers  # noqa: E402
from avm.features import build_frame, usable  # noqa: E402
from avm.freeze import BASE_START  # noqa: E402
from avm.gbm import GBMX  # noqa: E402
from avm.location import parcel_locations  # noqa: E402
from avm.priors import TimeEffect  # noqa: E402
from avm.progress import log, step  # noqa: E402
from avm.ratio_study import SEED, _stats  # noqa: E402
from avm.reporting import md  # noqa: E402
from avm.prospective import SPEC_PATH  # noqa: E402
from avm.stage4_hedonic import trend_point  # noqa: E402
from avm.stage6_geo import GROUPS  # noqa: E402

OUT = REPORTS_DIR / "stage11_prb"
CV_YEARS = [2021, 2022, 2023, 2024]
TEST_YEAR = 2025
TIERS = ["low", "mid", "high"]
PROXY_MIN_MONTHS = 24
OWN_MIN_MONTHS = 12          # avm.priors.OWN_MIN_MONTHS: the model's own-prior-sale rule
DELTA = 0.05                 # finite-difference step in log price for beta
BAND = 0.05                  # IAAO PRB target band
N_BOOT = 2000
N_SIM = 2000
N_NOISE_BOOT = 1000
N_BINS = 5
N_K = 8                      # tradeoff grid steps
TRADEOFF_FOLDS = [2022, 2023, 2024]
N_BOOT_CURVE = 400           # bootstrap and simulation draws per tradeoff point
MIN_RELIABILITY = 0.2        # below this, disattenuation is not identified
MAX_UNSTABLE = 0.025         # share of bootstrap draws allowed below it before the interval is not identified
LN2 = np.log(2)
METHODS = {"model": "pred_model", "assessor": "bench_assessed_total", "model_no_own_sale": "pred_noown"}
# Owen's wording (2026-10-22), used verbatim in the summary, README, methods and issue #4.
VERDICT_WORDING = (
    "Under the pre-registered rule (amended after the 2021 sample, before full results), the verdict is "
    "'{verdict}'. Substantively, no tier shows compression: the model's negative PRB is about what sale-price "
    "noise produces for a calibrated valuation of its accuracy, apart from a small mid-tier excess (E = {mid_e}). "
    "The model is calibrated for prediction; reaching the IAAO PRB band requires spreading values past "
    "calibration at a cost in accuracy, which is where the assessor sits. Caveat: the prior-sale subset "
    "over-represents White-majority and higher-income tracts.")
COLORS = {"model": "#2a78d6", "assessor": "#eb6834"}   # dataviz default slots 1-2, validated (light)
MARKERS = {"model": "o", "assessor": "s"}


# ------------------------------------------------------------ statistics ---

def prb(value: np.ndarray, price: np.ndarray) -> float:
    """IAAO PRB (avm.ratio_study._stats)."""
    return float(_stats(np.asarray(value, float)[None, :], np.asarray(price, float)[None, :])["prb"][0])


def proxy_prb(value: np.ndarray, price: np.ndarray, log_proxy: np.ndarray) -> float:
    """PRB with log2(proxy) as the value term: slope of (r - med)/med on
    log2(proxy). Noise in the current sale price does not enter the regressor."""
    r = np.asarray(value, float) / np.asarray(price, float)
    med = np.median(r)
    y = (r - med) / med
    x = np.asarray(log_proxy, float) / LN2
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc ** 2).sum())


def corrected_q(q_raw: float, var_lq: float, sigma_q2: np.ndarray, leak: np.ndarray) -> tuple[float, float, float]:
    """Q in log2-proxy units with the leak removed and proxy noise
    disattenuated. Regressing log ratio on ln q = ln q_true + e_q gives slope
    gamma * lambda + mean(leak * sigma_q^2) / Var(ln q), lambda = 1 -
    mean(sigma_q^2) / Var(ln q); leak = beta where the proxy is the model's
    feature sale, else 0. Returns (Q corrected, leak bias in Q units, lambda);
    Q corrected is NaN (not identified) when lambda <= MIN_RELIABILITY."""
    bias = LN2 * float(np.mean(leak * sigma_q2)) / var_lq
    lam = 1 - float(np.mean(sigma_q2)) / var_lq
    return ((q_raw - bias) / lam if lam > MIN_RELIABILITY else float("nan")), bias, lam


def case_shiller(d: np.ndarray, years: np.ndarray) -> tuple[float, float]:
    """Repeat-sale pair residual d (log, detrended) and years between sales:
    E[d^2] = 2 sigma_p^2 + drift * years. Returns (sigma_p^2, drift per year)."""
    b = np.linalg.lstsq(np.c_[np.ones(len(d)), years], d ** 2, rcond=None)[0]
    return float(b[0] / 2), float(b[1])


def inside_band(lo: float, hi: float) -> bool:
    return (lo <= 0 <= hi) or (lo >= -BAND and hi <= BAND)


def verdict(tiers: list[dict], with_noown: bool = True) -> dict:
    """The pre-registered rule as amended 2026-10-22 (verdict.json). tiers:
    dicts with tier, prb_obs, prb_null, e_lo, e_hi, d_lo, d_hi (D = Q -
    Q_null) and, for the model, dn_lo, dn_hi (the no-own-sale D)."""
    failing = [t for t in tiers if t["prb_obs"] < -BAND]
    if not failing:
        return {"verdict": "no failing tier", "failing_tiers": [], "reasons": ["no tier has observed CV PRB < -0.05"]}
    reasons, art, comp = [], [], []
    for t in failing:
        noise_half = t["prb_null"] <= 0.5 * t["prb_obs"] and t["prb_null"] < 0
        e_in, d_in = inside_band(t["e_lo"], t["e_hi"]), inside_band(t["d_lo"], t["d_hi"])
        e_below, d_below = t["e_hi"] < 0, t["d_hi"] < 0
        dn_in = inside_band(t["dn_lo"], t["dn_hi"]) if with_noown else True
        dn_below = t["dn_hi"] < 0 if with_noown else True
        art.append(noise_half and e_in and d_in and dn_in)
        comp.append(d_below and e_below and dn_below)
        parts = [f"noise explains >= half: {noise_half}", f"E within/includes 0: {e_in}", f"E below 0: {e_below}",
                 f"D within/includes 0: {d_in}", f"D below 0: {d_below}"]
        if with_noown:
            parts += [f"D_noown within/includes 0: {dn_in}", f"D_noown below 0: {dn_below}"]
            if d_below != dn_below or d_in != dn_in:
                parts.append("D and D_noown disagree in direction")
        if np.isnan(t["d_lo"]) or (with_noown and np.isnan(t["dn_lo"])):
            parts.append("D interval not identified (proxy reliability too low in too many draws)")
        reasons.append(f"{t['tier']}: " + "; ".join(parts))
    if all(art):
        v = "mostly measurement artifact"
    elif any(comp):
        v = "real compression"
    else:
        v = "mixed, not resolved"
    return {"verdict": v, "failing_tiers": [t["tier"] for t in failing], "reasons": reasons}


# ------------------------------------------------------------------ data ---

def history(s: pd.DataFrame, glob: pd.DataFrame, Y: int, loc: str) -> pd.DataFrame:
    """Detrended clean sales as fold Y sees them: the same steps as
    stage6_geo.fold_frame (frame, training-only TimeEffect at Dec Y-1, pool =
    usable non-outlier sales). Checked against the frame's prior_lp_detr."""
    train_rows = s.sale_year.between(BASE_START, Y - 1) & s.split.isin(("train",)) & ~s.ppsf_outlier
    sub = s[train_rows | (s.sale_year <= Y)]
    f = build_frame(sub, loc, train_rows[sub.index])
    f = f.merge(glob, on="sale_id", how="left", validate="one_to_one").set_index(sub.index)
    f["tk"], f["sale_date"], f["sale_year"] = sub.tk, sub.sale_date, sub.sale_year
    tr_mask = train_rows[sub.index] & usable(f)
    te = TimeEffect(f[tr_mask], trend_point(Y - 1))
    f["y_detr"] = f.lp - te(f.t.to_numpy(), f.moy.to_numpy())
    pool = f[usable(f) & ~s.loc[sub.index, "ppsf_outlier"]]
    flip = s.set_index("sale_id").first_sale_of_flip
    return pool[["sale_id", "tk", "sale_date", "y_detr"]].assign(first_sale_of_flip=pool.sale_id.map(flip).to_numpy())


def _latest_before(rows: pd.DataFrame, limit: pd.Series, hist: pd.DataFrame, exact: bool) -> pd.DataFrame:
    left = pd.DataFrame({"_i": rows.index, "tk": rows.tk.astype(str).to_numpy(), "limit": limit.to_numpy()})
    left = left.dropna(subset=["limit"])
    right = hist.assign(tk=hist.tk.astype(str)).rename(columns={"sale_date": "h_date", "sale_id": "h_id",
                                                               "y_detr": "h_y"})[["tk", "h_date", "h_id", "h_y"]]
    m = pd.merge_asof(left.sort_values("limit"), right.sort_values("h_date"), left_on="limit", right_on="h_date",
                      by="tk", direction="backward", allow_exact_matches=exact)
    return m.set_index("_i").reindex(rows.index)


def proxies(sc: pd.DataFrame, hist: pd.DataFrame, Y: int) -> pd.DataFrame:
    """Per scored sale: the model's feature sale, the proxy sale and the older
    (never-seen) sale, with detrended log prices and gaps in years."""
    cut = pd.Timestamp(f"{Y}-01-01")
    lim12 = (sc.sale_date - pd.DateOffset(months=OWN_MIN_MONTHS)).where(lambda x: x < cut, cut)
    feat = _latest_before(sc, lim12, hist, exact=False)
    # The model's feature (stage6_geo.fold_frame) must be exactly this sale.
    has = sc.has_prior_sale == 1
    if not (feat.h_id.notna() == has).all() or not np.allclose(feat.h_y[has], sc.prior_lp_detr[has], rtol=0, atol=1e-9):
        raise AssertionError(f"{Y}: rebuilt history does not reproduce the frame's own-prior-sale feature")
    clean = hist[~hist.first_sale_of_flip.astype(bool)]
    prox = _latest_before(sc, sc.sale_date - pd.DateOffset(months=PROXY_MIN_MONTHS), clean, exact=True)
    older = _latest_before(sc, feat.h_date.where(has), clean, exact=False)
    yrs = lambda d: (sc.sale_date - d).dt.days / 365.25  # noqa: E731
    return pd.DataFrame({"lq": prox.h_y, "gap": yrs(prox.h_date), "same_as_feature": (prox.h_id == feat.h_id) & has,
                         "lq_old": older.h_y, "gap_old": yrs(older.h_date)}, index=sc.index)


def fold_rows(s, glob, Y, spec, stage8) -> pd.DataFrame:
    prim = spec["models"]["primary"]
    cfg, cols = prim["config"], prim["features"]
    f = frames.fold(s, glob, Y, BASE_START, cfg["loc"])
    tr = f["train"][f["train"].sale_year >= cfg["train_start"]]
    sc = f["scored"]
    g = GBMX(cfg, cols, monotone=True).fit(tr, tr.y_detr)
    lv = g.predict(sc)
    if Y == TEST_YEAR:
        ref = pd.read_parquet(INTERIM_DIR / "frozen_primary_predictions_2025.parquet").set_index("sale_id").pred
    else:
        ref = stage8[(stage8.method == "primary") & (stage8.year == Y)].set_index("sale_id").pred
    d = np.abs(np.exp(lv) / ref.reindex(sc.sale_id).to_numpy() - 1).max()
    if not d < 1e-9:
        raise AssertionError(f"{Y}: refit differs from the stored frozen-spec predictions (max rel diff {d})")
    has = (sc.has_prior_sale == 1).to_numpy()
    up, dn = sc.copy(), sc.copy()
    up.loc[has, "prior_lp_detr"] += DELTA
    dn.loc[has, "prior_lp_detr"] -= DELTA
    beta = np.where(has, (g.predict(up) - g.predict(dn)) / (2 * DELTA), 0.0)
    cols_no = [c for c in cols if c not in GROUPS["own_prior_sale"]]
    pno = np.exp(GBMX(cfg, cols_no, monotone=True).fit(tr, tr.y_detr).predict(sc))
    px = proxies(sc, history(s, glob, Y, cfg["loc"]), Y)
    keep = ["sale_id", "tk", "sale_date", "neighborhood", "tier", "target_price", "bench_assessed_total",
            "has_benchmark", "land_use_y1", "bldg_area"]
    out = sc[keep].assign(year=Y, pred_model=np.exp(lv), pred_noown=pno, beta=beta).join(px)
    log(f"      {Y}: {len(out)} scored; proxy for {int(out.lq.notna().sum())}; same sale as the model's feature "
        f"{int(out.same_as_feature.sum())}; mean beta where used {beta[out.same_as_feature.to_numpy()].mean():.3f}")
    return out


def normalize(d: pd.DataFrame, col: str, target: str = "target_price") -> pd.Series:
    """Values divided by the method's median ratio within year and tier."""
    r = d[col] / d[target]
    return d[col] / r.groupby([d.year, d.tier]).transform("median")


# ------------------------------------------------------------ diagnostics ---

def repeat_sale_noise(s: pd.DataFrame, glob: pd.DataFrame, loc: str, last_year: int) -> tuple[pd.DataFrame, dict, dict]:
    """Case-Shiller per tier on consecutive clean sales of a parcel more than
    12 months apart (second sale <= last_year), detrended with fold 2025's
    training-only (2014-2024) time effect. d^2 = 2 sigma_p^2 + drift * years.
    3 x IQR trim on d. Returns the table, per-tier bootstrap draws of
    (sigma_p^2, drift) and per-tier standardized trimmed residuals (the shape
    for the noise sensitivity)."""
    h = history(s, glob, TEST_YEAR, loc)
    h = h[(h.sale_date.dt.year <= last_year) & ~h.first_sale_of_flip.astype(bool)].sort_values(["tk", "sale_date"])
    p = h.assign(prev_y=h.groupby("tk").y_detr.shift(), prev_d=h.groupby("tk").sale_date.shift()).dropna(
        subset=["prev_y"])
    p = p.assign(d=p.y_detr - p.prev_y, years=(p.sale_date - p.prev_d).dt.days / 365.25)
    p = p[p.years > 1].copy()
    p["sale_year"] = p.sale_date.dt.year
    p["neighborhood"] = s.set_index("sale_id").neighborhood.reindex(p.sale_id).to_numpy()
    p["tier"] = pd.concat([value_tiers(s, g, y) for y, g in p.groupby("sale_year")]).reindex(p.index)
    q1, q3 = p.d.quantile([0.25, 0.75])
    trim = (p.d < q1 - 3 * (q3 - q1)) | (p.d > q3 + 3 * (q3 - q1))
    rows, draws, shape = [], {}, {}
    rng = np.random.default_rng(SEED)
    for t in TIERS:
        g = p[(p.tier == t) & ~trim]
        sp2, dr = case_shiller(g.d.to_numpy(), g.years.to_numpy())
        idx = list(g.reset_index(drop=True).groupby("tk").indices.values())
        yr, dd = g.years.to_numpy(), g.d.to_numpy()
        boot = np.array([case_shiller(dd[i], yr[i]) for i in
                         (np.concatenate([idx[k] for k in rng.integers(0, len(idx), len(idx))])
                          for _ in range(N_NOISE_BOOT))])
        if sp2 <= 0 or (boot[:, 0] <= 0).mean() > 0.025:
            raise ValueError(f"tier {t}: sale-noise variance not identified (sigma_p^2 {sp2:.4f})")
        gb = g.assign(yb=np.floor(g.years).clip(upper=8)).groupby("yb")
        rb = gb.d.agg(lambda x: (1.4826 * np.median(np.abs(x - np.median(x)))) ** 2)
        w = np.sqrt(gb.size().to_numpy())
        rw = np.linalg.lstsq(np.c_[np.ones(len(rb)), gb.years.mean()] * w[:, None], rb.to_numpy() * w, rcond=None)[0]
        draws[t] = boot
        shape[t] = (dd - dd.mean()) / dd.std()
        rows.append({"tier": t, "pairs": len(g), "parcels": len(idx), "sigma_p2": sp2,
                     "sigma_p2_lo": np.percentile(boot[:, 0], 2.5), "sigma_p2_hi": np.percentile(boot[:, 0], 97.5),
                     "sigma_p": np.sqrt(sp2), "drift_per_year": dr, "drift_lo": np.percentile(boot[:, 1], 2.5),
                     "drift_hi": np.percentile(boot[:, 1], 97.5), "sigma_p2_robust_mad": rw[0] / 2,
                     "drift_robust_mad": rw[1], "residual_kurtosis": float(pd.Series(dd).kurt())})
    log(f"      repeat-sale pairs {len(p)} (trimmed {int(trim.sum())}), second sale <= {last_year}")
    return pd.DataFrame(rows), draws, shape


def cluster_boot(d: pd.DataFrame, n: int, seed: int):
    """Index arrays for a neighborhood-clustered bootstrap of d."""
    rng = np.random.default_rng(seed)
    codes = pd.factorize(d.neighborhood)[0]
    groups = [np.flatnonzero(codes == k) for k in range(codes.max() + 1)]
    for _ in range(n):
        yield np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])


def _q_rows(v: np.ndarray, P: np.ndarray, q: np.ndarray, sq2: np.ndarray) -> np.ndarray:
    """Disattenuated proxy PRB for each row of P and q (simulation, no leak)."""
    r = v[None, :] / P
    med = np.median(r, axis=1, keepdims=True)
    y = (r - med) / med
    x = q / LN2
    xc = x - x.mean(axis=1, keepdims=True)
    raw = (xc * (y - y.mean(axis=1, keepdims=True))).sum(axis=1) / (xc ** 2).sum(axis=1)
    lam = 1 - sq2.mean(axis=1) / q.var(axis=1)
    return np.where(lam > MIN_RELIABILITY, raw / lam, np.nan)


def evaluate(g: pd.DataFrame, v: np.ndarray, noise: np.ndarray, shape: np.ndarray, leak_correct: bool, seed: int,
             n_boot: int = N_BOOT, n_sim: int = N_SIM, lq: str = "lq", gap: str = "gap",
             same: str = "same_as_feature") -> dict:
    """One method in one tier. v: level-normalized values aligned with g.
    Observed PRB and Q with neighborhood-clustered bootstraps (noise
    parameters redrawn per draw); a perfectly calibrated null from the same
    simulation for both: P* = v exp(u + e), true log value log v + u, proxy
    = true log value + e_q. E = PRB - PRB_null, D = Q - Q_null."""
    p = g.target_price.to_numpy(float)
    ip = np.flatnonzero(g[lq].notna().to_numpy())
    x, gp = g[lq].to_numpy()[ip], g[gap].to_numpy()[ip]
    leak = (g.beta * g[same]).to_numpy(float)[ip] if leak_correct else np.zeros(len(ip))
    nb = noise[np.arange(max(n_boot, n_sim)) % len(noise)]

    def q_one(i, sp2, dr):
        q = proxy_prb(v[ip][i], p[ip][i], x[i])
        return (q,) + corrected_q(q, float(np.var(x[i])), np.clip(sp2 + dr * gp[i], 0, None), leak[i])
    # Observed, with bootstraps.
    prb_obs = prb(v, p)
    prb_b = np.array([prb(v[i], p[i]) for i in cluster_boot(g, n_boot, seed)])
    q_pt = q_one(np.arange(len(ip)), noise[:, 0].mean(), noise[:, 1].mean())
    q_b = np.array([q_one(i, *nb[b]) for b, i in enumerate(cluster_boot(g.iloc[ip], n_boot, seed))])
    q_unstable = float(np.isnan(q_b[:, 1]).mean())
    # Calibrated null.
    lr = np.log(p / v)
    s2 = (1.4826 * np.median(np.abs(lr - np.median(lr)))) ** 2
    rng = np.random.default_rng(seed + 1)
    sim_prb, sim_prb_r, sim_q = np.empty(n_sim), np.empty(n_sim), np.empty(n_sim)
    for k0 in range(0, n_sim, 200):
        k = np.arange(k0, min(k0 + 200, n_sim))
        sp2, dr = nb[k, 0], nb[k, 1]
        su = np.sqrt(np.clip(s2 - sp2, 0, None))[:, None]
        u = rng.standard_normal((len(k), len(v))) * su
        e = rng.standard_normal((len(k), len(v))) * np.sqrt(sp2)[:, None]
        er = shape[rng.integers(0, len(shape), (len(k), len(v)))] * np.sqrt(sp2)[:, None]
        V = np.broadcast_to(v, (len(k), len(v)))
        Pn = V * np.exp(u + e)
        sim_prb[k] = _stats(V, Pn)["prb"]
        sim_prb_r[k] = _stats(V, V * np.exp(u + er))["prb"]
        sq2 = np.clip(sp2[:, None] + dr[:, None] * gp[None, :], 0, None)
        qs = np.log(v[ip])[None, :] + u[:, ip] + rng.standard_normal((len(k), len(ip))) * np.sqrt(sq2)
        sim_q[k] = _q_rows(v[ip], Pn[:, ip], qs, sq2)
    qnull_unstable = float(np.isnan(sim_q).mean())
    e_d = prb_b - sim_prb[:n_boot]
    d_d = q_b[:, 1] - sim_q[:n_boot]
    ok = q_unstable <= MAX_UNSTABLE and qnull_unstable <= MAX_UNSTABLE
    pct = lambda a, lo: float(np.nanpercentile(a, 2.5 if lo else 97.5)) if ok else float("nan")  # noqa: E731
    q_null = float(np.nanmean(sim_q))
    return {"n": len(g), "mdape_ln": float(np.median(np.abs(v / p - 1))),
            "prb_obs": prb_obs, "prb_lo": np.percentile(prb_b, 2.5), "prb_hi": np.percentile(prb_b, 97.5),
            "prb_null": sim_prb.mean(), "null_lo": np.percentile(sim_prb, 2.5), "null_hi": np.percentile(sim_prb, 97.5),
            "prb_null_resid": sim_prb_r.mean(), "E": prb_obs - sim_prb.mean(),
            "e_lo": np.percentile(e_d, 2.5), "e_hi": np.percentile(e_d, 97.5),
            "noise_share": sim_prb.mean() / prb_obs if prb_obs else np.nan, "sigma_u2": float(np.clip(s2 - nb[:, 0], 0, None).mean()),
            "n_proxy": len(ip), "q_raw": q_pt[0], "q_raw_lo": np.percentile(q_b[:, 0], 2.5),
            "q_raw_hi": np.percentile(q_b[:, 0], 97.5), "leak_bias": q_pt[2], "reliability": q_pt[3],
            "q": q_pt[1], "q_lo": pct(q_b[:, 1], True), "q_hi": pct(q_b[:, 1], False), "q_null": q_null,
            "D": q_pt[1] - q_null, "d_lo": pct(d_d, True), "d_hi": pct(d_d, False),
            "unstable_draw_share": max(q_unstable, qnull_unstable)}


def proxy_bins(d: pd.DataFrame, cols: dict, seed: int) -> pd.DataFrame:
    """Median level-neutral ratio by proxy quintile within tier."""
    d = d[d.lq.notna()].copy()
    d["bin"] = pd.qcut(d.lq, N_BINS, labels=False) + 1
    rng = np.random.default_rng(seed)
    rows = []
    for m, c in cols.items():
        r = (normalize(d, c) / d.target_price).to_numpy()
        for b in range(1, N_BINS + 1):
            rb = r[(d.bin == b).to_numpy()]
            bs = [np.median(rng.choice(rb, len(rb))) for _ in range(500)]
            rows.append({"method": m, "bin": b, "n": len(rb), "median_ratio": np.median(rb),
                         "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5)})
    return pd.DataFrame(rows)


def calibration(d: pd.DataFrame, values: pd.Series, year_fe: bool) -> dict:
    y = np.log(d.target_price.to_numpy(float))
    X = pd.DataFrame({"lv": np.log(np.asarray(values, float))})
    if year_fe and d.year.nunique() > 1:
        X = X.join(pd.get_dummies(d.year.to_numpy(), prefix="y", drop_first=True, dtype=float))
    X = sm.add_constant(X)
    hc = sm.OLS(y, X).fit(cov_type="HC1")
    cl = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.neighborhood)[0]})
    b = hc.params["lv"]
    return {"n": len(d), "slope": b, "hc1_lo": b - 1.96 * hc.bse["lv"], "hc1_hi": b + 1.96 * hc.bse["lv"],
            "cluster_lo": b - 1.96 * cl.bse["lv"], "cluster_hi": b + 1.96 * cl.bse["lv"]}


def representativeness(d: pd.DataFrame) -> pd.DataFrame:
    tg = pd.read_csv(REPORTS_DIR / "stage8" / "tract_groups.csv", dtype={"tract": str}).set_index("tract")
    tract = d.tk.map(parcel_locations().set_index("tk").tract)
    d = d.assign(race=tract.map(tg.race_group).fillna("no group"), income=tract.map(tg.income_group).fillna("no group"))
    rows = []
    for name, g in [("all common sales", d), ("with proxy sale", d[d.lq.notna()]),
                    ("with an older never-seen sale", d[d.lq_old.notna()])]:
        r = {"subset": name, "n": len(g), "share_of_all": len(g) / len(d),
             "median_target_price": g.target_price.median(), "median_sqft": g.bldg_area.median(),
             "duplex_share": (g.land_use_y1.astype(str) == "8820").mean()}
        r.update({f"tier_{t}": (g.tier == t).mean() for t in TIERS})
        r.update({f"race: {k}": v for k, v in g.race.value_counts(normalize=True).sort_index().items()})
        r.update({f"income: {k}": v for k, v in g.income.value_counts(normalize=True).sort_index().items()})
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------- tradeoff ---

def tradeoff(cv: pd.DataFrame, noise: dict, shape: dict, seed: int) -> pd.DataFrame:
    """Scale log values around the fold-tier mean by k. Per fold Y and tier, the
    grid runs k = 1 .. 1/b in N_K steps, b = calibration slope (with year
    effects) on the out-of-fold predictions of CV folds before Y only; one
    extra point at k = b. Folds TRADEOFF_FOLDS (2021 has no earlier fold).
    Pooled over folds."""
    hist = cv
    steps = [(f"step {i}", i / (N_K - 1)) for i in range(N_K)] + [("calibrated (k = b)", None)]
    cols = ["mdape_ln", "prb_obs", "prb_null", "E", "q", "q_null", "D", "d_lo", "d_hi"]
    rows = []
    for t in TIERS:
        g = cv[(cv.tier == t) & cv.year.isin(TRADEOFF_FOLDS)].copy()
        lv = np.log(g.pred_model)
        mu = lv.groupby(g.year).transform("mean")
        b = {Y: calibration(hist[(hist.tier == t) & (hist.year < Y)],
                            hist[(hist.tier == t) & (hist.year < Y)].pred_model, True)["slope"]
             for Y in TRADEOFF_FOLDS}
        bY = g.year.map(b)
        for name, frac in steps:
            k = (1 + frac * (1 / bY - 1)) if frac is not None else bY
            g["pred_k"] = np.exp(mu + k * (lv - mu))
            # Scaling multiplies the own-sale leak by k as well.
            r = evaluate(g.assign(beta=g.beta * k), normalize(g, "pred_k").to_numpy(), noise[t], shape[t], True,
                         seed, n_boot=N_BOOT_CURVE, n_sim=N_BOOT_CURVE)
            rows.append({"tier": t, "point": name, "k_mean": float(k.mean()), "k_min": float(k.min()),
                         "k_max": float(k.max()), "slope_earlier_folds_mean": float(bY.mean()),
                         "calibration_slope": calibration(g, g.pred_k, True)["slope"],
                         **{c: r[c] for c in cols}})
        r = evaluate(g, normalize(g, "bench_assessed_total").to_numpy(), noise[t], shape[t], False, seed,
                     n_boot=N_BOOT_CURVE, n_sim=N_BOOT_CURVE)
        rows.append({"tier": t, "point": "assessor", "calibration_slope":
                     calibration(g, g.bench_assessed_total, True)["slope"], **{c: r[c] for c in cols}})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- run ---

def run_split(d: pd.DataFrame, label: str, noise: dict, shape: dict, seed: int, year_fe: bool,
              permit_sales: set) -> dict:
    out = {k: [] for k in ["eval", "bins", "slopes"]}
    for t in TIERS:
        g = d[d.tier == t]
        for m, c in METHODS.items():
            r = evaluate(g, normalize(g, c).to_numpy(), noise[t], shape[t], m == "model", seed)
            out["eval"].append({"split": label, "tier": t, "method": m, "robustness": "", **r})
            out["slopes"].append({"split": label, "tier": t, "method": m, **calibration(g, g[c], year_fe)})
        out["eval"].append({"split": label, "tier": t, "method": "model", "robustness": "older never-seen sale",
                            **evaluate(g, normalize(g, "pred_model").to_numpy(), noise[t], shape[t], False, seed,
                                       lq="lq_old", gap="gap_old")})
        nop = g[~g.sale_id.isin(permit_sales)]
        out["eval"].append({"split": label, "tier": t, "method": "model", "robustness": "no significant permit",
                            **evaluate(nop, normalize(nop, "pred_model").to_numpy(), noise[t], shape[t], True, seed)})
        out["bins"].append(proxy_bins(g, {"model": "pred_model", "assessor": "bench_assessed_total"}, seed)
                           .assign(split=label, tier=t))
    return out


def _style(a, ink, muted, grid):
    a.set_facecolor("#fcfcfb")
    a.grid(color=grid, lw=0.6)
    for sp in a.spines.values():
        sp.set_visible(False)
    a.tick_params(colors=muted)


def figure(ev: pd.DataFrame, bins: pd.DataFrame, path, label: str) -> None:
    ink, muted, grid = "#0b0b0b", "#898781", "#e1e0d9"
    fig, ax = plt.subplots(2, 3, figsize=(12, 7), sharey="row", facecolor="#fcfcfb")
    for j, t in enumerate(TIERS):
        a = ax[0, j]
        _style(a, ink, muted, grid)
        for i, m in enumerate(["model", "assessor"]):
            r = ev[(ev.tier == t) & (ev.method == m)].iloc[0]
            a.add_patch(plt.Rectangle((i - 0.3, r.null_lo), 0.6, r.null_hi - r.null_lo, color=grid, zorder=1))
            a.hlines(r.prb_null, i - 0.3, i + 0.3, color=muted, lw=2, zorder=2)
            a.errorbar(i, r.prb_obs, [[r.prb_obs - r.prb_lo], [r.prb_hi - r.prb_obs]], fmt=MARKERS[m], ms=8,
                       color=COLORS[m], lw=2, capsize=0, zorder=3)
        a.axhspan(-BAND, BAND, color=grid, alpha=0.35, zorder=0)
        a.axhline(0, color=muted, lw=0.8)
        a.set_xticks([0, 1], ["model", "assessor"], color=ink)
        a.set_xlim(-0.7, 1.7)
        a.set_title(f"{t} value tier", color=ink, fontsize=11)
        b = ax[1, j]
        _style(b, ink, muted, grid)
        for m in ["model", "assessor"]:
            r = bins[(bins.tier == t) & (bins.method == m)]
            b.errorbar(r.bin, r.median_ratio, [r.median_ratio - r.lo, r.hi - r.median_ratio], fmt=f"-{MARKERS[m]}",
                       ms=7, lw=2, color=COLORS[m], capsize=0, label=m)
        b.axhline(1, color=muted, lw=0.8)
        b.set_xticks(range(1, N_BINS + 1))
        b.set_xlabel("prior-sale proxy quintile (low to high)", color=muted)
    ax[0, 0].set_ylabel("within-tier PRB (marker: observed, 95% CI;\ngray: calibrated null, 95% band)", color=ink)
    ax[1, 0].set_ylabel("median level-neutral ratio", color=ink)
    ax[1, 0].legend(frameon=False)
    fig.suptitle(f"Within-tier PRB: observed vs a calibrated null, and ratios by an independent proxy ({label})",
                 color=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def tradeoff_figure(tc: pd.DataFrame, path) -> None:
    ink, muted, grid = "#0b0b0b", "#898781", "#e1e0d9"
    fig, ax = plt.subplots(2, 3, figsize=(12, 7.5), sharey="row", facecolor="#fcfcfb")
    for j, t in enumerate(TIERS):
        c = tc[(tc.tier == t) & tc.point.str.startswith("step")]
        cal = tc[(tc.tier == t) & (tc.point == "calibrated (k = b)")].iloc[0]
        a_ = tc[(tc.tier == t) & (tc.point == "assessor")].iloc[0]
        for i, (col, lab) in enumerate([("D", "D = Q - Q_null (0 = as regressive\nas a calibrated valuation)"),
                                        ("prb_obs", "within-tier PRB (sale prices)")]):
            a = ax[i, j]
            _style(a, ink, muted, grid)
            a.axhspan(-BAND, BAND, color=grid, alpha=0.35, zorder=0)
            a.axhline(0, color=muted, lw=0.8)
            a.plot(c.mdape_ln * 100, c[col], "-o", color=COLORS["model"], ms=6, lw=2, label="frozen model, k = 1 to 1/b")
            a.plot(cal.mdape_ln * 100, cal[col], "o", mfc="none", mec=COLORS["model"], ms=9, mew=2,
                   label="model rescaled to calibration (k = b)")
            a.plot(a_.mdape_ln * 100, a_[col], "s", color=COLORS["assessor"], ms=8, label="assessor")
            k1 = c.iloc[0]
            a.plot(k1.mdape_ln * 100, k1[col], "o", color=COLORS["model"], ms=11, mec="#fcfcfb", mew=2, zorder=4,
                   label="frozen model as is (k = 1)")
            r = c.iloc[-1]
            a.annotate(f"k={r.k_mean:.2f}", (r.mdape_ln * 100, r[col]), textcoords="offset points",
                       xytext=(6, 8), color=muted, fontsize=8)
            a.annotate(f"k={cal.k_mean:.2f}", (cal.mdape_ln * 100, cal[col]), textcoords="offset points",
                       xytext=(9, -3), color=muted, fontsize=8, va="center")
            if j == 0:
                a.set_ylabel(lab, color=ink)
            if i == 1:
                a.set_xlabel("level-neutral MdAPE (%)", color=muted)
            if i == 0:
                a.set_title(f"{t} value tier", color=ink, fontsize=11)
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Calibration vs equal ratios: scaling the model's values within tier (CV folds 2022-2024)",
                 color=ink)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="2021 fold only, outputs in sample/")
    ap.add_argument("--render-only", action="store_true", help="re-render verdict, figures, summary from saved tables")
    a = ap.parse_args()
    out = OUT / "sample" if a.sample else OUT
    out.mkdir(parents=True, exist_ok=True)
    cvl = "CV 2021-2024" if not a.sample else "CV 2021 (sample)"
    if a.render_only:
        render(out, cvl, a.sample)
        return
    years = [2021] if a.sample else CV_YEARS + [TEST_YEAR]
    spec = json.loads(SPEC_PATH.read_text())
    loc = spec["models"]["primary"]["config"]["loc"]
    s = frames.sales()
    glob = frames.global_table(s)
    stage8 = pd.read_parquet(INTERIM_DIR / "stage8_predictions.parquet")
    permit = set(s.sale_id[s.permit_significant.fillna(False).astype(bool)])

    with step("fold refits (frozen spec, no-own-sale variant, beta) and proxies", 7200):
        rows = pd.concat([fold_rows(s, glob, Y, spec, stage8) for Y in years], ignore_index=True)
        d = rows[rows.has_benchmark & rows.tier.isin(TIERS)].copy()
    with step("repeat-sale noise (Case-Shiller per tier)", 3600):
        noise_t, noise, shape = repeat_sale_noise(s, glob, loc, last_year=2024)
    with step("tier-specific time adjustment targets", 3600):
        from avm.stage3_assessor import tercile_target_prices
        from avm.time_adjust import seasonal_effects
        tt = tercile_target_prices(s, years, seasonal_effects(s[s.sale_year.between(2014, 2024)]))
    cv = d[d.year.isin(CV_YEARS)]
    with step("diagnostics 1-3", 7200):
        res = run_split(cv, cvl, noise, shape, SEED, True, permit)
        if not a.sample:
            r25 = run_split(d[d.year == TEST_YEAR], "2025", noise, shape, SEED, False, permit)
            res = {k: res[k] + r25[k] for k in res}
    with step("diagnostic 4: tier-specific time adjustment", 3600):
        d4 = d.assign(target_price=d.sale_id.map(tt)).dropna(subset=["target_price"])
        tier4 = []
        for split, g0 in [("CV", d4[d4.year.isin(CV_YEARS)]), ("2025", d4[d4.year == TEST_YEAR])]:
            for t in TIERS:
                gt = g0[g0.tier == t]
                if len(gt) < 30:
                    continue
                for m, c in METHODS.items():
                    base = d.loc[gt.index]
                    tier4.append({"split": split, "tier": t, "method": m, "n": len(gt),
                                  "prb_citywide_time_adj": prb(normalize(base, c).to_numpy(),
                                                               base.target_price.to_numpy(float)),
                                  "prb_tercile_time_adj": prb(normalize(gt, c).to_numpy(),
                                                              gt.target_price.to_numpy(float))})
    if a.sample:
        tc = pd.DataFrame()
        log("      tradeoff skipped in the sample: 2021 has no earlier CV fold")
    else:
        with step("tradeoff curve (folds 2022-2024, k grid from earlier CV folds)", 7200):
            tc = tradeoff(cv, noise, shape, SEED)

    ev, bins, slp = pd.DataFrame(res["eval"]), pd.concat(res["bins"]), pd.DataFrame(res["slopes"])
    t4 = pd.DataFrame(tier4)
    cov = d.groupby(["year", "tier"]).agg(n=("sale_id", "size"), with_proxy=("lq", lambda x: x.notna().sum()),
                                          proxy_is_model_feature=("same_as_feature", "sum"),
                                          with_older_sale=("lq_old", lambda x: x.notna().sum())).reset_index()
    for name, t in [("within_tier_eval.csv", ev), ("proxy_bins.csv", bins), ("calibration_slopes.csv", slp),
                    ("sale_noise.csv", noise_t), ("proxy_coverage.csv", cov),
                    ("proxy_representativeness.csv", representativeness(cv)),
                    ("tier_time_adjustment.csv", t4), ("tradeoff.csv", tc)]:
        t.to_csv(out / name, index=False)

    render(out, cvl, a.sample)


def _position(lo: float, hi: float) -> str:
    if np.isnan(lo):
        return "not identified"
    if hi < 0:
        return "below 0: more regressive against true value than a calibrated valuation"
    if inside_band(lo, hi):
        return "includes 0 or within +/-0.05"
    if lo > 0:
        return "above 0: less regressive against true value than a calibrated valuation"
    return "straddles the band edge"


def framing(tc: pd.DataFrame) -> str:
    """The calibration vs equal-ratios tradeoff, from tradeoff.csv only. Each
    qualitative claim is checked against the numbers before it is written."""
    lines = ["**The tradeoff.** Underwriting needs calibrated predictions: within a tier, a home valued at V "
             "sells for about V on average (calibration slope of log price on log value = 1). IAAO vertical "
             "equity asks for equal ratios across value (within-tier PRB within +/-0.05). Scaling the model's "
             "values within each tier by k moves along one curve between the two (CV folds 2022-2024; the k grid "
             "for each fold is fit on earlier folds):"]
    mono = []
    for t in TIERS:
        g = tc[tc.tier == t].set_index("point")
        k1, kx, cal, ass = g.loc["step 0"], g.loc[f"step {N_K - 1}"], g.loc["calibrated (k = b)"], g.loc["assessor"]
        f = lambda r: (f"slope {r.calibration_slope:.2f}, PRB {r.prb_obs:+.3f}, "  # noqa: E731
                       f"level-neutral MdAPE {r.mdape_ln:.1%}")
        lines.append(f"- {t} tier: rescaled to calibration (k = {cal.k_mean:.2f}): {f(cal)}. Frozen model (k = 1): "
                     f"{f(k1)}. Spread to k = {kx.k_mean:.2f}: {f(kx)}. Assessor: {f(ass)}.")
        steps = g[g.index.str.startswith("step")]
        mono.append(bool((np.diff(steps.prb_obs) > 0).all() and (np.diff(steps.calibration_slope) < 0).all()
                         and cal.prb_obs < k1.prb_obs))
    ass = tc[tc.point == "assessor"]
    k1 = tc[tc.point == "step 0"]
    if all(mono):
        lines.append("In every tier, PRB rises toward the IAAO band only as values are spread past calibration "
                     "(the calibration slope falls as k grows), and the calibrated rescale has the most negative "
                     f"PRB. The frozen model sits nearer calibration (slope {k1.calibration_slope.min():.2f} to "
                     f"{k1.calibration_slope.max():.2f}); the assessor sits at the spread end (slope "
                     f"{ass.calibration_slope.min():.2f} to {ass.calibration_slope.max():.2f}).")
    else:
        lines.append("The curve is not monotone in every tier; see the table.")
    return "\n".join(lines)


def render(out, cvl: str, sample: bool) -> None:
    """Rule, verdict.json, figures and summary from the saved tables."""
    reg = json.loads((OUT / "verdict.json").read_text())
    ev = pd.read_csv(out / "within_tier_eval.csv").fillna({"robustness": ""})
    bins, slp = pd.read_csv(out / "proxy_bins.csv"), pd.read_csv(out / "calibration_slopes.csv")
    noise_t, t4 = pd.read_csv(out / "sale_noise.csv"), pd.read_csv(out / "tier_time_adjustment.csv")
    tp = out / "tradeoff.csv"
    tc = pd.read_csv(tp) if tp.exists() and tp.stat().st_size > 1 else pd.DataFrame()
    main_ev = ev[(ev.split == cvl) & (ev.robustness == "")].set_index(["tier", "method"])
    keys = ["prb_obs", "prb_null", "e_lo", "e_hi", "d_lo", "d_hi"]
    tiers_m = [{"tier": t, **main_ev.loc[(t, "model"), keys].to_dict(),
                **main_ev.loc[(t, "model_no_own_sale"), ["d_lo", "d_hi"]].rename({"d_lo": "dn_lo", "d_hi": "dn_hi"}).to_dict()}
               for t in TIERS]
    tiers_a = [{"tier": t, **main_ev.loc[(t, "assessor"), keys].to_dict()} for t in TIERS]
    v_model, v_ass = verdict(tiers_m), verdict(tiers_a, with_noown=False)
    flags = [f"{r.method} {r.tier} tier ({r.split}): PRB {r.prb_citywide_time_adj:+.3f} -> {r.prb_tercile_time_adj:+.3f} "
             "with tier-specific time adjustment (moves into the +/-0.05 band)"
             for r in t4.itertuples() if r.prb_citywide_time_adj < -BAND and abs(r.prb_tercile_time_adj) <= BAND]
    result = {"split": cvl, "rule": "amended 2026-10-22 (D = Q - Q_null)", "model": v_model, "assessor": v_ass,
              "time_adjustment_flags": flags}
    reg = {k: v for k, v in reg.items() if k in ("preregistered", "amendments")}
    reg["results" if not sample else "sample_results"] = result
    (out / ("verdict.json" if not sample else "verdict_sample.json")).write_text(
        json.dumps(reg, indent=2, default=float) + "\n", encoding="utf-8")
    figure(ev[(ev.split == cvl) & (ev.robustness == "")], bins[bins.split == cvl], out / "prb_diagnostics.png", cvl)
    if len(tc):
        tradeoff_figure(tc, out / "tradeoff.png")

    def tier_lines(method):
        rows = []
        for t in TIERS:
            r = main_ev.loc[(t, method)]
            rows.append(f"- {t}: PRB {r.prb_obs:+.3f} [{r.prb_lo:+.3f}, {r.prb_hi:+.3f}], calibrated null "
                        f"{r.prb_null:+.3f}, E {r.E:+.3f} [{r.e_lo:+.3f}, {r.e_hi:+.3f}] "
                        f"({_position(r.e_lo, r.e_hi).split(':')[0]}); D {r.D:+.3f} [{r.d_lo:+.3f}, {r.d_hi:+.3f}] "
                        f"({_position(r.d_lo, r.d_hi)})")
        return "\n".join(rows)
    r3 = lambda df, cols: md(df[cols].round(3))  # noqa: E731
    (out / "stage11_summary.md").write_text("\n\n".join([
        f"# Stage 11: within-tier PRB ({cvl}{'' if sample else '; 2025 reported only'})",
        f"**Model verdict (rule as amended 2026-10-22): {v_model['verdict']}**; failing tiers "
        f"{v_model['failing_tiers']}. Assessor (same rule): {v_ass['verdict']}; failing tiers {v_ass['failing_tiers']}.",
        ("**Verdict.** " + VERDICT_WORDING.format(verdict=v_model["verdict"],
                                                    mid_e=f"{main_ev.loc[('mid', 'model'), 'E']:.3f}")
         if not sample else ""),
        "No v2 model was built: the pre-registered rule allowed fixes only after a finding of real compression.",
        "Model, per tier (" + cvl + "):\n" + tier_lines("model"),
        "Model without its own-prior-sale feature:\n" + tier_lines("model_no_own_sale"),
        "Assessor:\n" + tier_lines("assessor"),
        "Rule reasons (model): " + " | ".join(v_model["reasons"]),
        "Rule reasons (assessor): " + " | ".join(v_ass["reasons"]),
        framing(tc) if len(tc) else "Tradeoff: skipped in the sample.",
        "## Per tier: observed PRB, calibrated null, E, Q, Q_null, D",
        r3(ev, ["split", "tier", "method", "robustness", "n", "mdape_ln", "prb_obs", "prb_lo", "prb_hi", "prb_null",
                "prb_null_resid", "E", "e_lo", "e_hi", "n_proxy", "q_raw", "leak_bias", "reliability", "q", "q_lo",
                "q_hi", "q_null", "D", "d_lo", "d_hi"]),
        "## Calibration slopes", r3(slp, ["split", "tier", "method", "n", "slope", "hc1_lo", "hc1_hi", "cluster_lo",
                                          "cluster_hi"]),
        "## Tradeoff (CV folds 2022-2024): values scaled within tier by k",
        r3(tc, list(tc.columns)) if len(tc) else "skipped in the sample",
        "## Sale-price noise (repeat sales)", r3(noise_t, list(noise_t.columns)),
        "## Tier-specific time adjustment", r3(t4, list(t4.columns)) if len(t4) else "none",
        "Flags: " + ("; ".join(flags) if flags else "none"),
    ]) + "\n", encoding="utf-8")
    log(f"model verdict: {v_model['verdict']} | assessor: {v_ass['verdict']}")


if __name__ == "__main__":
    main()
