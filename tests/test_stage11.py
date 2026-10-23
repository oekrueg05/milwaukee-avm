"""Synthetic tests for Stage 11 (within-tier PRB diagnostics). No real data."""

import numpy as np
import pandas as pd
import pytest

from avm import stage11_prb as S

RNG = np.random.default_rng(7)


def _frame(value, price, n_nbhd=40):
    n = len(value)
    return pd.DataFrame({"neighborhood": np.arange(n) % n_nbhd, "year": 2021, "tier": "low",
                         "target_price": price, "val": value})


def test_case_shiller_recovers_noise_and_drift():
    n, sp, drift = 40000, 0.15, 0.004
    years = RNG.uniform(1, 10, n)
    d = RNG.normal(0, sp, n) - RNG.normal(0, sp, n) + RNG.normal(0, np.sqrt(drift * years))
    sp2, dr = S.case_shiller(d, years)
    assert sp2 == pytest.approx(sp ** 2, rel=0.05) and dr == pytest.approx(drift, rel=0.15)


def _calibrated(n, spread=1.0):
    # Calibrated value (truth = value + u), a noisy sale price, and a noisy earlier-sale proxy of the truth.
    # spread > 1: true values spread more than the values, i.e. the values are compressed.
    lv = RNG.normal(12, 0.4, n)
    lt = 12 + spread * (lv - 12) + RNG.normal(0, 0.12, n)
    price = np.exp(lt + RNG.normal(0, 0.15, n))
    return _frame(np.exp(lv), price).assign(lq=lt + RNG.normal(0, 0.15, n), gap=3.0, beta=0.0,
                                            same_as_feature=False)


NOISE = np.column_stack([np.full(300, 0.15 ** 2), np.zeros(300)])
SHAPE = RNG.standard_normal(5000)


def test_calibrated_valuation_has_negative_q_and_zero_d():
    d = _calibrated(6000)
    r = S.evaluate(d, d.val.to_numpy(), NOISE, SHAPE, False, seed=1, n_boot=300, n_sim=300)
    assert r["prb_obs"] < -0.02 and r["e_lo"] <= 0 <= r["e_hi"]           # PRB negative from noise alone
    assert r["q"] < -0.02 and r["q_null"] < -0.02                          # a calibrated value still has negative Q
    assert r["d_lo"] <= 0 <= r["d_hi"]                                     # but no excess over the null


def test_compressed_valuation_has_negative_d():
    d = _calibrated(6000, spread=1.3)
    r = S.evaluate(d, d.val.to_numpy(), NOISE, SHAPE, False, seed=1, n_boot=300, n_sim=300)
    assert r["d_hi"] < 0


def test_proxy_prb_finds_real_compression_and_corrections_recover_it():
    n = 20000
    lv = RNG.normal(12, 0.4, n)                          # true log value
    eq = RNG.normal(0, 0.15, n)                          # proxy noise
    lq = lv + eq
    beta = 0.3                                           # value leans on the (noisy) proxy
    lval = 12 + 0.8 * (lv - 12) + beta * eq              # compressed: slope 0.8 on truth
    price = np.exp(lv + RNG.normal(0, 0.1, n))
    q_raw = S.proxy_prb(np.exp(lval), price, lq)
    q, bias, lam = S.corrected_q(q_raw, float(np.var(lq)), np.full(n, 0.15 ** 2), np.full(n, beta))
    # True proxy PRB in log2 units: ln2 * (0.8 - 1) = -0.139.
    assert q == pytest.approx(np.log(2) * -0.2, abs=0.02)
    assert bias > 0 and q_raw > q                         # leak and attenuation both pull toward zero/positive
    assert lam == pytest.approx(1 - 0.15 ** 2 / np.var(lq), rel=1e-9)


def test_proxy_prb_is_zero_without_compression_despite_price_noise():
    n = 20000
    lv = RNG.normal(12, 0.4, n)
    lq = lv + RNG.normal(0, 0.15, n)
    price = np.exp(lv + RNG.normal(0, 0.2, n))
    assert abs(S.proxy_prb(np.exp(lv), price, lq)) < 0.02          # regular PRB would be strongly negative
    assert S.prb(np.exp(lv), price) < -0.05


def _t(tier, obs, null, e, dd, dn):
    return {"tier": tier, "prb_obs": obs, "prb_null": null, "e_lo": e[0], "e_hi": e[1], "d_lo": dd[0], "d_hi": dd[1],
            "dn_lo": dn[0], "dn_hi": dn[1]}


def test_verdict_rule():
    art = _t("low", -0.20, -0.15, (-0.06, 0.02), (-0.03, 0.04), (-0.04, 0.02))
    comp = _t("low", -0.20, -0.05, (-0.20, -0.08), (-0.12, -0.05), (-0.10, -0.03))
    ok = _t("high", 0.01, -0.01, (-0.02, 0.04), (-0.02, 0.02), (-0.02, 0.02))
    assert S.verdict([art, ok])["verdict"] == "mostly measurement artifact"
    assert S.verdict([comp, ok])["verdict"] == "real compression"
    # Q and Q_noown disagree: mixed, and the disagreement is named.
    split = _t("low", -0.20, -0.05, (-0.20, -0.08), (-0.12, -0.05), (-0.03, 0.02))
    v = S.verdict([split])
    assert v["verdict"] == "mixed, not resolved" and "disagree" in v["reasons"][0]
    # Noise explains less than half: not an artifact even if Q is null.
    weak = _t("low", -0.20, -0.05, (-0.06, 0.02), (-0.03, 0.04), (-0.04, 0.02))
    assert S.verdict([weak])["verdict"] == "mixed, not resolved"
    assert S.verdict([ok])["verdict"] == "no failing tier"
    # A Q interval that is not identified can be neither artifact nor compression.
    nan = _t("low", -0.20, -0.15, (-0.06, 0.02), (float("nan"), float("nan")), (-0.04, 0.02))
    v = S.verdict([nan])
    assert v["verdict"] == "mixed, not resolved" and "not identified" in v["reasons"][0]
    assert np.isnan(S.corrected_q(-0.1, 0.05, np.full(10, 0.045), np.zeros(10))[0])
    # Assessor form: no Q_noown condition.
    a = dict(comp)
    del a["dn_lo"], a["dn_hi"]
    assert S.verdict([a], with_noown=False)["verdict"] == "real compression"
