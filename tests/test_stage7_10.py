"""Synthetic tests for comps (Stage 7), the group guardrail and portfolio
privacy (Stage 9), the frozen-spec check, the write-up lint (Stage 10) and
the docs dash check. No real data is read."""

import numpy as np
import pandas as pd
import pytest

from avm import comps as C
from avm.ci_checks import find_dashes
from avm.freeze import spec_diffs
from avm.stage9_divergence import LAG_NOTE, check_a, check_b, guardrail_verdict
from avm.stage9_page import leaked_identifiers, strip_blocks
from avm.writeup import Numbers, lint, render

MI = C.MILE_M


# ---- comparable sales -------------------------------------------------------

def _pool(rows):
    """rows: (sale_id, x_mi, y_mi, months_before, duplex)."""
    df = pd.DataFrame(rows, columns=["sale_id", "x_mi", "y_mi", "months_before", "duplex"])
    df["x_m"], df["y_m"] = df.x_mi * MI, df.y_mi * MI
    df["tk"] = "T" + df.sale_id
    for c in C.STRUCT:
        df[c] = 1.0
    return df


def _subject(duplex=0):
    s = pd.DataFrame({"sale_id": ["S"], "tk": ["TS"], "x_m": [0.0], "y_m": [0.0], "duplex": [duplex]})
    for c in C.STRUCT:
        s[c] = 1.0
    return s


W = pd.Series(1.0, index=C.STRUCT)
SD = pd.Series(1.0, index=C.STRUCT)


def test_comps_window_first_then_radius():
    # 3 sales at 0.4 mi but 18 months old; 3 at 0.9 mi within 12 months.
    pool = _pool([(f"old{i}", 0.4, 0, 18, 0) for i in range(3)] + [(f"new{i}", 0.9, 0, 6, 0) for i in range(3)])
    out = C.comps_for(_subject(), pool, W, SD)
    assert set(out.comp_sale_id) == {"new0", "new1", "new2"}
    assert (out.comp_window_months == 12).all() and (out.comp_radius_mi == 1.0).all()


def test_comps_tightest_tier_and_max_five():
    pool = _pool([(f"a{i}", 0.1, 0, 3, 0) for i in range(8)])
    out = C.comps_for(_subject(), pool, W, SD)
    assert len(out) == C.K_COMPS
    assert (out.comp_window_months == 12).all() and (out.comp_radius_mi == 0.5).all()


def test_comps_same_property_type_only_and_none_when_too_few():
    pool = _pool([(f"d{i}", 0.1, 0, 3, 1) for i in range(5)] + [("s0", 0.1, 0, 3, 0), ("s1", 2.0, 0, 3, 0)])
    out = C.comps_for(_subject(duplex=0), pool, W, SD)
    assert out.empty  # only one single-family sale within 1.25 mi: fewer than MIN_COMPS


def test_comps_never_the_subjects_own_parcel():
    # The subject parcel's own earlier sale is the closest, most similar sale; it must not count,
    # so the 0.5 mi tier has only 2 others and the search widens to 1 mi.
    pool = _pool([("own", 0.0, 0, 6, 0), ("n0", 0.2, 0, 6, 0), ("n1", 0.3, 0, 6, 0), ("m0", 0.8, 0, 6, 0)])
    pool.loc[pool.sale_id == "own", "tk"] = "TS"
    out = C.comps_for(_subject(), pool, W, SD)
    assert "own" not in set(out.comp_sale_id)
    assert set(out.comp_sale_id) == {"n0", "n1", "m0"} and (out.comp_radius_mi == 1.0).all()


def test_comps_refuse_missing_tax_key():
    pool = _pool([(f"a{i}", 0.1, 0, 3, 0) for i in range(3)])
    with pytest.raises(ValueError):
        C.comps_for(_subject().drop(columns="tk"), pool, W, SD)


def test_eligible_pool_window_and_exclusions():
    rows = pd.DataFrame({"sale_id": list("abcde"),
                         "sale_date": pd.to_datetime(["2024-06-15", "2022-12-15", "2025-02-01", "2024-03-01",
                                                      "2024-09-01"]),
                         "x_m": 0.0, "y_m": 0.0})
    sales = pd.DataFrame({"sale_id": list("abcde"), "first_sale_of_flip": [False, False, False, True, False],
                          "ratio_outlier_iaao": [False] * 5, "ppsf_outlier": [False, False, False, False, True]})
    out = C.eligible_pool(rows, sales, "2025-01-01")
    assert list(out.sale_id) == ["a"]  # b too old, c after the valuation date, d flip, e outlier
    assert out.months_before.iloc[0] == 7


def test_eligible_pool_refuses_rows_missing_from_sales_table():
    rows = pd.DataFrame({"sale_id": ["x"], "sale_date": pd.to_datetime(["2024-06-01"]), "x_m": 0.0, "y_m": 0.0})
    sales = pd.DataFrame({"sale_id": ["y"], **{f: [False] for f in C.EXCLUDE_FLAGS}})
    with pytest.raises(ValueError):
        C.eligible_pool(rows, sales, "2025-01-01")


# ---- group guardrail --------------------------------------------------------

def test_guardrail_all_three_pass_under_assessed():
    a = check_a(np.log(0.955), np.log(0.979))
    b = check_b(0.954, 0.921, 0.977, np.log(0.966))
    c, v = guardrail_verdict(np.log(0.966), a, b, np.log(0.972), True, True)
    assert a and b and c and v == "under-assessed vs market share"


def test_guardrail_fails_only_c_gives_lag_note():
    c, v = guardrail_verdict(0.007, True, True, -0.002, False, True)
    assert not c and v == LAG_NOTE


def test_guardrail_c_needs_same_direction():
    c, _ = guardrail_verdict(0.02, True, True, -0.01, True, True)
    assert not c


def test_guardrail_checks_conflict_no_claim():
    # corrected says over-assessed, sales check says under-assessed: both exclude zero
    assert check_a(0.006, 0.016)
    b = check_b(0.987, 0.976, 0.996, 0.011)
    _, v = guardrail_verdict(0.011, True, b, 0.018, True, False)
    assert not b and v == "no claim: 2025 sales check does not agree with a CI excluding one"


def test_guardrail_corrected_interval_includes_zero():
    a = check_a(-0.010, 0.006)
    _, v = guardrail_verdict(-0.002, a, False, 0.0, False, False)
    assert not a and v == "no claim: corrected shift's CI includes zero"


# ---- portfolio privacy ------------------------------------------------------

HTML = ("<title><!--portfolio-->Check<!--/portfolio--><!--private-->Review<!--/private--></title>"
        "<p><!--private-->1234567890 at 123 N MAIN ST<!--/private--></p>"
        "<script>/*<!--private-->*/\nconst ROWS = ['1234567890'];\n/*<!--/private-->*/\n"
        "/*<!--portfolio-->*/\nconst P = 1;\n/*<!--/portfolio-->*/\n</script>")


def test_strip_blocks_portfolio_removes_private_markup_and_script():
    out = strip_blocks(HTML, "private", "portfolio")
    assert "1234567890" not in out and "MAIN ST" not in out and "ROWS" not in out
    assert "<title>Check</title>" in out and "const P = 1;" in out and "<!--" not in out


def test_strip_blocks_private_keeps_private_content():
    out = strip_blocks(HTML, "portfolio", "private")
    assert "<title>Review</title>" in out and "1234567890" in out and "const P" not in out


def test_leaked_identifiers_finds_planted_values_only():
    clean = strip_blocks(HTML, "private", "portfolio")
    ids = ["1234567890", "123 N MAIN ST"]
    assert leaked_identifiers(clean, ids) == []
    assert leaked_identifiers(clean + "<p>123 N MAIN ST</p>", ids) == ["123 N MAIN ST"]


# ---- frozen spec check ------------------------------------------------------

def test_spec_diffs_ignore_commit_stamp_only():
    old = {"frozen_at_commit": "abc", "models": {"primary": {"config": {"num_leaves": 127}}}, "seed": 1}
    assert spec_diffs(old, {**old, "frozen_at_commit": "def"}) == []
    assert spec_diffs(old, {**old, "models": {"primary": {"config": {"num_leaves": 63}}}}) == ["models"]


# ---- write-up lint ----------------------------------------------------------

def test_lint_accepts_placeholders_and_whitelisted_constants():
    lint("Error {{mdape}} on {{n}} sales in 2025; 80% ranges, 0.75 mi, 3 to 5 comps, Stage 3, Q1, Jan 1, F6.", "t")


@pytest.mark.parametrize("text", ["about 42 homes", "a rate of 20.80", "an — dash", "a range – here"])
def test_lint_rejects_stray_digits_and_dashes(text):
    with pytest.raises(ValueError):
        lint(text, "t")


def test_render_fills_placeholders_and_refuses_missing_keys():
    N = Numbers()
    N.add("n", 4197, "4,197", "test")
    text, used = render("Scored on {{n}} sales.", N, "t")
    assert text == "Scored on 4,197 sales." and used == {"n"}
    with pytest.raises(KeyError):
        render("Missing {{nope}}.", N, "t")


# ---- docs dash check --------------------------------------------------------

def test_find_dashes(tmp_path):
    (tmp_path / "ok.md").write_text("plain - hyphen\n", encoding="utf-8")
    (tmp_path / "bad.md").write_text("line one\nan — dash\n", encoding="utf-8")
    hits = find_dashes([tmp_path])
    assert len(hits) == 1 and hits[0].endswith("bad.md:2")


# ---- privacy check ----------------------------------------------------------
# Fake values are assembled at runtime so this file itself passes the repo scan.

from avm.ci_checks import scan_text  # noqa: E402

FAKE_TK = "98765" + "43210"
FAKE_ADDR = "7777 N" + " FAKEVILLE ST"


def test_privacy_flags_tax_keys_but_not_decimals_hashes_or_synthetic():
    assert scan_text("a.md", f"parcel {FAKE_TK} here")
    assert not scan_text("a.md", f"ratio 0.{FAKE_TK} and sha 4d{FAKE_TK}ab and id {FAKE_TK}1")
    assert not scan_text("a.md", "synthetic 1234567890")


def test_privacy_flags_street_addresses_but_not_synthetic():
    assert scan_text("a.md", f"sold at {FAKE_ADDR} in May")
    assert scan_text("a.md", "the 888-890 S" + " Fakeville Ave complex")
    assert not scan_text("a.md", "example 123 N MAIN ST")


def test_privacy_flags_unredacted_dictionary_rows_and_owner_columns():
    dd = "field,inferred_type,top_values,examples\nOwner1,text,DOE JANE (3),DOE JANE\nYR_BUILT,integer,1925 (9),1925\n"
    assert scan_text("reports/stage1/data_dictionary/x.csv", dd)
    ok = "field,inferred_type,top_values,examples\nOwner1,text,(redacted),(redacted)\nYR_BUILT,integer,1925 (9),1925\n"
    assert not scan_text("reports/stage1/data_dictionary/x.csv", ok)
    assert scan_text("reports/x.csv", "sale_id,OWNER_NAME_1\n1,DOE JANE\n")


def test_privacy_exact_match_of_private_values():
    assert scan_text("a.md", "see W FAKEVILLE ST unit", exact={"W FAKEVILLE ST unit"})
