"""Address parsing and MAI matching on synthetic rows. Address strings are
joined at runtime (A(...)) and tax keys are short labels, so this file
itself passes the privacy check."""

import pandas as pd
import pytest

from avm.address import (AddressError, MaiRow, display_addresses, mai_rows, match, nearby_numbers, ordinal, parse_address,
                         resolve)


def A(*parts):
    return " ".join(parts)


def test_duplex_short_range_with_city_and_zip():
    p = parse_address(A("2876-78", "N.", "40th", "St,", "Milwaukee,", "WI", "53210"))
    assert (p.lo, p.hi, p.sfx, p.dir, p.street, p.sttype, p.zip5) == (2876, 2878, "", "N", "40TH", "ST", "53210")
    assert p.normalized == A("2876-2878", "N", "40TH", "ST")
    assert parse_address(A("2876", "-", "2878", "n", "40", "street")).normalized == p.normalized


def test_suffix_units_and_omitted_parts():
    assert parse_address(A("120A", "N", "Main", "St")).sfx == "A"
    assert parse_address(A("120ADJ", "N", "Main", "St")).sfx == "ADJ"
    p = parse_address(A("124", "1/2", "N", "Main", "St"))
    assert (p.sfx, p.dir, p.street) == ("1/2", "N", "MAIN") and match(p, ROWS) == []
    for unit in ["Apt 2", "Unit B", "#2", "Upper", "rear"]:
        p = parse_address(A("120", "W", "Main", "St", unit))
        assert p.unit_ignored and (p.dir, p.street, p.sttype) == ("W", "MAIN", "ST")
    p = parse_address(A("120", "main"))
    assert (p.dir, p.street, p.sttype, p.unit_ignored) == (None, "MAIN", None, False)


def test_streets_named_like_directions_or_types():
    assert (lambda p: (p.dir, p.street, p.sttype))(parse_address(A("120", "W", "North", "Avenue"))) == ("W", "NORTH", "AV")
    assert (lambda p: (p.dir, p.street, p.sttype))(parse_address(A("120", "North", "Ave"))) == (None, "NORTH", "AV")
    assert (lambda p: (p.dir, p.street, p.sttype))(parse_address(A("120", "N", "Terrace"))) == ("N", "TERRACE", None)
    assert (lambda p: (p.dir, p.street, p.sttype))(parse_address(A("120", "N", "Court", "St"))) == ("N", "COURT", "ST")


def test_ordinals_and_aliases():
    assert [ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 101, 111)] == \
        ["1ST", "2ND", "3RD", "4TH", "11TH", "12TH", "13TH", "21ST", "101ST", "111TH"]
    assert parse_address(A("120", "N", "1", "St")).street == "1ST"
    assert parse_address(A("120", "N", "22nd")).street == "22ND"
    assert parse_address(A("120", "W", "MLK", "Dr")).street == "MARTIN L KING JR"
    assert parse_address(A("120", "N", "Dr", "Martin", "Luther", "King", "Jr", "Dr")).street == "MARTIN L KING JR"
    assert parse_address(A("120", "W", "Saint", "Paul", "Ave")).street == "ST PAUL"


@pytest.mark.parametrize("bad", ["", "Main St", A("2878-76", "N", "Main", "St"), "120"])
def test_unreadable_addresses_raise(bad):
    with pytest.raises(AddressError):
        parse_address(bad)


ROWS = [MaiRow("TA", 120, "", "N", "MAIN", "ST", "53201"), MaiRow("TA", 122, "", "N", "MAIN", "ST", "53201"),
        MaiRow("TB", 124, "", "N", "MAIN", "ST", "53201"), MaiRow("TF", 124, "A", "N", "MAIN", "ST", "53201"),
        MaiRow("TC", 130, "", "N", "MAIN", "ST", "53201"), MaiRow("TD", 130, "", "N", "MAIN", "ST", "53201"),
        MaiRow("TE", 120, "", "S", "MAIN", "ST", "53202"), MaiRow("TG", 140, "", "N", "MAIN", "ST", "53201"),
        MaiRow("TH", 150, "", "N", "MAIN", "ST", "53201"), MaiRow("TX", 121, "", "N", "MAIN", "ST", "53201")]


def test_match_ranges_directions_suffixes_and_zip():
    m = lambda *parts: match(parse_address(A(*parts)), ROWS)  # noqa: E731
    assert m("120-22", "N", "Main", "St") == ["TA"]
    assert m("122", "N", "Main") == ["TA"]
    assert m("120", "Main", "St") == ["TA", "TE"]                # no direction: both sides of the city
    assert m("120", "Main", "St", "53202") == ["TE"]             # ZIP breaks the tie
    assert m("120", "N", "Main", "St", "53299") == ["TA"]        # a ZIP that matches nothing is ignored
    assert m("124", "N", "Main", "St") == ["TB", "TF"]           # plain number also matches its suffixed variant
    assert m("124A", "N", "Main", "St") == ["TF"]
    assert m("130", "N", "Main", "St") == ["TC", "TD"]           # two parcels share the address
    assert m("126", "N", "Main", "St") == []
    assert m("120", "N", "Elm", "St") == []
    assert m("121", "N", "Main", "St") == ["TX"]                 # odd side
    assert m("120-22", "N", "Main", "St") == ["TA"]              # 120-122 skips 121 across the street
    assert m("120-150", "N", "Main", "St") == ["TA", "TH"]       # wide range: end numbers only


def test_nearby_numbers_at_most_five():
    near = nearby_numbers(parse_address(A("126", "N", "Main", "St")), ROWS)
    assert len(near) == 5
    assert near[:2] == [A("124", "N", "MAIN", "ST"), A("124A", "N", "MAIN", "ST")]
    assert A("121", "N", "MAIN", "ST") not in near                 # other side of the street comes last
    assert nearby_numbers(parse_address(A("126", "N", "Elm", "St")), ROWS) == []


def _mai(rows):
    cols = ["TAXKEY", "HSE_NBR", "SFX", "DIR", "STREET", "STTYPE", "UNIT_NBR", "ZIP_CODE"]
    return pd.DataFrame(rows, columns=cols)


def test_display_addresses_and_mai_rows():
    tk1, tk2 = str(10 ** 9 + 7), str(10 ** 9 + 9)  # 10-digit keys built at runtime
    a = _mai([[tk1, "120", "", "N", "MAIN", "ST", "", "532011234"], [tk1, "122", "", "N", "MAIN", "ST", "", "532011234"],
              [tk2, "456", "", "S", "MAIN", "ST", "", "532021234"]])
    d = display_addresses(a).set_index("tk")
    assert d.loc[tk1, "address"] == A("120-122", "N", "MAIN", "ST") and d.loc[tk1, "zip5"] == "53201"
    assert d.loc[tk2, "address"] == A("456", "S", "MAIN", "ST")
    r = mai_rows(a)
    assert list(r.columns) == ["tk", "hn", "sfx", "dir", "street", "sttype", "zip5"] and r.hn.tolist() == [120, 122, 456]
    with pytest.raises(ValueError):
        mai_rows(_mai([[tk1, "12O", "", "N", "MAIN", "ST", "", "532011234"]]))  # letter O, not zero


def test_resolve_suffix_variants_are_always_listed():
    r = lambda *parts, valued=("TA", "TB", "TF", "TC", "TD", "TE"): resolve(parse_address(A(*parts)), ROWS, set(valued))  # noqa: E731
    assert r("124", "N", "Main", "St") == ("multiple_matches", ["TB", "TF"], "this number exists with and without a letter suffix", [])
    # even when only the plain parcel has a model value, the suffixed one is not set aside silently
    assert r("124", "N", "Main", "St", valued=("TB",)).status == "multiple_matches"
    assert r("124A", "N", "Main", "St")[:2] == ("ok", ["TF"])


def test_resolve_shared_address_and_no_match():
    r = lambda *parts, valued: resolve(parse_address(A(*parts)), ROWS, set(valued))  # noqa: E731
    assert r("130", "N", "Main", "St", valued=("TC", "TD")) == ("multiple_matches", ["TC", "TD"], "several parcels share this address", [])
    ok = r("130", "N", "Main", "St", valued=("TD",))
    assert ok[:2] == ("ok", ["TD"]) and "1 other parcel" in ok.note
    assert r("120-22", "N", "Main", "St", valued=())[:2] == ("ok", ["TA"])
    miss = r("126", "N", "Main", "St", valued=())
    assert miss.status == "no_match" and 0 < len(miss.nearby) <= 5
