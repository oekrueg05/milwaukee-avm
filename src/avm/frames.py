"""Cached Stage 6 inputs shared by freeze, prospective and Stage 7.

The global spatial/roll table and each fold frame are pickled under
interim/frames/, keyed on the sha256 of sales_clean.parquet and
target_prices.parquet plus FRAMES_VERSION (bump when frame logic changes).
"""

import hashlib
import json

import pandas as pd

from avm.acquire import sha256_of
from avm.config import INTERIM_DIR
from avm.progress import log
from avm.spatial import static_features
from avm.stage4_hedonic import load
from avm.stage6_geo import fold_frame, roll_tables

CACHE = INTERIM_DIR / "frames"
FRAMES_VERSION = 1


def _key(*parts) -> str:
    base = {"v": FRAMES_VERSION, "sales": sha256_of(INTERIM_DIR / "sales_clean.parquet"),
            "targets": sha256_of(INTERIM_DIR / "target_prices.parquet"), "parts": [str(p) for p in parts]}
    return hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()[:16]


def _cached(name: str, key: str, build):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}_{key}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    obj = build()
    pd.to_pickle(obj, path)
    log(f"      cached {path.name}")
    return obj


def sales() -> pd.DataFrame:
    return load()


def global_table(s: pd.DataFrame) -> pd.DataFrame:
    def build():
        st = static_features()
        g = s[["sale_id", "tk"]].merge(st, on="tk", how="left").drop(columns="tk")
        return g.merge(roll_tables(s), on="sale_id", how="left")
    return _cached("glob", _key("glob"), build)


def fold(s: pd.DataFrame, glob: pd.DataFrame, Y: int, start: int, loc: str,
         train_splits: tuple[str, ...] = ("train",)) -> dict:
    return _cached(f"fold_{loc}_{start}_{Y}_{'-'.join(train_splits)}", _key("fold", Y, start, loc, train_splits),
                   lambda: fold_frame(s, Y, start, loc, glob, train_splits))
