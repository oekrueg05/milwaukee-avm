"""Download raw datasets from data.milwaukee.gov and record them in a manifest.

Usage:
    python -m avm.acquire --sample     # small subset for testing
    python -m avm.acquire              # everything in sources.SOURCES
    python -m avm.acquire --force      # re-download even if unchanged
    python -m avm.acquire --refresh-current   # also refresh pinned daily-updated files

A file is skipped when the manifest already has it with the same URL and the
same catalog last_modified, and the file on disk matches the recorded sha256.
"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from avm.config import CKAN_API, DATA_DIR, MANIFEST_PATH, RAW_DIR, REPORTS_DIR
from avm.progress import log
from avm.sources import SOURCES, Source

CONNECT_TIMEOUT_S = 30   # no TCP connection within this -> error
READ_TIMEOUT_S = 120     # no bytes received for this long -> error
FILE_TIMEOUT_S = 1800    # whole file (largest is ~200 MB) -> error

MANIFEST_COLS = [
    "key", "package", "resource_name", "resource_id", "url", "local_path",
    "bytes", "sha256", "catalog_last_modified", "downloaded_at_utc", "daily_updated",
]

_package_cache: dict[str, dict] = {}


def get_package(package: str) -> dict:
    if package not in _package_cache:
        r = requests.get(f"{CKAN_API}/package_show", params={"id": package},
                         timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S))
        r.raise_for_status()
        body = r.json()
        if not body.get("success"):
            raise RuntimeError(f"CKAN package_show failed for {package}: {body}")
        _package_cache[package] = body["result"]
    return _package_cache[package]


def find_resource(src: Source) -> dict:
    # Some catalog names carry trailing spaces ("2015 "), so compare stripped.
    matches = [r for r in get_package(src.package)["resources"]
               if (r["name"] or "").strip() == src.resource_name]
    if len(matches) != 1:
        raise RuntimeError(
            f"{src.key}: expected exactly 1 resource named {src.resource_name!r} "
            f"in package {src.package}, found {len(matches)}"
        )
    return matches[0]


def direct_resource(src: Source) -> dict:
    """A direct-URL source: the server's Last-Modified header stands in for
    the catalog's last_modified."""
    r = requests.head(src.url, allow_redirects=True, timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S))
    r.raise_for_status()
    return {"url": src.url, "id": "", "format": "", "last_modified": r.headers.get("Last-Modified", "")}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    """Stream to a .part file, then rename. Raises (and removes the .part) on
    a connect/read stall, on exceeding FILE_TIMEOUT_S in total, or on a size
    mismatch with Content-Length."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    t0, n = time.monotonic(), 0
    try:
        with requests.get(url, stream=True, timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S)) as r:
            r.raise_for_status()
            expected = None if r.headers.get("Content-Encoding") else r.headers.get("Content-Length")
            with open(part, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    n += len(chunk)
                    if time.monotonic() - t0 > FILE_TIMEOUT_S:
                        raise TimeoutError(f"{url}: exceeded {FILE_TIMEOUT_S}s ({n:,} bytes so far)")
        if expected is not None and n != int(expected):
            raise IOError(f"{url}: got {n:,} bytes, Content-Length says {int(expected):,}")
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(dest)


def download_arcgis_layer(layer_url: str, dest: Path, page: int = 1000) -> None:
    """Page through an ArcGIS REST layer and save all features as one GeoJSON."""
    features, offset, t0 = [], 0, time.monotonic()
    while True:
        if time.monotonic() - t0 > FILE_TIMEOUT_S:
            raise TimeoutError(f"{layer_url}: exceeded {FILE_TIMEOUT_S}s after {len(features)} features")
        r = requests.get(f"{layer_url}/query", timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S), params={
            "where": "1=1", "outFields": "*", "f": "geojson", "outSR": 4326,
            "resultOffset": offset, "resultRecordCount": page, "orderByFields": "OBJECTID",
        })
        r.raise_for_status()
        body = r.json()
        if "error" in body:
            raise RuntimeError(f"ArcGIS query failed for {layer_url}: {body['error']}")
        batch = body.get("features", [])
        features += batch
        if len(batch) < page:
            break
        offset += page
    count = requests.get(f"{layer_url}/query", timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
                         params={"where": "1=1", "returnCountOnly": "true", "f": "json"}).json()["count"]
    if count != len(features):
        raise RuntimeError(f"{layer_url}: server reports {count} features, fetched {len(features)}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")


def load_manifest() -> pd.DataFrame:
    if MANIFEST_PATH.exists():
        return pd.read_csv(MANIFEST_PATH, dtype=str)
    return pd.DataFrame(columns=MANIFEST_COLS)


def save_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    out = manifest.reset_index(drop=True)[MANIFEST_COLS].sort_values("key", ignore_index=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(MANIFEST_PATH, index=False)
    # Tracked copy in the repo so the source record survives with the code.
    tracked = REPORTS_DIR / "stage1" / "manifest.csv"
    tracked.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(tracked, index=False)
    return out


def acquire(sources: list[Source], force: bool = False, refresh_current: bool = False) -> pd.DataFrame:
    """Download each source. The manifest is saved after every file, so an
    interrupted run resumes by re-running: finished files are skipped.

    Daily-updated sources (Source.daily) are PINNED: once downloaded, they
    are not refreshed unless refresh_current (or force) is set, so a re-run
    reproduces the same inputs (rule 8). Their sha256 is in the manifest."""
    manifest = load_manifest().set_index("key", drop=False)
    if "daily_updated" not in manifest.columns:
        manifest["daily_updated"] = ""
    n_src = len(sources)
    for i, src in enumerate(sources, 1):
        tag = f"[{i}/{n_src}]"
        if src.key in manifest.index:
            manifest.loc[src.key, "daily_updated"] = str(src.daily)
        if src.daily and not (refresh_current or force) and src.key in manifest.index:
            prev = manifest.loc[src.key]
            path = DATA_DIR / prev["local_path"]
            if path.exists() and sha256_of(path) == prev["sha256"]:
                log(f"{tag} pinned   {src.key} (daily-updated; downloaded {prev['downloaded_at_utc']}, "
                    f"sha256 {prev['sha256'][:12]}; --refresh-current to update)")
                continue
        res = direct_resource(src) if src.url else find_resource(src)
        url = res["url"]
        last_mod = res.get("last_modified") or res.get("metadata_modified") or ""
        is_arcgis = (res.get("format") or "").strip() == "Esri REST"
        name = "layer.geojson" if is_arcgis else Path(url.split("?")[0]).name
        dest = RAW_DIR / src.key / name
        rel = dest.relative_to(DATA_DIR).as_posix()

        if not force and src.key in manifest.index:
            prev = manifest.loc[src.key]
            if (prev["url"] == url and prev["catalog_last_modified"] == last_mod
                    and dest.exists() and sha256_of(dest) == prev["sha256"]):
                # ArcGIS layers carry no last_modified; they are refreshed only with --force.
                log(f"{tag} skip     {src.key} (unchanged)")
                continue

        log(f"{tag} download {src.key} <- {url}")
        t0 = time.monotonic()
        try:
            if is_arcgis:
                download_arcgis_layer(url, dest)
            else:
                download(url, dest)
        except Exception as e:
            log(f"{tag} FAILED   {src.key}: {type(e).__name__}: {e}")
            raise
        row = {
            "key": src.key,
            "package": src.package,
            "resource_name": src.resource_name,
            "resource_id": res["id"],
            "url": url,
            "local_path": rel,
            "bytes": str(dest.stat().st_size),
            "sha256": sha256_of(dest),
            "catalog_last_modified": last_mod,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "daily_updated": str(src.daily),
        }
        manifest.loc[src.key] = row
        save_manifest(manifest)
        secs = time.monotonic() - t0
        mb = int(row["bytes"]) / 1e6
        log(f"{tag} done     {src.key}: {mb:,.1f} MB in {secs:.0f}s ({mb / max(secs, 0.01):.1f} MB/s)")

    return save_manifest(manifest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="download only the sample subset")
    ap.add_argument("--force", action="store_true", help="re-download even if unchanged")
    ap.add_argument("--refresh-current", action="store_true",
                    help="re-download pinned daily-updated sources (current MPROP, parcels, permits, codes)")
    ap.add_argument("--only", nargs="+", metavar="KEY", help="download only these source keys")
    args = ap.parse_args()
    sources = [s for s in SOURCES if s.sample] if args.sample else SOURCES
    if args.only:
        unknown = set(args.only) - {s.key for s in SOURCES}
        if unknown:
            raise KeyError(f"unknown source keys: {sorted(unknown)}")
        sources = [s for s in SOURCES if s.key in args.only]
    m = acquire(sources, force=args.force, refresh_current=args.refresh_current)
    log(f"manifest: {MANIFEST_PATH} ({len(m)} rows)")


if __name__ == "__main__":
    main()
