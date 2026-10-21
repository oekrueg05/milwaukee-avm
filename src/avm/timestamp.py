"""OpenTimestamps proof for the prospective 2026 predictions.

Usage:
    python -m avm.timestamp stamp     # write predictions_2026.parquet.ots (submits the SHA-256 digest only)
    python -m avm.timestamp info      # show the proof tree
    python -m avm.timestamp upgrade   # once Bitcoin has confirmed (hours), fetch the full attestation
    python -m avm.timestamp verify    # check each Bitcoin attestation against public block data

Uses the `opentimestamps` library directly, as the `ots` client's stamp and
upgrade commands do; the client itself fails to import on Windows (its Bitcoin
library needs an OpenSSL DLL). Only the file's SHA-256 digest, behind a random
nonce, is sent to the public calendar servers; the file never leaves the
machine. Verify independently with the standard client on any machine:
`ots verify reports/prospective/predictions_2026.parquet.ots`.
"""

import argparse
import os

from opentimestamps.calendar import RemoteCalendar
from opentimestamps.core.notary import BitcoinBlockHeaderAttestation, PendingAttestation
from opentimestamps.core.op import OpAppend, OpSHA256
from opentimestamps.core.serialize import StreamDeserializationContext, StreamSerializationContext
from opentimestamps.core.timestamp import DetachedTimestampFile

from avm.acquire import sha256_of
from avm.prospective import PRED_PATH, check

OTS_PATH = PRED_PATH.with_name(PRED_PATH.name + ".ots")
CALENDARS = ["https://a.pool.opentimestamps.org", "https://b.pool.opentimestamps.org",
             "https://a.pool.eternitywall.com", "https://ots.btc.catallaxy.com"]
MIN_CALENDARS = 2


def stamp() -> None:
    if OTS_PATH.exists():
        raise FileExistsError(f"{OTS_PATH} exists; a file is stamped once")
    check()  # refuse to stamp a modified prediction file
    with open(PRED_PATH, "rb") as fd:
        dtf = DetachedTimestampFile.from_fd(OpSHA256(), fd)
    nonced = dtf.timestamp.ops.add(OpAppend(os.urandom(16)))
    root = nonced.ops.add(OpSHA256())
    ok = []
    for url in CALENDARS:
        try:
            root.merge(RemoteCalendar(url).submit(root.msg, timeout=15))
            ok.append(url)
        except Exception as e:  # noqa: BLE001 - one calendar down is fine, below MIN_CALENDARS is not
            print(f"calendar failed: {url}: {e}")
    if len(ok) < MIN_CALENDARS:
        raise RuntimeError(f"only {len(ok)} calendars accepted the digest; need {MIN_CALENDARS}")
    with open(OTS_PATH, "wb") as out:
        dtf.serialize(StreamSerializationContext(out))
    print(f"stamped sha256 {dtf.file_digest.hex()} via {len(ok)} calendars -> {OTS_PATH}")


def _load() -> DetachedTimestampFile:
    with open(OTS_PATH, "rb") as fd:
        return DetachedTimestampFile.deserialize(StreamDeserializationContext(fd))


def info() -> None:
    dtf = _load()
    print(f"file sha256 {dtf.file_digest.hex()} (current file {sha256_of(PRED_PATH)})")
    print(dtf.timestamp.str_tree())


def _directly_verified(stamp):
    """Sub-timestamps that carry attestations (as in the ots client's upgrade)."""
    if stamp.attestations:
        yield stamp
    else:
        for sub in stamp.ops.values():
            yield from _directly_verified(sub)


def _bitcoin_heights(stamp) -> list[int]:
    return sorted(a.height for _, a in stamp.all_attestations() if isinstance(a, BitcoinBlockHeaderAttestation))


def upgrade() -> None:
    dtf = _load()
    changed = False
    for sub in list(_directly_verified(dtf.timestamp)):
        for att in list(sub.attestations):
            if isinstance(att, PendingAttestation):
                try:
                    upgraded = RemoteCalendar(att.uri).get_timestamp(sub.msg, timeout=15)
                except Exception as e:  # noqa: BLE001 - not yet confirmed is normal
                    print(f"not upgraded yet from {att.uri}: {type(e).__name__}: {e}")
                    continue
                if any(isinstance(a, BitcoinBlockHeaderAttestation) for _, a in upgraded.all_attestations()):
                    sub.merge(upgraded)
                    changed = True
                    print(f"upgraded via {att.uri}")
                else:
                    print(f"pending at {att.uri} (no Bitcoin attestation yet)")
    heights = _bitcoin_heights(dtf.timestamp)
    if changed:
        with open(OTS_PATH, "wb") as out:
            dtf.serialize(StreamSerializationContext(out))
        print(f"upgraded {OTS_PATH}; Bitcoin block heights attested: {heights}")
    else:
        print(f"no upgrade written; Bitcoin block heights attested so far: {heights}")


def verify() -> None:
    """Each Bitcoin attestation's commitment must equal that block's Merkle
    root (internal byte order) from a public block explorer, and the proof's
    digest must equal the file's sha256. Prints each block's time."""
    import json
    import urllib.request
    from datetime import datetime, timezone
    dtf = _load()
    ok = dtf.file_digest.hex() == sha256_of(PRED_PATH)
    print(f"proof digest matches the file's sha256: {ok}")
    get = lambda u: urllib.request.urlopen(u, timeout=20).read().decode()  # noqa: E731
    atts = sorted((a.height, m) for m, a in dtf.timestamp.all_attestations()
                  if isinstance(a, BitcoinBlockHeaderAttestation))
    if not atts:
        raise RuntimeError("no Bitcoin attestation yet; run upgrade after confirmation")
    for height, msg in atts:
        b = json.loads(get(f"{EXPLORER}/block/{get(f'{EXPLORER}/block-height/{height}').strip()}"))
        match = msg == bytes.fromhex(b["merkle_root"])[::-1]
        ok &= match
        when = datetime.fromtimestamp(b["timestamp"], tz=timezone.utc).isoformat()
        print(f"block {height}: commitment matches Merkle root: {match}; block time {when}")
    if not ok:
        raise AssertionError("timestamp proof does not verify")
    print("verified: the file existed by the earliest block time above")


EXPLORER = "https://blockstream.info/api"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["stamp", "info", "upgrade", "verify"])
    {"stamp": stamp, "info": info, "upgrade": upgrade, "verify": verify}[ap.parse_args().command]()


if __name__ == "__main__":
    main()
