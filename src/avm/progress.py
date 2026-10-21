"""Progress lines and hang detection for long pipeline steps.

    with step("join_key_check", timeout_s=600):
        ...

prints a start line and a done line with elapsed seconds (flushed, so they
show up in redirected logs). If the block runs past timeout_s, faulthandler
dumps every thread's stack to stderr and exits the process with code 1: a
hang becomes a loud failure that says where it was stuck.
"""

import faulthandler
import sys
import time
from contextlib import contextmanager
from datetime import datetime


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


@contextmanager
def step(name: str, timeout_s: float):
    log(f"start {name} (timeout {timeout_s:.0f}s)")
    t0 = time.monotonic()
    faulthandler.dump_traceback_later(timeout_s, exit=True, file=sys.stderr)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
    log(f"done  {name} in {time.monotonic() - t0:.1f}s")
