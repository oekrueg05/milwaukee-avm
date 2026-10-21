"""One command to rebuild everything from raw downloads (rule 8).

    python -m avm.pipeline            # full run
    python -m avm.pipeline --sample   # sample subsets where a stage has one

Stages are added here as they are built. Each step is idempotent: downloads
skip unchanged files and MPROP extraction is cached by raw-file sha256.
"""

import argparse
import sys

from avm import (acquire, clean_sales, profile_data, stage1_checks, stage3_assessor, stage4_hedonic, stage5_gbm,
                 stage6_geo, freeze, stage7_explain, prospective,
                 stage8_equity, stage9_divergence, stage9_page, writeup)
from avm.progress import log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--refresh-current", action="store_true", help="refresh pinned daily-updated files")
    a = ap.parse_args()
    sys.argv = sys.argv[:1]  # stage mains parse their own (empty) arguments
    log("pipeline: stage 1 acquire")
    acquire.acquire([s for s in acquire.SOURCES if s.sample] if a.sample else acquire.SOURCES,
                    refresh_current=a.refresh_current)
    if not a.sample:
        log("pipeline: stage 1 data dictionary")
        profile_data.main()
        log("pipeline: stage 1 checks")
        stage1_checks.main()
    log("pipeline: stage 2 clean sales")
    if a.sample:
        clean_sales.build(clean_sales.SAMPLE_YEARS, clean_sales.REPORTS_DIR / "stage2" / "sample",
                          clean_sales.INTERIM_DIR / "sales_clean_sample.parquet")
    else:
        clean_sales.build(clean_sales.TRAIN_YEARS + [clean_sales.TEST_YEAR],
                          clean_sales.REPORTS_DIR / "stage2", clean_sales.INTERIM_DIR / "sales_clean.parquet")
        log("pipeline: stage 3 assessor ratio study")
        stage3_assessor.main()
        log("pipeline: stage 4 hedonic baseline")
        stage4_hedonic.main()
        log("pipeline: stage 5 LightGBM")
        stage5_gbm.main()
        log("pipeline: stage 6 geospatial features, calibration, intervals")
        stage6_geo.main()
        log("pipeline: freeze check (re-tune on CV 2021-2024 only; must reproduce the committed spec)")
        freeze.main(verify=True)
        log("pipeline: stage 7 explainability")
        stage7_explain.main()
        log("pipeline: stage 8 equity audit (needs CENSUS_API_KEY in .env)")
        stage8_equity.main()
        log("pipeline: stage 9 divergence, tax dollars, duplex yields")
        stage9_divergence.main()
        log("pipeline: stage 9 pages (private review and portfolio)")
        stage9_page.main()
        log("pipeline: stage 10 write-up (README.md, docs/methods.md)")
        writeup.main()
        # The prospective 2026 predictions are NOT regenerated here: they are hashed and tagged.
        log("pipeline: prospective predictions check")
        prospective.check()
    log("pipeline: done")


if __name__ == "__main__":
    main()
