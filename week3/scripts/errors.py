#!/usr/bin/env python3
"""Estimate naive, binned, and autocorrelation-aware errors."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EVIDENCE, block_count_standard_error, column, ensure_evidence, integrated_autocorrelation_time, load_runs, merge_runs, run_for


def main() -> None:
    runs = load_runs("metropolis")
    ensure_evidence()
    with (EVIDENCE / "errors.txt").open("w") as handle:
        handle.write("L\tT\tmean_abs_M\tnaive_se\tblock50_se\tratio\ttau_int\tn_effective\n")
        for lattice_size in (32, 64):
            run = merge_runs([
                run_for(runs, lattice_size, "metropolis", contains="coarse"),
                run_for(runs, lattice_size, "metropolis", contains="window"),
            ])
            for temperature in run.temperatures():
                values = np.abs(column(run.at(float(temperature)), "M"))
                naive = float(np.std(values, ddof=1) / np.sqrt(len(values)))
                block = block_count_standard_error(values, 50)
                tau, _ = integrated_autocorrelation_time(values)
                ratio = block / naive if naive > 0 else float("nan")
                n_eff = len(values) / (2.0 * tau)
                handle.write(f"{lattice_size}\t{temperature:.8f}\t{np.mean(values):.8f}\t{naive:.8g}\t{block:.8g}\t{ratio:.8g}\t{tau:.8g}\t{n_eff:.8g}\n")
                print(f"L={lattice_size} T={temperature:.2f} mean={np.mean(values):.5f} naive={naive:.4g} block50={block:.4g} ratio={ratio:.3g} tau={tau:.3g}")


if __name__ == "__main__":
    main()
