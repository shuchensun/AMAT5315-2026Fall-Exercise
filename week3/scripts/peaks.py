#!/usr/bin/env python3
"""Locate finite-size susceptibility peaks and extrapolate Tc."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EVIDENCE, TC_EXACT, load_runs, merge_runs, run_for, temperature_observables, five_point_peak, ensure_evidence


def main() -> None:
    runs = load_runs("metropolis")
    rows = []
    peaks = {}
    for lattice_size in (32, 64):
        window = run_for(runs, lattice_size, "metropolis", contains="window")
        merged = merge_runs([run_for(runs, lattice_size, "metropolis", contains="coarse"), window])
        obs = temperature_observables(window)
        temperatures = np.array([obs[key]["T"] for key in sorted(obs)])
        chi = np.array([obs[key]["chi"] for key in sorted(obs)])
        peak_t, peak_chi, _ = five_point_peak(temperatures, chi)
        merged_obs = temperature_observables(merged)
        cold_temperature = min(merged_obs)
        cold = merged_obs[cold_temperature]["mean_abs_m"]
        peaks[lattice_size] = peak_t
        rows.append((lattice_size, cold, peak_t, peak_chi))
    extrapolated = 2.0 * peaks[64] - peaks[32]
    ensure_evidence()
    with (EVIDENCE / "peaks.txt").open("w") as handle:
        handle.write("L\tcold_mean_abs_M\tT_peak\tchi_peak\n")
        for lattice_size, cold, peak_t, peak_chi in rows:
            handle.write(f"{lattice_size}\t{cold:.8f}\t{peak_t:.8f}\t{peak_chi:.8f}\n")
        handle.write(f"Tc_extrapolated\t{extrapolated:.8f}\tdeviation\t{(extrapolated - TC_EXACT) / TC_EXACT:.8%}\n")
    print(f"Onsager Tc = {TC_EXACT:.8f}")
    for lattice_size, cold, peak_t, peak_chi in rows:
        print(f"L={lattice_size}: cold mean |M|={cold:.8f}, T_peak={peak_t:.8f}, chi_peak={peak_chi:.8f}")
    print(f"extrapolated Tc = {extrapolated:.8f} (relative deviation {(extrapolated - TC_EXACT) / TC_EXACT:.3%})")


if __name__ == "__main__":
    main()
