#!/usr/bin/env python3
"""Compare Metropolis and Wolff autocorrelation in spin-update work."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EVIDENCE, TC_EXACT, ensure_evidence, five_point_peak, integrated_autocorrelation_time, load_runs, run_for


def block_bootstrap_se(values: np.ndarray, block_length: int, seed: int = 7, repetitions: int = 200) -> float:
    nblocks = len(values) // block_length
    blocks = values[: nblocks * block_length].reshape(nblocks, block_length).mean(axis=1)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, nblocks, size=(repetitions, nblocks))
    estimates = blocks[indexes].mean(axis=1)
    return float(np.std(estimates, ddof=1))


def magnetization_compare(runs) -> None:
    metropolis = run_for(runs, 64, "metropolis", contains="window")
    wolff = run_for(runs, 64, "wolff")
    temperatures = metropolis.temperatures()
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    for run, label, color in ((metropolis, "Metropolis", "tab:blue"), (wolff, "Wolff", "tab:orange")):
        means = []
        errors = []
        chis = []
        for temperature in temperatures:
            rows = run.at(float(temperature))
            m = np.asarray([row["M"] for row in rows], dtype=float)
            abs_m = np.abs(m)
            means.append(np.mean(abs_m))
            errors.append(block_bootstrap_se(abs_m, 2000, seed=int(temperature * 1000) + run.l))
            chis.append(run.l * run.l * (np.mean(m * m) - np.mean(abs_m) ** 2) / temperature)
        left.errorbar(temperatures, means, yerr=errors, fmt="o-", ms=3, capsize=2, label=label, color=color)
        right.plot(temperatures, chis, "o-", ms=3, label=label, color=color)
    wolff_obs = {round(float(t), 8): None for t in temperatures}
    wolff_chi = []
    for temperature in temperatures:
        rows = wolff.at(float(temperature))
        m = np.asarray([row["M"] for row in rows], dtype=float)
        wolff_chi.append(wolff.l * wolff.l * (np.mean(m * m) - np.mean(np.abs(m)) ** 2) / temperature)
    peak_t, _, _ = five_point_peak(temperatures, np.asarray(wolff_chi))
    right.axvline(peak_t, color="tab:orange", ls=":", label=f"Wolff fit {peak_t:.4f}")
    left.axvline(TC_EXACT, color="black", ls="--")
    right.axvline(TC_EXACT, color="black", ls="--", label=f"Tc={TC_EXACT:.4f}")
    left.set(xlabel="temperature T", ylabel="mean |m|")
    right.set(xlabel="temperature T", ylabel="cluster susceptibility chi")
    left.legend(fontsize=8)
    right.legend(fontsize=8)
    ensure_evidence()
    fig.tight_layout()
    fig.savefig(EVIDENCE / "magnetization-compare.png", dpi=160)
    plt.close(fig)


def write_summary(runs) -> None:
    metropolis = run_for(runs, 64, "metropolis", contains="window")
    wolff = run_for(runs, 64, "wolff")
    temperature = 2.3
    metro_m = np.asarray([row["M"] for row in metropolis.at(temperature)], dtype=float)
    wolff_m = np.asarray([row["M"] for row in wolff.at(temperature)], dtype=float)
    metro_abs = np.abs(metro_m)
    wolff_abs = np.abs(wolff_m)
    metro_se = block_bootstrap_se(metro_abs, 2000, seed=2300 + 64)
    wolff_se = block_bootstrap_se(wolff_abs, 2000, seed=2300 + 64)
    d = abs(np.mean(metro_abs) - np.mean(wolff_abs)) / np.sqrt(metro_se**2 + wolff_se**2)
    metro_tau, _ = integrated_autocorrelation_time(metro_abs)
    wolff_tau, _ = integrated_autocorrelation_time(wolff_abs)
    mean_cluster = float(np.mean([row["cluster_size"] for row in wolff.at(temperature)]))
    wolff_work = wolff_tau * mean_cluster / (wolff.l * wolff.l)
    ratio = metro_tau / wolff_work
    grid = wolff.temperatures()
    peaks = {}
    for lattice_size in (32, 64):
        run = run_for(runs, lattice_size, "wolff")
        chis = []
        for t in run.temperatures():
            rows = run.at(float(t))
            m = np.asarray([row["M"] for row in rows], dtype=float)
            chis.append(run.l * run.l * (np.mean(m * m) - np.mean(np.abs(m)) ** 2) / t)
        peaks[lattice_size], _, _ = five_point_peak(run.temperatures(), np.asarray(chis))
    tc = 2 * peaks[64] - peaks[32]
    ensure_evidence()
    with (EVIDENCE / "compare.txt").open("w") as handle:
        handle.write(f"L=64 T=2.3 Metropolis mean_abs_M {np.mean(metro_abs):.8f} block_bootstrap_se {metro_se:.8f}\n")
        handle.write(f"L=64 T=2.3 Wolff mean_abs_M {np.mean(wolff_abs):.8f} block_bootstrap_se {wolff_se:.8f}\n")
        handle.write(f"agreement_d {d:.8f}\n")
        handle.write("agreement_status provisional (block-length stability still requires checking)\n")
        handle.write(f"Metropolis tau_sweeps {metro_tau:.8f}\n")
        handle.write(f"Wolff tau_moves {wolff_tau:.8f}\n")
        handle.write(f"Wolff mean_cluster_size {mean_cluster:.8f}\n")
        handle.write(f"Wolff tau_work {wolff_work:.8f}\n")
        handle.write(f"work_ratio_metropolis_over_wolff {ratio:.8f}\n")
        handle.write(f"Wolff T_peak_L32 {peaks[32]:.8f}\n")
        handle.write(f"Wolff T_peak_L64 {peaks[64]:.8f}\n")
        handle.write(f"Wolff Tc_extrapolated {tc:.8f}\n")


def main() -> None:
    runs = load_runs()
    magnetization_compare(runs)
    write_summary(runs)
    metropolis = run_for(runs, 64, "metropolis", contains="window")
    wolff = run_for(runs, 64, "wolff")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run, label, color, work_factor in ((metropolis, "Metropolis", "tab:blue", None), (wolff, "Wolff", "tab:orange", True)):
        temperatures = run.temperatures()
        values = []
        for temperature in temperatures:
            rows = run.at(float(temperature))
            m = np.asarray([row["M"] for row in rows], dtype=float)
            tau, _ = integrated_autocorrelation_time(np.abs(m))
            if work_factor:
                cluster = np.asarray([row["cluster_size"] for row in rows], dtype=float)
                tau *= float(np.mean(cluster)) / (run.l * run.l)
            values.append(tau)
        ax.plot(temperatures, values, "o-", label=label, color=color)
    ax.set_yscale("log")
    ax.set(xlabel="temperature T", ylabel="tau in spin-update work", title="Work-normalized autocorrelation, L=64")
    ax.legend()
    ensure_evidence()
    fig.tight_layout()
    fig.savefig(EVIDENCE / "tau-compare.png", dpi=160)
    plt.close(fig)
    print(f"wrote {EVIDENCE / 'magnetization-compare.png'} and {EVIDENCE / 'tau-compare.png'}")


if __name__ == "__main__":
    main()
