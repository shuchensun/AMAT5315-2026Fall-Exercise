#!/usr/bin/env python3
"""Generate the figures required by the Week 3 evidence checklist."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    EVIDENCE,
    TC_EXACT,
    autocorrelation,
    block_means,
    block_standard_error,
    column,
    ensure_evidence,
    five_point_peak,
    integrated_autocorrelation_time,
    load_run,
    ROOT,
    load_runs,
    onsager_magnetization,
    run_for,
    temperature_observables,
)


def save(fig: plt.Figure, name: str) -> None:
    ensure_evidence()
    fig.tight_layout()
    fig.savefig(EVIDENCE / name, dpi=160)
    plt.close(fig)


def observables(run):
    data = temperature_observables(run)
    keys = sorted(data)
    return (
        np.array([data[key]["T"] for key in keys]),
        np.array([data[key]["mean_abs_m"] for key in keys]),
        np.array([data[key]["chi"] for key in keys]),
    )


def plot_magnetization(runs) -> None:
    run = run_for(runs, 64, "metropolis", contains="coarse")
    window = run_for(runs, 64, "metropolis", contains="window")
    from common import merge_runs
    run = merge_runs([run, window])
    temperatures, means, _ = observables(run)
    dense = np.linspace(float(temperatures.min()), float(temperatures.max()), 500)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(dense, onsager_magnetization(dense), label="Onsager, infinite lattice", color="black")
    ax.plot(temperatures, means, "o-", label="measured, L=64")
    ax.axvline(TC_EXACT, color="tab:red", ls="--", label=f"Tc = {TC_EXACT:.4f}")
    ax.set(xlabel="temperature T", ylabel="mean |m|", ylim=(0, 1.05))
    ax.legend()
    save(fig, "magnetization.png")


def plot_susceptibility(runs) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lattice_size, color in ((32, "tab:blue"), (64, "tab:orange")):
        from common import merge_runs
        run = merge_runs([run_for(runs, lattice_size, "metropolis", contains="coarse"), run_for(runs, lattice_size, "metropolis", contains="window")])
        temperatures, _, chi = observables(run)
        peak_t, _, _ = five_point_peak(temperatures, chi)
        ax.plot(temperatures, chi, "o-", color=color, label=f"L={lattice_size}")
        ax.axvline(peak_t, color=color, ls=":", label=f"L={lattice_size} fit {peak_t:.4f}")
    ax.axvline(TC_EXACT, color="black", ls="--", label=f"Tc = {TC_EXACT:.4f}")
    ax.set(xlabel="temperature T", ylabel="susceptibility chi")
    ax.legend(fontsize=8)
    save(fig, "susceptibility.png")


def plot_trace(runs) -> None:
    from common import merge_runs
    run = merge_runs([run_for(runs, 64, "metropolis", contains="coarse"), run_for(runs, 64, "metropolis", contains="window")])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for temperature, color in ((2.3, "tab:red"), (3.0, "tab:blue")):
        values = np.abs(column(run.at(temperature), "M"))[:2000]
        ax.plot(np.arange(1, len(values) + 1), values, lw=0.7, color=color, label=f"T={temperature}")
    ax.set(xlabel="measurement sweep", ylabel="|m|")
    ax.legend()
    save(fig, "trace.png")


def plot_acf_binning(runs) -> None:
    from common import merge_runs
    run = merge_runs([run_for(runs, 64, "metropolis", contains="coarse"), run_for(runs, 64, "metropolis", contains="window")])
    values = np.abs(column(run.at(2.3), "M"))
    acf = autocorrelation(values)
    lags = np.arange(min(len(acf), 5001))
    block_lengths = np.array([1, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000, 5000])
    block_errors = np.array([block_standard_error(values, int(length)) for length in block_lengths])
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.plot(lags, acf[: len(lags)])
    left.axhline(0, color="black", lw=0.7)
    left.set(xlabel="lag t (sweeps)", ylabel="autocorrelation of |m|", ylim=(-0.1, 1.05))
    right.plot(block_lengths, block_errors, "o-")
    right.set_xscale("log")
    right.set(xlabel="block length (sweeps)", ylabel="error bar on mean |m|")
    save(fig, "acf-binning.png")


def plot_tau(runs) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lattice_size, color in ((32, "tab:blue"), (64, "tab:orange")):
        from common import merge_runs
        run = merge_runs([run_for(runs, lattice_size, "metropolis", contains="coarse"), run_for(runs, lattice_size, "metropolis", contains="window")])
        taus = []
        temperatures = run.temperatures()
        for temperature in temperatures:
            values = np.abs(column(run.at(float(temperature)), "M"))
            tau, _ = integrated_autocorrelation_time(values)
            taus.append(tau)
        ax.plot(temperatures, taus, "o-", color=color, label=f"L={lattice_size}")
    ax.axvline(TC_EXACT, color="black", ls="--", label=f"Tc = {TC_EXACT:.4f}")
    ax.set_yscale("log")
    ax.set(xlabel="temperature T", ylabel="integrated autocorrelation time (sweeps)")
    ax.legend()
    save(fig, "tau.png")


def block_statistics(values: np.ndarray, block_length: int) -> tuple[np.ndarray, np.ndarray]:
    nblocks = len(values) // block_length
    trimmed = values[: nblocks * block_length]
    blocks = trimmed.reshape(nblocks, block_length)
    return np.abs(blocks).mean(axis=1), (blocks * blocks).mean(axis=1)


def bootstrap_peaks(run, block_length: int, centers: int, repetitions: int, rng: np.random.Generator):
    temperatures = run.temperatures()
    original = temperature_observables(run)
    original_chi = np.array([original[round(float(t), 8)]["chi"] for t in temperatures])
    stats = []
    for temperature in temperatures:
        values = column(run.at(float(temperature)), "M")
        mean_m, mean_m2 = block_statistics(values, block_length)
        stats.append((mean_m, mean_m2))
    coefficients = []
    peaks = []
    failures = 0
    for _ in range(repetitions):
        chi = []
        for (block_m, block_m2), temperature in zip(stats, temperatures):
            indexes = rng.integers(0, len(block_m), size=len(block_m))
            mean_m = np.mean(block_m[indexes])
            mean_m2 = np.mean(block_m2[indexes])
            chi.append(run.l * run.l * (mean_m2 - mean_m * mean_m) / temperature)
        try:
            peak_t, _, coeff = five_point_peak(temperatures, np.asarray(chi), center=centers)
            if not (temperatures[centers - 2] <= peak_t <= temperatures[centers + 2]):
                raise ValueError("peak outside fit window")
        except ValueError:
            failures += 1
            continue
        coefficients.append(coeff)
        peaks.append(peak_t)
    return original_chi, np.asarray(coefficients), np.asarray(peaks), failures


def plot_bootstrap(runs) -> None:
    run32 = run_for(runs, 32, "metropolis", contains="window")
    run64 = run_for(runs, 64, "metropolis", contains="window")
    temperatures = run64.temperatures()
    original32 = temperature_observables(run32)
    original64 = temperature_observables(run64)
    chi32 = np.array([original32[round(float(t), 8)]["chi"] for t in run32.temperatures()])
    chi64 = np.array([original64[round(float(t), 8)]["chi"] for t in run64.temperatures()])
    center32 = int(np.argmax(chi32))
    center64 = int(np.argmax(chi64))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    summary = []
    bootstrap_cache = {}
    for block_length in (2000, 4000, 8000):
        bootstrap_cache[block_length] = (
            bootstrap_peaks(run32, block_length, center32, 500, np.random.default_rng(3200 + block_length)),
            bootstrap_peaks(run64, block_length, center64, 500, np.random.default_rng(6400 + block_length)),
        )
    for ax, run, chi, center, label, seed in (
        (axes[0], run32, chi32, center32, "L=32", 3200),
        (axes[1], run64, chi64, center64, "L=64", 6400),
    ):
        ax.plot(run.temperatures(), chi, "o", color="black", label="measured")
        for block_length, alpha in ((2000, 0.15), (4000, 0.22), (8000, 0.30)):
            result32, result64 = bootstrap_cache[block_length]
            _, coeffs, peaks, failures = result32 if run.l == 32 else result64
            if len(coeffs):
                xfit = np.linspace(run.temperatures()[center - 2], run.temperatures()[center + 2], 150)
                curves = np.asarray([np.polyval(coeff, xfit) for coeff in coeffs])
                ax.fill_between(xfit, np.min(curves, axis=0), np.max(curves, axis=0), alpha=alpha, label=f"blocks {block_length}")
            summary.append((label, block_length, len(peaks), failures, float(np.std(peaks, ddof=1)) if len(peaks) > 1 else float("nan")))
        ax.axvline(TC_EXACT, color="black", ls="--")
        ax.set(xlabel="temperature T", title=label)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("susceptibility chi")
    save(fig, "chi-bootstrap.png")
    with (EVIDENCE / "bootstrap.txt").open("w") as handle:
        handle.write("quantity\tblock_length\tvalid\tfailures\tbootstrap_sd\n")
        for row in summary:
            handle.write("%s\t%d\t%d\t%d\t%.8g\n" % row)
        for block_length in (2000, 4000, 8000):
            p32 = bootstrap_cache[block_length][0][2]
            p64 = bootstrap_cache[block_length][1][2]
            n = min(len(p32), len(p64))
            tc_samples = 2.0 * p64[:n] - p32[:n]
            failures = (500 - len(p32)) + (500 - len(p64))
            handle.write("Tc_extrapolated\t%d\t%d\t%d\t%.8g\n" % (block_length, n, failures, float(np.std(tc_samples, ddof=1))))


def viewer_frames() -> None:
    candidates = sorted((ROOT / "runs" / "ramp").glob("spins.jsonl"))
    if not candidates:
        return
    frame_path = candidates[-1]
    frames = []
    import json

    with frame_path.open() as handle:
        for line in handle:
            if line.strip():
                frames.append(json.loads(line))
    for temperature in (1.8, 2.3, 3.0):
        candidates = [frame for frame in frames if abs(frame["T"] - temperature) < 1e-8]
        if not candidates:
            continue
        frame = candidates[-1]
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(np.asarray(frame["spins"]).reshape(frame["L"], frame["L"]), cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"L=64, T={temperature:.1f}, m={frame['m']:.3f}")
        ax.set_axis_off()
        save(fig, f"viewer-T{temperature:.1f}.png")


def plot_boltzmann() -> None:
    candidates = sorted((ROOT / "runs" / "ramp").glob("series.jsonl"))
    if not candidates:
        return
    run = load_run(candidates[-1].parent)
    by_temperature = {}
    for temperature in (3.0, 3.1):
        rows = run.at(temperature)
        by_temperature[temperature] = np.asarray([row["E"] * run.l * run.l for row in rows], dtype=float)
    low, high = by_temperature[3.0], by_temperature[3.1]
    lo = min(low.min(), high.min())
    hi = max(low.max(), high.max())
    bins = np.linspace(lo, hi, 28)
    cold_counts, edges = np.histogram(low, bins=bins)
    hot_counts, _ = np.histogram(high, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = (cold_counts > 0) & (hot_counts > 0)
    log_ratio = np.log(hot_counts[mask] / cold_counts[mask])
    predicted_slope = 1.0 / 3.0 - 1.0 / 3.1
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(centers[mask], log_ratio, label="measured log histogram ratio")
    if np.count_nonzero(mask) >= 2:
        fit = np.polyfit(centers[mask], log_ratio, 1)
        x = np.linspace(centers[mask].min(), centers[mask].max(), 100)
        ax.plot(x, np.polyval(fit, x), label=f"fit slope={fit[0]:.5f}")
    ax.plot(centers[mask], predicted_slope * centers[mask], "--", label=f"Boltzmann slope={predicted_slope:.5f}")
    ax.set(xlabel="total energy E", ylabel="log P3.1(E) / P3.0(E)")
    ax.legend()
    save(fig, "boltzmann.png")


def main() -> None:
    runs = load_runs()
    metropolis = [run for run in runs if run.update == "metropolis"]
    if len(metropolis) >= 2:
        plot_magnetization(metropolis)
        plot_susceptibility(metropolis)
        plot_trace(metropolis)
        plot_acf_binning(metropolis)
        plot_tau(metropolis)
        plot_bootstrap(metropolis)
    viewer_frames()
    plot_boltzmann()
    print(f"wrote figures to {EVIDENCE}")


if __name__ == "__main__":
    main()
