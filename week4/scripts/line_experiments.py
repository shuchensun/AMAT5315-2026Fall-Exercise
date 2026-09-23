#!/usr/bin/env python3
"""Reproduce the Week 4 line-equation stability and accuracy figures."""
from __future__ import annotations

import json
import io
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
ROOT = WEEK.parent
BIN = WEEK / "target" / "release" / "line"
ART = WEEK / "artifacts" / "line"
EVIDENCE = WEEK / "evidence"
ART.mkdir(parents=True, exist_ok=True)
EVIDENCE.mkdir(parents=True, exist_ok=True)


def run_json(*args: str) -> dict:
    return json.loads(subprocess.check_output([str(BIN), *args], text=True))


def slope(xs: list[float], ys: list[float]) -> float:
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def make_stability() -> None:
    limit = json.loads(subprocess.check_output(
        [str(BIN), "stability-limit", "--method", "rk4", "--n", "64", "--nu", "0.05", "--c", "1"],
        text=True))
    (ART / "stability-limit.json").write_text(json.dumps(limit, indent=2) + "\n")
    print(f"measured RK4 line stability limit: dt={limit['critical_dt']:.7f}")
    nx, ny = 281, 321
    grid_text = subprocess.check_output(
        [str(BIN), "growth-grid", "--method", "rk4", "--xmin", "-4.5", "--xmax", "2",
         "--ymin", "-4", "--ymax", "4", "--nx", str(nx), "--ny", str(ny)], text=True)
    grid = np.loadtxt(io.StringIO(grid_text), delimiter=",")
    re = grid[:, 0].reshape(ny, nx)
    im = grid[:, 1].reshape(ny, nx)
    measured = grid[:, 2].reshape(ny, nx)
    maps = {"rk4": measured}
    for method in ("euler", "rk2"):
        text = subprocess.check_output([str(BIN), "growth-grid", "--method", method, "--xmin", "-4.5", "--xmax", "2",
                                        "--ymin", "-4", "--ymax", "4", "--nx", str(nx), "--ny", str(ny)], text=True)
        m = np.loadtxt(io.StringIO(text), delimiter=",")
        maps[method] = m[:, 2].reshape(ny, nx)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    ax = axes[0]
    pcm = ax.pcolormesh(re, im, np.maximum(measured, 1e-5), shading="auto",
                        norm=LogNorm(vmin=0.05, vmax=max(2.0, float(np.nanpercentile(measured, 99.8)))), cmap="magma")
    fig.colorbar(pcm, ax=ax, label="measured RK4 growth |R(z)|")
    for method, style in (("euler", "--"), ("rk2", ":"), ("rk4", "-")):
        ax.contour(re, im, maps[method], levels=[1.0], colors="black", linewidths=1.1,
                   linestyles=style)
    nu, c, n = 0.05, 1.0, 64
    ks = np.arange(-n // 2, n // 2)
    lam = -nu * ks.astype(float) ** 2 - 1j * c * ks
    lam[ks == -n // 2] = -nu * (n // 2) ** 2
    for dt, marker, color in ((0.045, "o", "cyan"), (0.056, "x", "lime")):
        z = lam * dt
        ax.scatter(z.real, z.imag, s=15, marker=marker, color=color, label=f"line modes, dt={dt:g}")
    ax.set(xlim=(-4.5, 2), ylim=(-4, 4), xlabel="Re(z)", ylabel="Im(z)", title="One-step growth measured by the library")
    ax.legend(fontsize=8, loc="upper right")
    for ax, dt in zip(axes[1:], (0.045, 0.056)):
        record = run_json("pulse", "--method", "rk4", "--n", "64", "--nu", "0.05", "--c", "1",
                          "--dt", str(dt), "--t-end", "6", "--sigma", "0.35", "--history")
        path = ART / f"pulse-rk4-dt{dt}.json"
        path.write_text(json.dumps(record))
        values = np.array([[np.nan if v is None else v for v in frame["u"]] for frame in record["history"]])
        times = np.array([frame["t"] for frame in record["history"]])
        ax.imshow(values, origin="upper", aspect="auto", extent=[0, 2*np.pi, times[-1], 0],
                  cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set(xlabel="x", title=f"RK4 pulse, dt={dt:g}")
        ax.set_ylabel("time (downward)")
    fig.suptitle("Measured stability boundary and a periodic pulse below/above the limit")
    fig.tight_layout()
    fig.savefig(EVIDENCE / "line-stability.png", dpi=180)
    plt.close(fig)


def make_accuracy() -> None:
    exact = None
    profiles = []
    for method, dt, derivative, label in (
        ("rk4", "0.02", "fourier", "RK4, Fourier"),
        ("rk4", "0.02", "fd", "RK4, centered differences"),
        ("euler", "0.005", "fourier", "Euler, Fourier"),
    ):
        record = run_json("pulse", "--method", method, "--n", "64", "--nu", "0.002", "--c", "1",
                          "--dt", dt, "--t-end", str(2*np.pi), "--sigma", "0.25", "--derivative", derivative)
        profiles.append((label, record))
        exact = np.asarray(record["exact"])
        print(f"{label}: max error = {record['max_error']:.8e}")

    steps = [0.02, 0.01, 0.005, 0.0025]
    series: dict[str, dict[str, list[float] | float]] = {}
    for method, label in (("euler", "Euler"), ("rk2", "midpoint"), ("rk4", "RK4"), ("equal-rk4", "equal-weight RK4")):
        errors = []
        for dt in steps:
            record = run_json("pulse", "--method", method, "--n", "64", "--nu", "0.05", "--c", "1",
                              "--dt", str(dt), "--t-end", "1", "--sigma", "0.35")
            errors.append(record["max_error"])
        fit = slope(steps, errors)
        print(f"{label}: fitted temporal order = {fit:.4f}")
        series[label] = {"dt": steps, "max_error": errors, "slope": fit}

    (ART / "accuracy.json").write_text(json.dumps({"profiles": {name: rec["max_error"] for name, rec in profiles}, "orders": series}, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    ax.plot(profiles[0][1]["x"], exact, "k--", lw=2, label="exact")
    for label, rec in profiles:
        ax.plot(rec["x"], rec["final"], label=f"{label} (err={rec['max_error']:.1e})")
    ax.set(xlabel="x", ylabel="u(x, 2π)", title="One lap of a periodic Gaussian")
    ax.legend(fontsize=8)
    ax = axes[1]
    for label, values in series.items():
        xs, ys, fit = values["dt"], values["max_error"], values["slope"]
        ax.loglog(xs, ys, "o-", label=f"{label}, slope={fit:.2f}")
    ax.set(xlabel="time step Δt", ylabel="maximum error at t=1", title="Temporal order")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(EVIDENCE / "line-accuracy.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    make_stability()
    make_accuracy()
