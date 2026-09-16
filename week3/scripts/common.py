"""Shared analysis helpers for the Week 3 Ising evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
EVIDENCE = ROOT / "evidence"
TC_EXACT = 2.0 / np.log(1.0 + np.sqrt(2.0))


@dataclass
class Run:
    path: Path
    meta: dict
    rows: list[dict]

    @property
    def l(self) -> int:
        return int(self.meta["L"])

    @property
    def update(self) -> str:
        return str(self.meta["update"])

    def at(self, temperature: float) -> np.ndarray:
        key = round(float(temperature), 8)
        values = [row for row in self.rows if round(float(row["T"]), 8) == key]
        if not values:
            raise KeyError(f"no rows at L={self.l}, T={temperature} in {self.path}")
        return np.asarray(values, dtype=object)

    def temperatures(self) -> np.ndarray:
        return np.asarray(self.meta["t_grid"], dtype=float)


def load_run(path: Path) -> Run:
    with (path / "run.json").open() as handle:
        meta = json.load(handle)
    with (path / "series.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return Run(path, meta, rows)


def load_runs(update: str | None = None) -> list[Run]:
    if not ARTIFACTS.exists():
        raise FileNotFoundError(f"missing {ARTIFACTS}; run the sampler first")
    runs = []
    for path in sorted(ARTIFACTS.iterdir()):
        if path.is_dir() and (path / "run.json").exists() and (path / "series.jsonl").exists():
            run = load_run(path)
            if update is None or run.update == update:
                runs.append(run)
    return runs


def run_for(runs: Iterable[Run], l: int, update: str | None = None, contains: str | None = None) -> Run:
    candidates = [
        run
        for run in runs
        if run.l == l
        and (update is None or run.update == update)
        and (contains is None or contains in run.path.name)
    ]
    if len(candidates) != 1:
        names = ", ".join(str(run.path) for run in candidates)
        raise ValueError(f"expected one run for L={l}, got {len(candidates)}: {names}")
    return candidates[0]


def merge_runs(runs: Iterable[Run]) -> Run:
    """Merge runs for one lattice/update, preferring later rows at duplicate T."""
    runs = list(runs)
    if not runs:
        raise ValueError("cannot merge an empty run list")
    by_temperature: dict[float, list[dict]] = {}
    for run in runs:
        for temperature in run.temperatures():
            by_temperature[round(float(temperature), 8)] = list(run.at(float(temperature)))
    grid = sorted(by_temperature)
    rows = [row for temperature in grid for row in by_temperature[temperature]]
    meta = dict(runs[-1].meta)
    meta["t_grid"] = grid
    return Run(runs[-1].path, meta, rows)


def column(rows: np.ndarray, name: str) -> np.ndarray:
    return np.asarray([row[name] for row in rows], dtype=float)


def temperature_observables(run: Run) -> dict[float, dict[str, float]]:
    out: dict[float, dict[str, float]] = {}
    for temperature in run.temperatures():
        rows = run.at(float(temperature))
        m = column(rows, "M")
        out[round(float(temperature), 8)] = {
            "T": float(temperature),
            "mean_abs_m": float(np.mean(np.abs(m))),
            "mean_m2": float(np.mean(m * m)),
            "chi": float(run.l * run.l * (np.mean(m * m) - np.mean(np.abs(m)) ** 2) / temperature),
            "n": float(len(m)),
        }
    return out


def block_means(values: np.ndarray, block_length: int) -> np.ndarray:
    nblocks = len(values) // block_length
    if nblocks < 2:
        return np.asarray([], dtype=float)
    trimmed = values[: nblocks * block_length]
    return trimmed.reshape(nblocks, block_length).mean(axis=1)


def block_standard_error(values: np.ndarray, block_length: int) -> float:
    means = block_means(values, block_length)
    if len(means) < 2:
        return float("nan")
    return float(np.std(means, ddof=1) / np.sqrt(len(means)))


def block_count_standard_error(values: np.ndarray, count: int) -> float:
    if count <= 1:
        return float("nan")
    length = len(values) // count
    if length < 1:
        return float("nan")
    return block_standard_error(values, length)


def autocorrelation(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.ones(len(values))
    centered = values - np.mean(values)
    variance = np.dot(centered, centered)
    if variance == 0.0:
        return np.ones(len(values))
    nfft = 1 << (2 * len(values) - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=nfft)
    acf = np.fft.irfft(spectrum * np.conjugate(spectrum), n=nfft)[: len(values)]
    acf /= np.arange(len(values), 0, -1)
    acf /= acf[0]
    return acf


def integrated_autocorrelation_time(values: np.ndarray) -> tuple[float, np.ndarray]:
    acf = autocorrelation(values)
    tau = 0.5
    for lag in range(1, len(acf)):
        if lag > 6.0 * tau:
            break
        tau += float(acf[lag])
        if tau <= 0.0:
            return 0.5, acf
    return float(max(tau, 0.5)), acf


def five_point_peak(temperatures: np.ndarray, values: np.ndarray, center: int | None = None) -> tuple[float, float, np.ndarray]:
    temperatures = np.asarray(temperatures, dtype=float)
    values = np.asarray(values, dtype=float)
    if center is None:
        center = int(np.nanargmax(values))
    if center < 2 or center + 2 >= len(values):
        raise ValueError("the largest point must have two neighbours on each side")
    x = temperatures[center - 2 : center + 3]
    y = values[center - 2 : center + 3]
    coefficients = np.polyfit(x, y, 2)
    a, b, c = coefficients
    if a >= 0:
        raise ValueError("quadratic fit is not concave")
    peak_t = -b / (2.0 * a)
    peak_value = np.polyval(coefficients, peak_t)
    return float(peak_t), float(peak_value), coefficients


def bootstrap_blocks(values: np.ndarray, block_length: int, rng: np.random.Generator) -> np.ndarray:
    blocks = block_means_as_rows(values, block_length)
    if len(blocks) == 0:
        raise ValueError(f"not enough data for block length {block_length}")
    choices = rng.integers(0, len(blocks), size=len(blocks))
    return blocks[choices].reshape(-1)


def block_means_as_rows(values: np.ndarray, block_length: int) -> np.ndarray:
    nblocks = len(values) // block_length
    return values[: nblocks * block_length].reshape(nblocks, block_length)


def ensure_evidence() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)


def onsager_magnetization(temperatures: np.ndarray) -> np.ndarray:
    temperatures = np.asarray(temperatures, dtype=float)
    result = np.zeros_like(temperatures)
    mask = temperatures < TC_EXACT
    result[mask] = (1.0 - np.sinh(2.0 / temperatures[mask]) ** -4) ** 0.125
    return result
