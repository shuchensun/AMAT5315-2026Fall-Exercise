# Week 3 — Monte Carlo simulation

This directory implements the two-dimensional, zero-field Ising model from the
AMAT5315 Week 3 learning sheet. It contains a Rust sampler and Python analysis
scripts. The sampler writes raw JSON artifacts; every plot and reported number
is recomputed from those rows.

## Environment

Use the course environment created for this project:

```bash
conda activate amat5315
cd /Users/charles/Documents/ChatGPT/amat5315/week3
```

The full runs use a release build. First run the unit tests:

```bash
cargo test
```

## Sampler contract

The command is specified in `ising.design.toml`. It supports:

- `metropolis`: one step is `L*L` independent single-spin proposals;
- `wolff`: one step is one cluster flip;
- periodic boundaries, `J=1`, no magnetic field;
- deterministic `ChaCha8Rng` streams from the supplied seed;
- `run.json`, `series.jsonl`, and optional `spins.jsonl` outputs.

## Reproduce Part 2

The four contract runs are:

```bash
cargo build --release
mkdir -p artifacts
./target/release/ising --update metropolis --l 32 --t-from 1.5 --t-to 3.5 --t-step 0.1 --discard 2000 --measure 5000 --seed 1042 --out artifacts/coarse-l32
./target/release/ising --update metropolis --l 64 --t-from 1.5 --t-to 3.5 --t-step 0.1 --discard 2000 --measure 5000 --seed 42 --out artifacts/coarse-l64
./target/release/ising --update metropolis --l 32 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 2000 --measure 100000 --seed 1042 --out artifacts/window-l32
./target/release/ising --update metropolis --l 64 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 2000 --measure 100000 --seed 42 --out artifacts/window-l64
```

Then generate the finite-size plots, peak estimate, and error table:

```bash
python3 scripts/peaks.py
python3 scripts/errors.py
python3 scripts/plots.py
```

This writes `evidence/magnetization.png`, `susceptibility.png`,
`trace.png`, `acf-binning.png`, `tau.png`, `chi-bootstrap.png`, the viewer
snapshots, `peaks.txt`, `errors.txt`, and `bootstrap.txt`.

The critical-temperature estimate is

```text
Tc ≈ 2*T_peak(L=64) - T_peak(L=32)
```

where each peak is a quadratic fit through the five grid points around the
largest susceptibility. The error analysis reports naive standard errors,
50-block errors, integrated autocorrelation times, effective sample counts,
and a block bootstrap at lengths 2000, 4000, and 8000.

## Reproduce Part 4

```bash
./target/release/ising --update wolff --l 64 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 20000 --measure 100000 --seed 42 --out artifacts/wolff-l64
./target/release/ising --update wolff --l 32 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 20000 --measure 100000 --seed 1042 --out artifacts/wolff-l32
python3 scripts/compare.py
```

The comparison uses the same observable for both samplers and converts the
Wolff autocorrelation time into spin-update work:

```text
tau_work = tau_moves * mean(cluster_size) / L^2
```

The raw `runs/` and `artifacts/` directories are intentionally ignored by Git;
the generated evidence figures and summary tables are small enough to track.

## Scientific limitations

The two-size extrapolation removes the leading `1/L` finite-size shift under
its stated scaling assumption; it does not remove higher-order finite-size or
five-point-fit bias. A block-bootstrap error bar measures sampling uncertainty,
not those systematic effects. If the error estimate is still changing with
block length, the README should report the sampling error as unresolved.
