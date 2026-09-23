# Week 4 — Fourier methods for incompressible flow

This directory implements the Week 4 learning sheet: a shared ODE integrator
interface, a periodic advection–diffusion test problem, and a two-dimensional
pseudospectral vorticity solver. The Rust field/solver command contracts are
specified in `field.design.toml` and `fluid.design.toml`; both files were
compared block by block with the course's published reference files.

## Environment and build

From a clean clone, create the course environment and build the Rust programs:

```bash
conda env create -f week4/environment.yml
conda activate amat5315
cd week4
cargo test
cargo build --release --bins
```

The environment supplies Python 3.11, NumPy, Matplotlib, and Rust/Cargo. Cargo
resolves the Rust dependencies in `Cargo.toml` and records exact versions in
`Cargo.lock`. If the `amat5315` environment already exists, activate it and run
the two Cargo commands instead of creating it again.

## Solver design

The crate has one `Integrator` trait, implemented by forward Euler, explicit
midpoint (`rk2`), classical RK4, and an equal-stage-weight RK4 control. The
line problem advances a periodic Gaussian under advection and diffusion, using
either Fourier multipliers or centered finite differences.

The fluid solver evolves vorticity on `[0, 2π)²`. Its forward 2-D FFT is
unnormalized and its inverse is normalized by `n²`; the odd Fourier derivative
of the Nyquist mode is zero. It solves `ψ̂ = ω̂/|k|²`, recovers
`u = ∂y ψ`, `v = −∂x ψ`, evaluates the nonlinear advection pseudospectrally,
and applies the two-thirds mask to the state and nonlinear products. Viscosity
acts as `−ν|k|² ω̂`. The Taylor–Green field is an exact check; the random field
uses equal-amplitude vorticity modes on the inclusive radial band and scales
them to `E(0)=0.5`. Phase generation is seeded per mode, so it is reproducible
and independent of grid-size iteration order.

`fluid` reads the field JSON from stdin. It saves step zero and then every
`round(every/dt)` steps, never shortens a step to hit a snapshot, and stops on
the first non-finite energy. Each run directory contains its `run.json`,
six-decimal `fields.jsonl`, and the tab-separated diagnostics captured by the
analysis scripts.

## Reproduce the checks and evidence

Run these from `week4/`, after the release build:

```bash
python scripts/derivative_checks.py
python scripts/line_experiments.py
python scripts/flow_experiments.py
python scripts/order_convergence.py
```

The two primary command pipelines are:

```bash
./target/release/field taylor-green --n 64 --nu 0.1 --t 0 \
  | ./target/release/fluid --method rk4 --nu 0.1 --dt 0.01 --t-end 1 --every 0.1 --out artifacts/taylor-green-example

./target/release/field random --n 128 --seed 2026 --k-min 2 --k-max 6 \
  | ./target/release/fluid --method rk4 --nu 0.004 --dt 0.01 --t-end 10 --every 0.1 --out artifacts/random-example
```

The scripts write the committed evidence as follows:

| Command | Output |
| --- | --- |
| `python scripts/line_experiments.py` | `evidence/line-stability.png`, `evidence/line-accuracy.png` |
| `python scripts/flow_experiments.py` | `evidence/taylor-green.png`, `evidence/random.png`, `evidence/blowup.png`, `evidence/sensitivity.png` |
| `python scripts/order_convergence.py` | `evidence/order.png`, `evidence/convergence.png`, `evidence/convergence.json` |
| `python scripts/derivative_checks.py` | Console checks and `artifacts/derivatives.json` |

The line script measures the stability maps by advancing the library's actual
integrators on `y′=zy`. It also makes both pulse plots and prints each maximum
error. The flow script runs the exact Taylor–Green check, the random-field
decay, the two stability scans, and paired perturbed/unperturbed runs. Its
random-flow scan brackets the stability limit for this deterministic phase
realization rather than assuming the example's particular random phases. The
final script runs the Taylor–Green order series and random-flow self-convergence
against `dt=0.0025`, then uses the measured fourth-order Richardson estimate to
select the largest tested step below the `5×10⁻⁶` tolerance. All raw run data
and intermediate plots are under the Git-ignored `artifacts/` directory.

## Numerical checks from this run

- The Taylor–Green run at `N=64`, `ν=0.1`, `dt=0.01`, `t=1` gives `E=0.167580`,
  `Z=0.335160`, and relative velocity error `7.04×10⁻⁷`.
- The line temporal-error slopes are `1.033` (Euler), `2.006` (midpoint),
  `4.004` (RK4), and `2.003` (equal-weight RK4). The RK4 Fourier pulse's maximum
  error after one lap is `1.80×10⁻⁵`; the measured RK4 line-stability limit is
  `dt≈0.0494`. Centered differences show the expected phase lag and spatial
  error.
- The random run decays from `E=0.500000`, `Z=6.634685` to
  `E=0.293908`, `Z=0.962827` at `t=10`. Its initial maximum speed is `2.408991`,
  giving the advective estimate `dt≤0.01978`; the measured RK4 bracket is
  `0.030` stable versus `0.035` unstable for this phase realization.
- For Taylor–Green, `dt=0.032` remains stable through the requested interval,
  while `dt=0.033` becomes non-finite at `t=7.821`, bracketing the predicted
  diffusive limit `0.0316`.
- The measured fluid order slopes are `4.104` (Taylor–Green) and `4.039`
  (random self-convergence). Richardson predicts relative error `2.98×10⁻⁶`
  for the selected random-flow step `dt=0.01`; the fine-reference error is
  `2.89×10⁻⁶`.

The random initial condition is deterministic but uses its own keyed phase
draws, so its exact enstrophy history and measured stability bracket need not
match a different valid seeded phase generator. These are explicit, finite-step
simulations: the stability boundary is specific to the retained spectrum and
flow, and the convergence error is temporal self-convergence on the same grid,
not an estimate of spatial discretization error.
