#!/usr/bin/env python3
"""Run the Week 4 2-D flow checks and create their evidence figures."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
BIN = WEEK / "target" / "release"
ART = WEEK / "artifacts"
EVIDENCE = WEEK / "evidence"
ART.mkdir(parents=True, exist_ok=True)
EVIDENCE.mkdir(parents=True, exist_ok=True)


def field(case: str, n: int, **kwargs) -> dict:
    command = [str(BIN / "field"), case, "--n", str(n)]
    for key, value in kwargs.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    return json.loads(subprocess.check_output(command, text=True))


def run_flow(initial: dict, name: str, *, method: str, nu: float, dt: float, end: float, every: float) -> tuple[Path, str, int]:
    out = ART / name
    input_json = ART / f"{name}-initial.json"
    input_json.parent.mkdir(parents=True, exist_ok=True)
    input_json.write_text(json.dumps(initial, separators=(",", ":")))
    command = [str(BIN / "fluid"), "--method", method, "--nu", str(nu), "--dt", str(dt),
               "--t-end", str(end), "--every", str(every), "--out", str(out)]
    proc = subprocess.run(command, input=json.dumps(initial), text=True, capture_output=True)
    (out / "diagnostics.tsv").write_text(proc.stdout)
    if proc.stderr:
        (out / "stderr.txt").write_text(proc.stderr)
    return out, proc.stdout, proc.returncode


def frames(path: Path) -> list[dict]:
    return [json.loads(line) for line in (path / "fields.jsonl").read_text().splitlines() if line.strip()]


def diagnostics(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    for line in (path / "diagnostics.tsv").read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 3:
            try:
                rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                rows.append((float(parts[0]), np.nan, np.nan))
    a = np.asarray(rows)
    return a[:, 0], a[:, 1], a[:, 2]


def image(a: list[float], n: int) -> np.ndarray:
    return np.asarray(a, dtype=float).reshape(n, n)


def make_taylor_green() -> None:
    initial = field("taylor-green", 64, nu=0.1, t=0)
    path, _, code = run_flow(initial, "taylor-green", method="rk4", nu=0.1, dt=0.01, end=1, every=0.2)
    assert code == 0, f"Taylor-Green validation run failed: {path}"
    fs = frames(path)
    f0, f1 = fs[0], fs[-1]
    n = 64
    xx, yy = np.meshgrid(np.arange(n) * 2*np.pi/n, np.arange(n) * 2*np.pi/n)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    vmax = 2.0
    for ax, frame in zip(axes, (f0, f1)):
        t = frame["t"]
        om = image(frame["omega"], n)
        im = ax.imshow(om, extent=[0, 2*np.pi, 0, 2*np.pi], origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        stride = 6
        ax.quiver(xx[::stride, ::stride], yy[::stride, ::stride], image(frame["u"],n)[::stride,::stride],
                  image(frame["v"],n)[::stride,::stride], color="black", alpha=0.7, scale=28)
        ax.set(title=f"t={t:g}", xlabel="x", ylabel="y", aspect="equal")
    fig.colorbar(im, ax=axes, label="vorticity ω", shrink=0.84)
    fig.suptitle("Taylor–Green vortex: exact viscous decay (ν=0.1, RK4 Δt=0.01)")
    fig.savefig(EVIDENCE / "taylor-green.png", dpi=180)
    plt.close(fig)
    amp = np.exp(-0.2)
    uex = np.cos(xx)*np.sin(yy)*amp
    vex = -np.sin(xx)*np.cos(yy)*amp
    rel = np.sqrt(np.mean((image(f1["u"],n)-uex)**2+(image(f1["v"],n)-vex)**2)/np.mean(uex**2+vex**2))
    print(f"Taylor–Green t=1: E={diagnostics(path)[1][-1]:.6f}, Z={diagnostics(path)[2][-1]:.6f}, relative velocity error={rel:.3e}")


def make_random() -> dict:
    initial = field("random", 128, seed=2026, k_min=2, k_max=6)
    path, _, code = run_flow(initial, "random", method="rk4", nu=0.004, dt=0.01, end=10, every=0.1)
    assert code == 0, f"random baseline unexpectedly failed: {path}"
    fs = frames(path)
    t, energy, enstrophy = diagnostics(path)
    print(f"random t=0..10: E {energy[0]:.6f} -> {energy[-1]:.6f}; Z {enstrophy[0]:.6f} -> {enstrophy[-1]:.6f}")
    speed = np.hypot(image(initial["u"], 128), image(initial["v"], 128))
    print(f"random initial max speed Umax={speed.max():.6f}; advective estimate dt <= {2.83/(speed.max()*np.sqrt(2)*42):.6f}")

    chosen = []
    for target in (0, 2, 5, 10):
        chosen.append(min(fs, key=lambda f: abs(f["t"]-target)))
    bound = float(np.max(np.abs(image(chosen[0]["omega"],128))))
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.7), constrained_layout=True)
    for ax, frame in zip(axes, chosen):
        om = image(frame["omega"], 128)
        ax.imshow(om, origin="lower", extent=[0,2*np.pi,0,2*np.pi], cmap="RdBu_r", vmin=-bound, vmax=bound)
        ax.set(title=f"t={frame['t']:g}", xticks=[], yticks=[], aspect="equal")
    fig.colorbar(axes[0].images[0], ax=axes, label="vorticity ω (shared scale)", shrink=0.85)
    fig.suptitle("Random band-limited flow: filaments fade before large vortices")
    fig.savefig(EVIDENCE / "random.png", dpi=180)
    plt.close(fig)
    return initial


def perturb(initial: dict) -> dict:
    n = int(initial["n"])
    u = image(initial["u"], n)
    v = image(initial["v"], n)
    scale = max(float(np.max(np.abs(u))), float(np.max(np.abs(v))))
    x = np.arange(n)*2*np.pi/n
    xx, yy = np.meshgrid(x, x)
    du = (7e-5*scale*4/25.0)*np.cos(3*xx)*np.sin(4*yy)
    dv = (-7e-5*scale*3/25.0)*np.sin(3*xx)*np.cos(4*yy)
    result = dict(initial)
    result["u"] = (u+du).ravel().tolist()
    result["v"] = (v+dv).ravel().tolist()
    result["perturbation"] = {"relative_vorticity_amplitude": 7e-5, "M": scale}
    return result


def make_sensitivity(random_initial: dict) -> dict:
    tg = field("taylor-green", 64, nu=0.1, t=0)
    runs = {}
    for case, init, nu in (("Taylor–Green", tg, 0.1), ("random", random_initial, 0.004)):
        for tag, source in (("base", init), ("perturbed", perturb(init))):
            name = f"sensitivity-{case.lower().replace('–','').replace(' ','-')}-{tag}"
            path, _, code = run_flow(source, name, method="rk4", nu=nu, dt=0.01, end=20, every=0.5)
            assert code == 0, f"sensitivity run failed: {name}"
            runs[(case,tag)] = frames(path)
    fig, ax = plt.subplots(figsize=(7.2,4.8))
    for case in ("Taylor–Green", "random"):
        base, pert = runs[(case,"base")], runs[(case,"perturbed")]
        ts=[]; ds=[]
        for a,b in zip(base,pert):
            wa=image(a["omega"], int(round(np.sqrt(len(a["omega"])))))
            wb=image(b["omega"], int(round(np.sqrt(len(b["omega"])))))
            ts.append(a["t"])
            ds.append(np.sqrt(np.mean((wa-wb)**2))/max(np.sqrt(np.mean(wa**2)),1e-15))
        display_ds=np.maximum(ds,1e-6)
        ax.semilogy(ts,display_ds,label=case)
        (ART / f"sensitivity-{case.lower().replace('–','').replace(' ','-')}.json").write_text(json.dumps({"t":ts,"relative_distance":ds},indent=2))
        print(f"sensitivity {case}: relative distance {ds[0]:.3e} -> {ds[-1]:.3e}")
    ax.axhline(1e-6,color="gray",ls=":",label="six-decimal output floor (indicative)")
    ax.set(xlabel="time t",ylabel="||ωpert−ωbase||₂ / ||ωbase||₂",title="Sensitivity to a small physical perturbation")
    ax.grid(True,which="both",alpha=0.25);ax.legend()
    fig.tight_layout();fig.savefig(EVIDENCE/"sensitivity.png",dpi=180);plt.close(fig)
    return tg


def make_blowup(tg:dict, random_initial:dict)->None:
    tg_runs=[]
    for dt in (0.032,0.033):
        path,_,code=run_flow(tg,f"scan-taylor-green-rk4-dt{dt}",method="rk4",nu=0.1,dt=dt,end=8,every=0.5)
        tg_runs.append((dt,path,code))
        print(f"Taylor–Green dt={dt}: exit={code}, final record={diagnostics(path)[0][-1]:.6g}")

    # Bracket this seeded field's empirical limit near its advective estimate.
    trials=[]
    speed=np.hypot(image(random_initial["u"],128),image(random_initial["v"],128)).max()
    estimate=2.83/(speed*np.sqrt(2)*42)
    scan_steps=[0.020,0.025,0.030]
    for dt in scan_steps:
        path,_,code=run_flow(random_initial,f"scan-random-rk4-dt{dt}",method="rk4",nu=0.004,dt=dt,end=10,every=0.5)
        ts,es,_=diagnostics(path)
        trials.append((dt,path,code,ts[-1]))
        print(f"random RK4 dt={dt}: exit={code}, last finite/snapshot t={ts[-1]:.6g}")
    if not any(r[2]==0 for r in trials):
        for dt in (0.018,0.015625):
            path,_,code=run_flow(random_initial,f"scan-random-rk4-dt{dt}",method="rk4",nu=0.004,dt=dt,end=10,every=0.5)
            ts,_,_=diagnostics(path);trials.append((dt,path,code,ts[-1]))
            print(f"random RK4 dt={dt}: exit={code}, last finite/snapshot t={ts[-1]:.6g}")
    if not any(r[2]!=0 for r in trials):
        for dt in (0.035,0.040):
            path,_,code=run_flow(random_initial,f"scan-random-rk4-dt{dt}",method="rk4",nu=0.004,dt=dt,end=10,every=0.5)
            ts,_,_=diagnostics(path);trials.append((dt,path,code,ts[-1]))
            print(f"random RK4 dt={dt}: exit={code}, last finite/snapshot t={ts[-1]:.6g}")
    trials.sort(key=lambda r:r[0])
    stable=next((r for r in reversed(trials) if r[2]==0),None)
    unstable=next((r for r in trials if r[2]!=0 and stable is not None and r[0]>stable[0]),None)
    if stable is None or unstable is None:
        raise RuntimeError(f"could not bracket random RK4 limit; trials={[(r[0],r[2],r[3]) for r in trials]}")
    euler_path,_,euler_code=run_flow(random_initial,"scan-random-euler-dt0.01",method="euler",nu=0.004,dt=0.01,end=10,every=0.5)
    print(f"random Euler dt=.01: exit={euler_code}, last record={diagnostics(euler_path)[0][-1]:.6g}")
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    ax=axes[0]
    for dt,path,code in tg_runs:
        t,e,_=diagnostics(path);ax.semilogy(t,e,label=f"RK4 Δt={dt:g}")
    t=np.linspace(0,8,300);ax.semilogy(t,0.25*np.exp(-0.4*t),"k--",lw=1.4,label="exact Taylor–Green energy")
    ax.set(title="Diffusive stability limit, N=64, ν=0.1",xlabel="time t",ylabel="energy E");ax.grid(True,which="both",alpha=.22);ax.legend(fontsize=8)
    ax=axes[1]
    for dt,path,code,tlast in trials:
        if dt==stable[0] or dt==unstable[0]:
            t,e,_=diagnostics(path);ax.semilogy(t,e,label=f"RK4 Δt={dt:g}"+(" (unstable)" if code else ""))
    t,e,_=diagnostics(euler_path);ax.semilogy(t,e,":",label="Euler Δt=0.01")
    if unstable[2]!=0:ax.axvline(unstable[3],color="red",ls="--",alpha=.7,label=f"RK4 stop near t={unstable[3]:g}")
    ax.set(title=f"Random flow: estimate {estimate:.4f}; stable {stable[0]:g}, unstable {unstable[0]:g}",xlabel="time t",ylabel="energy E");ax.grid(True,which="both",alpha=.22);ax.legend(fontsize=8)
    fig.suptitle("Stable decay versus time-step blow-up")
    fig.tight_layout();fig.savefig(EVIDENCE/"blowup.png",dpi=180);plt.close(fig)


if __name__ == "__main__":
    make_taylor_green()
    random_field = make_random()
    tg_field = make_sensitivity(random_field)
    make_blowup(tg_field, random_field)
