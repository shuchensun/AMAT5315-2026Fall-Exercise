#!/usr/bin/env python3
"""Measure RK4 order on Taylor–Green and random flow; select a step by Richardson."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from flow_experiments import EVIDENCE, ART, field, frames, image, run_flow


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a-b)**2)) / np.sqrt(np.mean(b**2)))


def fitted_slope(steps: list[float], errors: list[float]) -> float:
    return float(np.polyfit(np.log(steps), np.log(errors), 1)[0])


def final_frame(path: Path) -> dict:
    all_frames = frames(path)
    return min(all_frames, key=lambda frame: abs(frame["t"]-2.0))


def run_order() -> tuple[list[float], list[float], float]:
    steps = [0.4, 0.25, 0.2]
    n, nu, end = 8, 0.5, 2.0
    u0 = field("taylor-green", n, nu=nu, t=0)
    exact = field("taylor-green", n, nu=nu, t=end)
    ue, ve = np.asarray(exact["u"]), np.asarray(exact["v"])
    errors=[]
    for dt in steps:
        path,_,code=run_flow(u0,f"order/rk4-dt{dt:g}",method="rk4",nu=nu,dt=dt,end=end,every=end)
        if code: raise RuntimeError(f"order run failed at dt={dt}")
        f=final_frame(path)
        un,vn=np.asarray(f["u"]),np.asarray(f["v"])
        error=np.sqrt(np.mean((un-ue)**2+(vn-ve)**2))/np.sqrt(np.mean(ue**2+ve**2))
        errors.append(float(error))
        print(f"Taylor–Green order: dt={dt:g}, relative velocity error={error:.8e}")
    fit=fitted_slope(steps,errors)
    print(f"Taylor–Green RK4 fitted slope={fit:.4f}")
    fig,ax=plt.subplots(figsize=(6.6,4.8))
    ax.loglog(steps,errors,"o",label=f"measured RK4, slope={fit:.3f}")
    xx=np.geomspace(min(steps)*0.9,max(steps)*1.1,100)
    ref=errors[-1]*(xx/steps[-1])**4
    ax.loglog(xx,ref,"--",label="fourth-order guide")
    ax.set(xlabel="time step Δt",ylabel="relative velocity error at t=2",title="Taylor–Green temporal order, N=8, ν=0.5")
    ax.grid(True,which="both",alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(EVIDENCE/"order.png",dpi=180);plt.close(fig)
    return steps,errors,fit


def run_convergence() -> dict:
    n,nu,end=128,0.004,2.0
    initial=field("random",n,seed=2026,k_min=2,k_max=6)
    steps=[0.02,0.0125,0.01]
    results={}
    solutions={}
    for dt in [*steps,0.0025]:
        path,_,code=run_flow(initial,f"convergence/random-rk4-dt{dt:g}",method="rk4",nu=nu,dt=dt,end=end,every=end)
        if code:raise RuntimeError(f"random convergence run failed at dt={dt}")
        frame=final_frame(path)
        solutions[dt]=np.asarray(frame["omega"],dtype=float)
    reference=solutions[0.0025]
    errors=[relative_error(solutions[dt],reference) for dt in steps]
    fit=fitted_slope(steps,errors)
    for dt,error in zip(steps,errors):
        results[f"{dt:g}"]={"relative_omega_error_vs_dt_0.0025":error}
        print(f"random RK4: dt={dt:g}, relative omega error={error:.8e}")
    results["0.0025"]={"relative_omega_error_vs_dt_0.0025":0.0,"reference":True}
    fine_difference=relative_error(solutions[0.02],solutions[0.01])
    # The ratio uses the finer field in the denominator and fourth-order divisor 2^4-1.
    richardson_e01=float(np.sqrt(np.mean((solutions[0.02]-solutions[0.01])**2)) /
                         (15.0*np.sqrt(np.mean(solutions[0.01]**2))))
    predicted_by_step={dt:richardson_e01*(dt/0.01)**4 for dt in steps}
    eligible=[dt for dt in steps if predicted_by_step[dt]<5e-6]
    candidate=max(eligible) if eligible else None
    predicted=predicted_by_step[candidate] if candidate is not None else None
    measured=errors[steps.index(candidate)] if candidate is not None else None
    results["fit_slope"]=fit
    results["richardson"]={"base_fine_step":0.01,"coarse_step":0.02,"estimated_error_at_0.01":richardson_e01,
                           "candidate_predicted_errors":{f"{dt:g}":predicted_by_step[dt] for dt in steps},
                           "predicted_error_at_selected_step":predicted,"measured_error_vs_reference":measured,
                           "tolerance":5e-6,"selected_step":candidate,
                           "coarse_fine_difference_relative":fine_difference}
    (EVIDENCE/"convergence.json").write_text(json.dumps(results,indent=2)+"\n")
    print(f"random convergence fitted slope={fit:.4f}")
    print(f"Richardson predicts error {predicted:.4e} at dt={candidate:g}; measured {measured:.4e}; tolerance 5e-6")

    fig,ax=plt.subplots(figsize=(7.2,5.0))
    ax.loglog(steps,errors,"o-",label=f"against dt=0.0025 reference (slope={fit:.2f})")
    ax.axhline(5e-6,color="black",ls="--",label="error tolerance 5×10⁻⁶")
    if candidate is not None:
        ax.axvline(candidate,color="tab:green",ls=":",label=f"selected Δt={candidate:g}")
        ax.scatter([candidate],[predicted],marker="s",s=55,color="tab:green",label="Richardson prediction")
    ax.set(xlabel="RK4 time step Δt",ylabel="relative vorticity error at t=2",title="Random-flow self-convergence and step selection")
    ax.grid(True,which="both",alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(EVIDENCE/"convergence.png",dpi=180);plt.close(fig)
    return results


if __name__ == "__main__":
    run_order()
    run_convergence()
