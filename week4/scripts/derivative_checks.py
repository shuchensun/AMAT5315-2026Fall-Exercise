#!/usr/bin/env python3
"""Run the solver's spectral/Taylor–Green and centered-difference checks."""
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
CHECK = WEEK / "target" / "release" / "check"
OUT = WEEK / "artifacts" / "derivatives.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
result = json.loads(subprocess.check_output([str(CHECK)], text=True))
OUT.write_text(json.dumps(result, indent=2) + "\n")
for item in result["spectral_checks"]:
    print(f"N={item['n']}: Taylor–Green ω error={item['taylor_green_vorticity_l2_grid']:.3e}; "
          f"Fourier derivative error={item['fourier_derivative_l2_grid']:.3e}; E={item['energy']:.6f}, Z={item['enstrophy']:.6f}")
for item in result["centered_fd_checks"]:
    print(f"N={item['n']}: centered-difference derivative error={item['centered_fd_derivative_l2']:.6e}")
