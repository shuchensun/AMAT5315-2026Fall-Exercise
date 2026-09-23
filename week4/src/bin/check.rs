use amat5315_week4::spectral::FlowSolver;
use serde_json::json;
use std::f64::consts::PI;
fn main() {
    let mut results = Vec::new();
    for n in [32usize, 64] {
        let s = FlowSolver::new(n, 0.1);
        let mut u = Vec::new();
        let mut v = Vec::new();
        let mut omega = Vec::new();
        let mut f = Vec::new();
        for j in 0..n {
            let y = 2.0 * PI * j as f64 / n as f64;
            for i in 0..n {
                let x = 2.0 * PI * i as f64 / n as f64;
                u.push(x.cos() * y.sin());
                v.push(-x.sin() * y.cos());
                omega.push(-2.0 * x.cos() * y.cos());
                f.push((3.0 * x).sin() * (2.0 * y).cos());
            }
        }
        let got = s.from_velocity(&u, &v);
        let d = s.diagnostics(&got);
        let werr = d
            .omega
            .iter()
            .zip(&omega)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
            / n as f64;
        let dx = s.derivative(&f, 0);
        let exact: Vec<_> = (0..n)
            .flat_map(|j| {
                (0..n).map(move |i| {
                    let x = 2.0 * PI * i as f64 / n as f64;
                    let y = 2.0 * PI * j as f64 / n as f64;
                    3.0 * (3.0 * x).cos() * (2.0 * y).cos()
                })
            })
            .collect();
        let ferr = dx
            .iter()
            .zip(&exact)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
            / n as f64;
        results.push(json!({"n":n,"taylor_green_vorticity_l2_grid":werr,"energy":d.energy,"enstrophy":d.enstrophy,"fourier_derivative_l2_grid":ferr}));
    }
    let mut fd = Vec::new();
    for n in [32usize, 64] {
        let dx = 2.0 * PI / n as f64;
        let values: Vec<_> = (0..n * n)
            .map(|q| {
                let x = 2.0 * PI * (q % n) as f64 / n as f64;
                let y = 2.0 * PI * (q / n) as f64 / n as f64;
                (3.0 * x).sin() * (2.0 * y).cos()
            })
            .collect();
        let err = (0..n * n)
            .map(|q| {
                let x = 2.0 * PI * (q % n) as f64 / n as f64;
                let y = 2.0 * PI * (q / n) as f64 / n as f64;
                let ix = q % n;
                let iy = q / n;
                let forward = iy * n + (ix + 1) % n;
                let backward = iy * n + (ix + n - 1) % n;
                let d = (values[forward] - values[backward]) / (2.0 * dx);
                let exact = 3.0 * (3.0 * x).cos() * (2.0 * y).cos();
                (d - exact).powi(2)
            })
            .sum::<f64>()
            .sqrt()
            / n as f64;
        fd.push(json!({"n":n,"centered_fd_derivative_l2":err}));
    }
    println!(
        "{}",
        json!({"spectral_checks":results,"centered_fd_checks":fd})
    );
}
