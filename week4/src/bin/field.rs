use amat5315_week4::spectral::FlowSolver;
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::{json, Value};
use std::f64::consts::PI;

fn arg(args: &[String], key: &str, default: Option<&str>) -> Result<String, String> {
    let flag = format!("--{key}");
    let i = args
        .iter()
        .position(|x| x == &flag)
        .ok_or_else(|| format!("missing {flag}"));
    match i {
        Ok(i) => args
            .get(i + 1)
            .cloned()
            .ok_or_else(|| format!("missing value for {flag}")),
        Err(e) => default.map(str::to_string).ok_or(e),
    }
}
fn usize_arg(a: &[String], k: &str, d: Option<&str>) -> Result<usize, String> {
    arg(a, k, d)?
        .parse()
        .map_err(|_| format!("--{k} must be an integer"))
}
fn f64_arg(a: &[String], k: &str, d: Option<&str>) -> Result<f64, String> {
    arg(a, k, d)?
        .parse()
        .map_err(|_| format!("--{k} must be a number"))
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 1 {
        return Err("usage: field taylor-green|random ...".into());
    }
    let case = a[0].as_str();
    let n = usize_arg(&a, "n", None)?;
    if n < 4 || n % 2 != 0 {
        return Err("n must be an even integer >= 4".into());
    }
    let mut obj: Value;
    let (u, v) = if case == "taylor-green" {
        let t = f64_arg(&a, "t", Some("0"))?;
        let nu = f64_arg(&a, "nu", if t == 0.0 { Some("0") } else { None })?;
        let scale = (-2.0 * nu * t).exp();
        let mut u = Vec::with_capacity(n * n);
        let mut v = Vec::with_capacity(n * n);
        for j in 0..n {
            let y = 2.0 * PI * j as f64 / n as f64;
            for i in 0..n {
                let x = 2.0 * PI * i as f64 / n as f64;
                u.push(x.cos() * y.sin() * scale);
                v.push(-x.sin() * y.cos() * scale);
            }
        }
        obj = json!({"case":"taylor-green","n":n,"seed":Value::Null,"k_band":Value::Null});
        (u, v)
    } else if case == "random" {
        let seed = usize_arg(&a, "seed", None)? as u64;
        let kmin = usize_arg(&a, "k-min", None)?;
        let kmax = usize_arg(&a, "k-max", None)?;
        if kmin == 0 || kmax < kmin {
            return Err("require 1 <= k-min <= k-max".into());
        }
        if kmax > n / 2 {
            return Err("k-max must not exceed the Nyquist wavenumber n/2".into());
        }
        let mut wh = vec![Complex64::new(0.0, 0.0); n * n];
        for ky in -(kmax as i32)..=kmax as i32 {
            for kx in -(kmax as i32)..=kmax as i32 {
                if ky < 0 || (ky == 0 && kx <= 0) {
                    continue;
                }
                let radius = ((kx * kx + ky * ky) as f64).sqrt();
                if radius + 1e-12 < kmin as f64 || radius > kmax as f64 + 1e-12 {
                    continue;
                }
                let key = seed
                    ^ ((kx as u64).wrapping_mul(0x9E3779B97F4A7C15))
                    ^ ((ky as u64).wrapping_mul(0xD1B54A32D192ED03));
                let mut rng = ChaCha8Rng::seed_from_u64(key);
                let phase = rng.gen_range(0.0..2.0 * PI);
                let z = Complex64::from_polar(1.0, phase);
                let ix = kx.rem_euclid(n as i32) as usize;
                let iy = ky.rem_euclid(n as i32) as usize;
                let jx = (-kx).rem_euclid(n as i32) as usize;
                let jy = (-ky).rem_euclid(n as i32) as usize;
                wh[iy * n + ix] = z;
                wh[jy * n + jx] = z.conj();
            }
        }
        let solver = FlowSolver::new(n, 0.0);
        let d = solver.diagnostics(&wh);
        let scale = (0.5 / d.energy).sqrt();
        for z in &mut wh {
            *z *= scale
        }
        let d = solver.diagnostics(&wh);
        obj = json!({"case":"random","n":n,"seed":seed,"k_band":[kmin,kmax]});
        (d.u, d.v)
    } else {
        return Err("case must be taylor-green or random".into());
    };
    obj["u"] = json!(u);
    obj["v"] = json!(v);
    println!("{}", serde_json::to_string(&obj)?);
    Ok(())
}
