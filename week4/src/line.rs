use crate::integrator::{Integrator, Method};
use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

pub struct LineFft {
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
}
impl LineFft {
    pub fn new(n: usize) -> Self {
        let mut p = FftPlanner::new();
        Self {
            forward: p.plan_fft_forward(n),
            inverse: p.plan_fft_inverse(n),
        }
    }
    pub fn forward(&self, a: &[f64]) -> Vec<Complex64> {
        let mut z: Vec<_> = a.iter().map(|x| Complex64::new(*x, 0.0)).collect();
        self.forward.process(&mut z);
        z
    }
    pub fn inverse(&self, a: &[Complex64]) -> Vec<f64> {
        let mut z = a.to_vec();
        self.inverse.process(&mut z);
        z.iter().map(|x| x.re / a.len() as f64).collect()
    }
}

pub fn rate_hat(
    state: &[Complex64],
    n: usize,
    nu: f64,
    c: f64,
    derivative: &str,
) -> Vec<Complex64> {
    let dx = 2.0 * std::f64::consts::PI / n as f64;
    state
        .iter()
        .enumerate()
        .map(|(i, z)| {
            let k = if i <= n / 2 {
                i as i32
            } else {
                i as i32 - n as i32
            };
            let (d1, d2) = if derivative == "fd" {
                (
                    (k as f64 * dx).sin() / dx,
                    -4.0 * (0.5 * k as f64 * dx).sin().powi(2) / (dx * dx),
                )
            } else {
                (
                    if n % 2 == 0 && i == n / 2 {
                        0.0
                    } else {
                        k as f64
                    },
                    -(k as f64).powi(2),
                )
            };
            *z * Complex64::new(nu * d2, -c * d1)
        })
        .collect()
}

pub fn gaussian(n: usize, sigma: f64) -> Vec<f64> {
    let l = 2.0 * std::f64::consts::PI;
    let x0 = std::f64::consts::FRAC_PI_2;
    (0..n)
        .map(|j| {
            let x = l * j as f64 / n as f64;
            (-3..=3)
                .map(|m| {
                    let q = (x - x0 + m as f64 * l) / sigma;
                    (-0.5 * q * q).exp()
                })
                .sum()
        })
        .collect()
}

pub fn exact_gaussian(n: usize, sigma: f64, nu: f64, c: f64, t: f64) -> Vec<f64> {
    let l = 2.0 * std::f64::consts::PI;
    let st = (sigma * sigma + 2.0 * nu * t).sqrt();
    let center = (std::f64::consts::FRAC_PI_2 + c * t).rem_euclid(l);
    (0..n)
        .map(|j| {
            let x = l * j as f64 / n as f64;
            (sigma / st)
                * (-3..=3)
                    .map(|m| {
                        let q = (x - center + m as f64 * l) / st;
                        (-0.5 * q * q).exp()
                    })
                    .sum::<f64>()
        })
        .collect()
}

pub fn evolve(
    n: usize,
    sigma: f64,
    nu: f64,
    c: f64,
    dt: f64,
    t_end: f64,
    method: Method,
    derivative: &str,
) -> Vec<f64> {
    let fft = LineFft::new(n);
    let mut state = fft.forward(&gaussian(n, sigma));
    let steps = (t_end / dt).round() as usize;
    for _ in 0..steps {
        state = method.step(&state, dt, |y| rate_hat(y, n, nu, c, derivative));
    }
    fft.inverse(&state)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn single_wave_has_the_exact_multiplier() {
        let n = 64;
        let fft = LineFft::new(n);
        let u: Vec<_> = (0..n)
            .map(|j| (3.0 * (2.0 * std::f64::consts::PI * j as f64 / n as f64)).cos())
            .collect();
        let mut y = fft.forward(&u);
        let dt = 0.001;
        let steps = 100;
        for _ in 0..steps {
            y = Method::Rk4.step(&y, dt, |z| rate_hat(z, n, 0.02, 0.7, "fourier"));
        }
        let got = fft.inverse(&y);
        let t = dt * steps as f64;
        for j in 0..n {
            let x = 2.0 * std::f64::consts::PI * j as f64 / n as f64;
            let ex = (-0.02 * 9.0 * t).exp() * (3.0 * (x - 0.7 * t)).cos();
            assert!((got[j] - ex).abs() < 1e-10);
        }
    }
}
