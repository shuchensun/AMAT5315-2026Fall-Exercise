use crate::{derivative_multiplier, signed_mode};
use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

pub struct Fft2 {
    pub n: usize,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
}

impl Fft2 {
    pub fn new(n: usize) -> Self {
        let mut planner = FftPlanner::<f64>::new();
        Self {
            n,
            forward: planner.plan_fft_forward(n),
            inverse: planner.plan_fft_inverse(n),
        }
    }
    pub fn forward(&self, input: &[f64]) -> Vec<Complex64> {
        let mut data: Vec<_> = input.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        self.transform(&mut data, false);
        data
    }
    pub fn inverse_real(&self, input: &[Complex64]) -> Vec<f64> {
        let mut data = input.to_vec();
        self.transform(&mut data, true);
        let scale = 1.0 / (self.n * self.n) as f64;
        data.iter().map(|z| z.re * scale).collect()
    }
    pub fn transform(&self, data: &mut [Complex64], inverse: bool) {
        assert_eq!(data.len(), self.n * self.n);
        let fft = if inverse {
            &self.inverse
        } else {
            &self.forward
        };
        let mut scratch = vec![Complex64::new(0.0, 0.0); self.n];
        for row in 0..self.n {
            let start = row * self.n;
            fft.process_with_scratch(&mut data[start..start + self.n], &mut scratch);
        }
        let mut column = vec![Complex64::new(0.0, 0.0); self.n];
        for x in 0..self.n {
            for y in 0..self.n {
                column[y] = data[y * self.n + x];
            }
            fft.process_with_scratch(&mut column, &mut scratch);
            for y in 0..self.n {
                data[y * self.n + x] = column[y];
            }
        }
    }
}

pub struct FlowSolver {
    pub n: usize,
    pub nu: f64,
    fft: Fft2,
}

#[derive(Clone)]
pub struct Diagnostics {
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub omega: Vec<f64>,
    pub energy: f64,
    pub enstrophy: f64,
}

impl FlowSolver {
    pub fn new(n: usize, nu: f64) -> Self {
        Self {
            n,
            nu,
            fft: Fft2::new(n),
        }
    }
    pub fn mask(&self, a: &mut [Complex64]) {
        let cutoff = (self.n / 3) as i32;
        for y in 0..self.n {
            for x in 0..self.n {
                if signed_mode(x, self.n).abs() > cutoff || signed_mode(y, self.n).abs() > cutoff {
                    a[y * self.n + x] = Complex64::new(0.0, 0.0);
                }
            }
        }
    }
    pub fn from_velocity(&self, u: &[f64], v: &[f64]) -> Vec<Complex64> {
        let uh = self.fft.forward(u);
        let vh = self.fft.forward(v);
        let mut w = vec![Complex64::new(0.0, 0.0); self.n * self.n];
        for y in 0..self.n {
            for x in 0..self.n {
                let i = y * self.n + x;
                let kx = signed_mode(x, self.n);
                let ky = signed_mode(y, self.n);
                w[i] = derivative_multiplier(kx, self.n) * vh[i]
                    - derivative_multiplier(ky, self.n) * uh[i];
            }
        }
        self.mask(&mut w);
        w
    }
    pub fn diagnostics(&self, omega_hat: &[Complex64]) -> Diagnostics {
        let (u_hat, v_hat) = self.velocity_hat(omega_hat);
        let u = self.fft.inverse_real(&u_hat);
        let v = self.fft.inverse_real(&v_hat);
        let omega = self.fft.inverse_real(omega_hat);
        let inv = 1.0 / (self.n * self.n) as f64;
        let energy = 0.5 * inv * u.iter().zip(&v).map(|(a, b)| a * a + b * b).sum::<f64>();
        let enstrophy = 0.5 * inv * omega.iter().map(|a| a * a).sum::<f64>();
        Diagnostics {
            u,
            v,
            omega,
            energy,
            enstrophy,
        }
    }
    pub fn velocity_hat(&self, omega_hat: &[Complex64]) -> (Vec<Complex64>, Vec<Complex64>) {
        let mut u = vec![Complex64::new(0.0, 0.0); self.n * self.n];
        let mut v = u.clone();
        for y in 0..self.n {
            for x in 0..self.n {
                let i = y * self.n + x;
                let kx = signed_mode(x, self.n);
                let ky = signed_mode(y, self.n);
                let k2 = (kx * kx + ky * ky) as f64;
                if k2 > 0.0 {
                    let psi = omega_hat[i] / k2;
                    u[i] = derivative_multiplier(ky, self.n) * psi;
                    v[i] = -derivative_multiplier(kx, self.n) * psi;
                }
            }
        }
        (u, v)
    }
    fn spectral_derivative(&self, a: &[Complex64], axis: usize) -> Vec<Complex64> {
        let mut out = a.to_vec();
        for y in 0..self.n {
            for x in 0..self.n {
                let k = if axis == 0 {
                    signed_mode(x, self.n)
                } else {
                    signed_mode(y, self.n)
                };
                let i = y * self.n + x;
                out[i] = derivative_multiplier(k, self.n) * out[i];
            }
        }
        out
    }
    fn laplacian(&self, a: &[Complex64]) -> Vec<Complex64> {
        let mut out = a.to_vec();
        for y in 0..self.n {
            for x in 0..self.n {
                let kx = signed_mode(x, self.n) as f64;
                let ky = signed_mode(y, self.n) as f64;
                let i = y * self.n + x;
                out[i] = -(kx * kx + ky * ky) * out[i];
            }
        }
        out
    }
    pub fn rhs(&self, omega_hat: &[Complex64]) -> Vec<Complex64> {
        let (u_hat, v_hat) = self.velocity_hat(omega_hat);
        let u = self.fft.inverse_real(&u_hat);
        let v = self.fft.inverse_real(&v_hat);
        let ox = self
            .fft
            .inverse_real(&self.spectral_derivative(omega_hat, 0));
        let oy = self
            .fft
            .inverse_real(&self.spectral_derivative(omega_hat, 1));
        let u_omega_x: Vec<f64> = u.iter().zip(&ox).map(|(u, ox)| u * ox).collect();
        let v_omega_y: Vec<f64> = v.iter().zip(&oy).map(|(v, oy)| v * oy).collect();
        let mut u_omega_x_hat = self.fft.forward(&u_omega_x);
        let mut v_omega_y_hat = self.fft.forward(&v_omega_y);
        self.mask(&mut u_omega_x_hat);
        self.mask(&mut v_omega_y_hat);
        let diff = self.laplacian(omega_hat);
        let mut out: Vec<_> = u_omega_x_hat
            .iter()
            .zip(&v_omega_y_hat)
            .zip(diff)
            .map(|((x, y), b)| -*x - *y + b * self.nu)
            .collect();
        self.mask(&mut out);
        out
    }
    pub fn derivative(&self, a: &[f64], axis: usize) -> Vec<f64> {
        self.fft
            .inverse_real(&self.spectral_derivative(&self.fft.forward(a), axis))
    }
    pub fn fft(&self) -> &Fft2 {
        &self.fft
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    #[test]
    fn fft_roundtrip() {
        let n = 16;
        let f = Fft2::new(n);
        let a: Vec<_> = (0..n * n).map(|i| (0.17 * i as f64).sin()).collect();
        let b = f.inverse_real(&f.forward(&a));
        for (x, y) in a.iter().zip(b) {
            assert_abs_diff_eq!(*x, y, epsilon = 1e-12)
        }
    }
    #[test]
    fn derivative_and_taylor_green_decay() {
        let n = 32;
        let s = FlowSolver::new(n, 0.1);
        let mut u = Vec::new();
        let mut v = Vec::new();
        let mut exact_w = Vec::new();
        for j in 0..n {
            let y = 2.0 * std::f64::consts::PI * j as f64 / n as f64;
            for i in 0..n {
                let x = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                u.push(x.cos() * y.sin());
                v.push(-x.sin() * y.cos());
                exact_w.push(-2.0 * x.cos() * y.cos());
            }
        }
        let w = s.from_velocity(&u, &v);
        let d = s.diagnostics(&w);
        for (a, b) in d.omega.iter().zip(&exact_w) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-11)
        }
        assert_abs_diff_eq!(d.energy, 0.25, epsilon = 1e-12);
        let dx = s.derivative(&u, 0);
        for j in 0..n {
            let y = 2.0 * std::f64::consts::PI * j as f64 / n as f64;
            for i in 0..n {
                let x = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                assert_abs_diff_eq!(dx[j * n + i], -x.sin() * y.sin(), epsilon = 1e-11)
            }
        }
    }
}
