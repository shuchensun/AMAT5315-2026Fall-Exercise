pub mod integrator;
pub mod line;
pub mod spectral;

use num_complex::Complex64;

pub fn signed_mode(i: usize, n: usize) -> i32 {
    if i <= n / 2 {
        i as i32
    } else {
        i as i32 - n as i32
    }
}

pub fn derivative_multiplier(k: i32, n: usize) -> Complex64 {
    if n % 2 == 0 && k.unsigned_abs() as usize == n / 2 {
        Complex64::new(0.0, 0.0)
    } else {
        Complex64::new(0.0, k as f64)
    }
}

pub fn round6(x: f64) -> f64 {
    (x * 1_000_000.0).round() / 1_000_000.0
}
