use num_complex::Complex64;

pub trait Integrator {
    fn name(&self) -> &'static str;
    fn step<F>(&self, y: &[Complex64], dt: f64, rhs: F) -> Vec<Complex64>
    where
        F: Fn(&[Complex64]) -> Vec<Complex64>;
}

#[derive(Clone, Copy, Debug)]
pub enum Method {
    Euler,
    Midpoint,
    Rk4,
    EqualWeightsRk4,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ForwardEuler;
#[derive(Clone, Copy, Debug, Default)]
pub struct ExplicitMidpoint;
#[derive(Clone, Copy, Debug, Default)]
pub struct ClassicalRk4;
#[derive(Clone, Copy, Debug, Default)]
pub struct EqualWeightRk4;

fn add_scaled(y: &[Complex64], k: &[Complex64], scale: f64) -> Vec<Complex64> {
    y.iter().zip(k).map(|(a, b)| *a + *b * scale).collect()
}

impl Integrator for Method {
    fn name(&self) -> &'static str {
        match self {
            Self::Euler => "euler",
            Self::Midpoint => "rk2",
            Self::Rk4 => "rk4",
            Self::EqualWeightsRk4 => "equal-rk4",
        }
    }
    fn step<F>(&self, y: &[Complex64], dt: f64, rhs: F) -> Vec<Complex64>
    where
        F: Fn(&[Complex64]) -> Vec<Complex64>,
    {
        let k1 = rhs(y);
        match self {
            Self::Euler => add_scaled(y, &k1, dt),
            Self::Midpoint => {
                let mid = add_scaled(y, &k1, dt * 0.5);
                add_scaled(y, &rhs(&mid), dt)
            }
            Self::Rk4 => {
                let k2 = rhs(&add_scaled(y, &k1, dt * 0.5));
                let k3 = rhs(&add_scaled(y, &k2, dt * 0.5));
                let k4 = rhs(&add_scaled(y, &k3, dt));
                y.iter()
                    .enumerate()
                    .map(|(i, v)| *v + (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) * (dt / 6.0))
                    .collect()
            }
            Self::EqualWeightsRk4 => {
                let k2 = rhs(&add_scaled(y, &k1, dt * 0.5));
                let k3 = rhs(&add_scaled(y, &k2, dt * 0.5));
                let k4 = rhs(&add_scaled(y, &k3, dt));
                y.iter()
                    .enumerate()
                    .map(|(i, v)| *v + (k1[i] + k2[i] + k3[i] + k4[i]) * (dt / 4.0))
                    .collect()
            }
        }
    }
}

macro_rules! delegate_integrator {
    ($type:ty, $method:ident) => {
        impl Integrator for $type {
            fn name(&self) -> &'static str {
                Method::$method.name()
            }
            fn step<F>(&self, y: &[Complex64], dt: f64, rhs: F) -> Vec<Complex64>
            where
                F: Fn(&[Complex64]) -> Vec<Complex64>,
            {
                Method::$method.step(y, dt, rhs)
            }
        }
    };
}

delegate_integrator!(ForwardEuler, Euler);
delegate_integrator!(ExplicitMidpoint, Midpoint);
delegate_integrator!(ClassicalRk4, Rk4);
delegate_integrator!(EqualWeightRk4, EqualWeightsRk4);

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn named_integrators_share_the_trait() {
        fn identity<I: Integrator>(stepper: I, expected: &str) {
            assert_eq!(stepper.name(), expected);
            let y = [Complex64::new(2.0, -1.0)];
            let next = stepper.step(&y, 0.1, |_| vec![Complex64::new(0.0, 0.0)]);
            assert_abs_diff_eq!(next[0].re, y[0].re, epsilon = 1e-14);
            assert_abs_diff_eq!(next[0].im, y[0].im, epsilon = 1e-14);
        }
        identity(ForwardEuler, "euler");
        identity(ExplicitMidpoint, "rk2");
        identity(ClassicalRk4, "rk4");
        identity(EqualWeightRk4, "equal-rk4");
    }

    #[test]
    fn measured_orders_match_the_methods() {
        for (method, expected) in [
            (Method::Euler, 1.0),
            (Method::Midpoint, 2.0),
            (Method::Rk4, 4.0),
            (Method::EqualWeightsRk4, 2.0),
        ] {
            let mut errors = Vec::new();
            for dt in [0.1_f64, 0.05, 0.025, 0.0125] {
                let steps = (1.0 / dt).round() as usize;
                let mut y = vec![Complex64::new(1.0, 0.0)];
                for _ in 0..steps {
                    y = method.step(&y, dt, |q| q.iter().map(|z| *z).collect());
                }
                errors.push((y[0] - Complex64::new(std::f64::consts::E, 0.0)).norm());
            }
            let slope = (errors[2] / errors[3]).ln() / 2.0_f64.ln();
            assert!(
                (slope - expected).abs() < 0.2,
                "{} slope {slope}",
                method.name()
            );
            assert!(errors.windows(2).all(|w| w[1] < w[0]));
        }
        assert_abs_diff_eq!(1.0, 1.0, epsilon = 0.0);
    }
}
