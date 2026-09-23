use amat5315_week4::{
    integrator::{Integrator, Method},
    line::{exact_gaussian, gaussian, LineFft},
};
use num_complex::Complex64;
use serde_json::json;
use std::io::{self, Write};

fn val(a: &[String], k: &str, d: Option<&str>) -> Result<String, String> {
    let f = format!("--{k}");
    match a.iter().position(|x| x == &f) {
        Some(i) => a.get(i + 1).cloned().ok_or_else(|| format!("missing {f}")),
        None => d.map(str::to_owned).ok_or_else(|| format!("missing {f}")),
    }
}
fn num<T: std::str::FromStr>(a: &[String], k: &str, d: Option<&str>) -> Result<T, String> {
    val(a, k, d)?.parse().map_err(|_| format!("invalid --{k}"))
}
fn method(s: &str) -> Result<Method, String> {
    match s {
        "euler" => Ok(Method::Euler),
        "rk2" => Ok(Method::Midpoint),
        "rk4" => Ok(Method::Rk4),
        "equal-rk4" => Ok(Method::EqualWeightsRk4),
        _ => Err("method must be euler, rk2, rk4, or equal-rk4".into()),
    }
}
fn stable(z: Complex64, m: Method) -> f64 {
    let y = [Complex64::new(1.0, 0.0)];
    m.step(&y, 1.0, |q| q.iter().map(|v| *v * z).collect())[0].norm()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.is_empty() {
        return Err("usage: line pulse|growth|growth-grid ...".into());
    }
    match a[0].as_str() {
        "pulse" => {
            let method = method(&val(&a, "method", Some("rk4"))?)?;
            let n = num(&a, "n", Some("64"))?;
            let nu = num(&a, "nu", Some("0.05"))?;
            let c = num(&a, "c", Some("1"))?;
            let dt: f64 = num(&a, "dt", Some("0.01"))?;
            let end: f64 = num(&a, "t-end", Some("1"))?;
            let sigma = num(&a, "sigma", Some("0.35"))?;
            let deriv = val(&a, "derivative", Some("fourier"))?;
            let history = a.iter().any(|x| x == "--history");
            let fft = LineFft::new(n);
            let mut y = fft.forward(&gaussian(n, sigma));
            let mut frames = vec![json!({"t":0.0,"u":gaussian(n,sigma)})];
            let mut t = 0.0;
            while t < end - 1e-13 {
                let h = dt.min(end - t);
                y = method.step(&y, h, |q| {
                    amat5315_week4::line::rate_hat(q, n, nu, c, &deriv)
                });
                t += h;
                if history {
                    let u = fft.inverse(&y);
                    frames.push(json!({"t":t,"u":u.iter().map(|x|if x.is_finite(){json!(x)}else{serde_json::Value::Null}).collect::<Vec<_>>()}));
                }
            }
            let final_u = fft.inverse(&y);
            let exact = exact_gaussian(n, sigma, nu, c, end);
            let err = final_u
                .iter()
                .zip(&exact)
                .filter(|(x, _)| x.is_finite())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f64::max);
            let x: Vec<f64> = (0..n)
                .map(|j| 2.0 * std::f64::consts::PI * j as f64 / n as f64)
                .collect();
            let result = if history {
                json!({"method":method.name(),"derivative":deriv,"n":n,"nu":nu,"c":c,"dt":dt,"t_end":end,"sigma":sigma,"x":x,"initial":gaussian(n,sigma),"final":final_u,"exact":exact,"max_error":err,"history":frames})
            } else {
                json!({"method":method.name(),"derivative":deriv,"n":n,"nu":nu,"c":c,"dt":dt,"t_end":end,"sigma":sigma,"x":x,"initial":gaussian(n,sigma),"final":final_u,"exact":exact,"max_error":err})
            };
            println!("{}", serde_json::to_string(&result)?);
        }
        "growth" => {
            let m = method(&val(&a, "method", Some("rk4"))?)?;
            let z = Complex64::new(num(&a, "re", None)?, num(&a, "im", None)?);
            println!("{:.15e}", stable(z, m));
        }
        "stability-limit" => {
            let m = method(&val(&a, "method", Some("rk4"))?)?;
            let n: usize = num(&a, "n", Some("64"))?;
            let nu: f64 = num(&a, "nu", Some("0.05"))?;
            let c: f64 = num(&a, "c", Some("1"))?;
            let max_growth = |dt: f64| {
                (0..n)
                    .map(|index| {
                        let k = if index <= n / 2 {
                            index as i32
                        } else {
                            index as i32 - n as i32
                        };
                        let derivative = if n % 2 == 0 && index == n / 2 {
                            0.0
                        } else {
                            k as f64
                        };
                        stable(
                            Complex64::new(-nu * (k as f64).powi(2), -c * derivative) * dt,
                            m,
                        )
                    })
                    .fold(0.0_f64, f64::max)
            };
            let mut low = 0.0;
            let mut high = 1.0;
            for _ in 0..70 {
                let mid = 0.5 * (low + high);
                if max_growth(mid) <= 1.0 + 1e-12 {
                    low = mid;
                } else {
                    high = mid;
                }
            }
            println!(
                "{}",
                json!({"method":m.name(),"n":n,"nu":nu,"c":c,"critical_dt":low,"max_growth_at_limit":max_growth(low)})
            );
        }
        "growth-grid" => {
            let nx: usize = num(&a, "nx", Some("241"))?;
            let ny: usize = num(&a, "ny", Some("241"))?;
            let xmin: f64 = num(&a, "xmin", Some("-4"))?;
            let xmax: f64 = num(&a, "xmax", Some("2"))?;
            let ymin: f64 = num(&a, "ymin", Some("-4"))?;
            let ymax: f64 = num(&a, "ymax", Some("4"))?;
            let m = method(&val(&a, "method", Some("rk4"))?)?;
            let mut out = io::BufWriter::new(io::stdout().lock());
            for j in 0..ny {
                let im = ymin + (ymax - ymin) * j as f64 / (ny - 1) as f64;
                for i in 0..nx {
                    let re = xmin + (xmax - xmin) * i as f64 / (nx - 1) as f64;
                    writeln!(
                        out,
                        "{re:.9},{im:.9},{:.12e}",
                        stable(Complex64::new(re, im), m)
                    )?;
                }
            }
        }
        _ => return Err("command must be pulse, growth, or growth-grid".into()),
    }
    Ok(())
}
