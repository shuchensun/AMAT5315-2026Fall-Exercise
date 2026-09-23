use amat5315_week4::{
    integrator::{Integrator, Method},
    round6,
    spectral::FlowSolver,
};
use num_complex::Complex64;
use serde_json::{json, Map, Value};
use std::{
    fs,
    io::{self, Read, Write},
    path::PathBuf,
};

fn opt(a: &[String], k: &str) -> Result<String, String> {
    let f = format!("--{k}");
    let i = a
        .iter()
        .position(|x| x == &f)
        .ok_or_else(|| format!("missing {f}"))?;
    a.get(i + 1)
        .cloned()
        .ok_or_else(|| format!("missing value for {f}"))
}
fn number(a: &[String], k: &str) -> Result<f64, String> {
    opt(a, k)?
        .parse()
        .map_err(|_| format!("--{k} must be a number"))
}
fn method(s: &str) -> Result<Method, String> {
    match s {
        "euler" => Ok(Method::Euler),
        "rk2" => Ok(Method::Midpoint),
        "rk4" => Ok(Method::Rk4),
        _ => Err("method must be euler, rk2, or rk4".into()),
    }
}
fn rounded(a: &[f64]) -> Vec<f64> {
    a.iter().map(|x| round6(*x)).collect()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let m = method(&opt(&a, "method")?)?;
    let nu = number(&a, "nu")?;
    let dt = number(&a, "dt")?;
    let end = number(&a, "t-end")?;
    let every = number(&a, "every")?;
    let out = PathBuf::from(opt(&a, "out")?);
    if !(dt > 0.0 && every > 0.0 && end >= 0.0 && nu >= 0.0) {
        return Err("require dt,every > 0 and t-end,nu >= 0".into());
    }
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let source: Value = serde_json::from_str(&input)?;
    let n = source["n"].as_u64().ok_or("field JSON has no integer n")? as usize;
    let readfield = |key: &str| -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let vals = source[key]
            .as_array()
            .ok_or("field JSON is missing u/v array")?;
        let v = vals
            .iter()
            .map(|x| x.as_f64().ok_or("non-numeric field value"))
            .collect::<Result<Vec<_>, _>>()?;
        if v.len() != n * n {
            return Err(format!("{} has {} values, expected {}", key, v.len(), n * n).into());
        }
        Ok(v)
    };
    let solver = FlowSolver::new(n, nu);
    let mut omega = solver.from_velocity(&readfield("u")?, &readfield("v")?);
    let stride = (every / dt).round().max(1.0) as usize;
    fs::create_dir_all(&out)?;
    let mut metadata = source.as_object().cloned().unwrap_or_else(Map::new);
    metadata.remove("u");
    metadata.remove("v");
    metadata.insert("method".into(), json!(m.name()));
    metadata.insert("nu".into(), json!(nu));
    metadata.insert("dt".into(), json!(dt));
    metadata.insert("t_end".into(), json!(end));
    metadata.insert("snapshot_every".into(), json!(every));
    fs::write(
        out.join("run.json"),
        serde_json::to_vec_pretty(&Value::Object(metadata))?,
    )?;
    let mut fields = fs::File::create(out.join("fields.jsonl"))?;
    println!("t\tE\tZ");
    let mut save = |step: usize,
                    state: &[Complex64]|
     -> Result<(f64, f64, bool), Box<dyn std::error::Error>> {
        let d = solver.diagnostics(state);
        let time = step as f64 * dt;
        let finite = d.energy.is_finite() && d.enstrophy.is_finite();
        if finite {
            let frame = json!({"t":round6(time),"step":step,"u":rounded(&d.u),"v":rounded(&d.v),"omega":rounded(&d.omega)});
            writeln!(fields, "{}", serde_json::to_string(&frame)?)?;
        }
        println!(
            "{:.8}\t{}\t{}",
            time,
            if finite {
                format!("{:.9}", d.energy)
            } else {
                "non-finite".into()
            },
            if finite {
                format!("{:.9}", d.enstrophy)
            } else {
                "non-finite".into()
            }
        );
        Ok((d.energy, d.enstrophy, finite))
    };
    let (_, _, mut finite) = save(0, &omega)?;
    let max_steps = (end / dt).floor() as usize;
    for step in 1..=max_steps {
        omega = m.step(&omega, dt, |y| solver.rhs(y));
        let d = solver.diagnostics(&omega);
        finite = d.energy.is_finite() && d.enstrophy.is_finite();
        if !finite {
            println!("{:.8}\tnon-finite\tnon-finite", step as f64 * dt);
            break;
        }
        if step % stride == 0 {
            let _ = save(step, &omega)?;
        }
    }
    fields.flush()?;
    if !finite {
        std::process::exit(1)
    }
    Ok(())
}
