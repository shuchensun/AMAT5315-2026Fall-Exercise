use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use std::env;
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Update {
    Metropolis,
    Wolff,
}

impl Update {
    fn as_str(self) -> &'static str {
        match self {
            Self::Metropolis => "metropolis",
            Self::Wolff => "wolff",
        }
    }
}

#[derive(Debug)]
struct Config {
    update: Update,
    l: usize,
    t_from: f64,
    t_to: f64,
    t_step: f64,
    discard: usize,
    measure: usize,
    seed: u64,
    every: usize,
    out: String,
}

#[derive(Serialize)]
struct RunMetadata<'a> {
    #[serde(rename = "L")]
    l: usize,
    update: &'a str,
    t_grid: Vec<f64>,
    discard: usize,
    measure: usize,
    seed: u64,
    sample_every: usize,
    time_unit: &'a str,
}

#[derive(Serialize)]
struct SeriesRow {
    #[serde(rename = "L")]
    l: usize,
    #[serde(rename = "T")]
    t: f64,
    sweep: usize,
    #[serde(rename = "M")]
    m: f64,
    #[serde(rename = "E")]
    e: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    cluster_size: Option<usize>,
}

#[derive(Serialize)]
struct SpinFrame {
    #[serde(rename = "L")]
    l: usize,
    #[serde(rename = "T")]
    t: f64,
    sweep: usize,
    m: f64,
    spins: Vec<i8>,
}

struct Lattice {
    l: usize,
    spins: Vec<i8>,
    energy: i32,
}

impl Lattice {
    fn all_up(l: usize) -> Self {
        let n = l * l;
        Self {
            l,
            spins: vec![1; n],
            energy: -2 * n as i32,
        }
    }

    #[inline]
    fn idx(&self, row: usize, col: usize) -> usize {
        (row % self.l) * self.l + (col % self.l)
    }

    #[inline]
    fn neighbours(&self, idx: usize) -> [usize; 4] {
        let row = idx / self.l;
        let col = idx % self.l;
        [
            self.idx((row + self.l - 1) % self.l, col),
            self.idx((row + 1) % self.l, col),
            self.idx(row, (col + self.l - 1) % self.l),
            self.idx(row, (col + 1) % self.l),
        ]
    }

    #[inline]
    fn magnetization(&self) -> f64 {
        self.spins.iter().map(|&s| s as f64).sum::<f64>() / self.spins.len() as f64
    }

    #[cfg(test)]
    fn recompute_energy(&self) -> i32 {
        let mut energy = 0i32;
        for row in 0..self.l {
            for col in 0..self.l {
                let i = self.idx(row, col);
                let right = self.idx(row, (col + 1) % self.l);
                let down = self.idx((row + 1) % self.l, col);
                energy -= (self.spins[i] * self.spins[right]) as i32;
                energy -= (self.spins[i] * self.spins[down]) as i32;
            }
        }
        energy
    }

    fn metropolis_sweep(&mut self, t: f64, rng: &mut ChaCha8Rng) -> usize {
        let n = self.spins.len();
        let mut accepted = 0usize;
        let mut acceptance = [0.0f64; 5];
        for (slot, delta) in [-8i32, -4, 0, 4, 8].into_iter().enumerate() {
            acceptance[slot] = if delta <= 0 { 1.0 } else { (-delta as f64 / t).exp() };
        }
        for _ in 0..n {
            let i = rng.gen_range(0..n);
            let neighbour_sum: i8 = self
                .neighbours(i)
                .iter()
                .map(|&j| self.spins[j])
                .sum();
            let delta = 2 * self.spins[i] as i32 * neighbour_sum as i32;
            let slot = ((delta + 8) / 4) as usize;
            if rng.gen::<f64>() < acceptance[slot] {
                self.spins[i] = -self.spins[i];
                self.energy += delta;
                accepted += 1;
            }
        }
        accepted
    }

    fn wolff_move(&mut self, t: f64, rng: &mut ChaCha8Rng, marked: &mut [bool]) -> usize {
        let p_add = 1.0 - (-2.0 / t).exp();
        let seed = rng.gen_range(0..self.spins.len());
        let original_spin = self.spins[seed];
        let mut cluster = Vec::with_capacity(self.spins.len().min(1024));
        let mut stack = vec![seed];
        marked[seed] = true;

        while let Some(i) = stack.pop() {
            cluster.push(i);
            for j in self.neighbours(i) {
                if !marked[j] && self.spins[j] == original_spin && rng.gen::<f64>() < p_add {
                    marked[j] = true;
                    stack.push(j);
                }
            }
        }

        let mut delta_energy = 0i32;
        for &i in &cluster {
            for j in self.neighbours(i) {
                if !marked[j] {
                    delta_energy += 2 * self.spins[i] as i32 * self.spins[j] as i32;
                }
            }
        }
        for &i in &cluster {
            self.spins[i] = -self.spins[i];
            marked[i] = false;
        }
        self.energy += delta_energy;
        cluster.len()
    }
}

fn usage() -> &'static str {
    "ising --update metropolis|wolff --l L --t-from T --t-to T --t-step DT --discard N --measure N --seed SEED --every N --out DIR"
}

fn parse_args() -> Result<Config, String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut values = std::collections::HashMap::<String, String>::new();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].clone();
        if key == "--help" || key == "-h" {
            println!("{}", usage());
            std::process::exit(0);
        }
        if !key.starts_with("--") || i + 1 >= args.len() {
            return Err(format!("expected a flag followed by a value; usage: {}", usage()));
        }
        values.insert(key.trim_start_matches("--").to_string(), args[i + 1].clone());
        i += 2;
    }
    let get = |name: &str| values.get(name).ok_or_else(|| format!("missing --{name}"));
    let update = match get("update")?.as_str() {
        "metropolis" => Update::Metropolis,
        "wolff" => Update::Wolff,
        other => return Err(format!("unknown --update {other}")),
    };
    let parse = |name: &str| -> Result<String, String> { get(name).cloned() };
    let l: usize = parse("l")?.parse().map_err(|_| "--l must be an integer".to_string())?;
    let t_from: f64 = parse("t-from")?.parse().map_err(|_| "--t-from must be a number".to_string())?;
    let t_to: f64 = parse("t-to")?.parse().map_err(|_| "--t-to must be a number".to_string())?;
    let t_step: f64 = parse("t-step")?.parse().map_err(|_| "--t-step must be a number".to_string())?;
    let discard: usize = parse("discard")?.parse().map_err(|_| "--discard must be an integer".to_string())?;
    let measure: usize = parse("measure")?.parse().map_err(|_| "--measure must be an integer".to_string())?;
    let seed: u64 = parse("seed")?.parse().map_err(|_| "--seed must be an integer".to_string())?;
    let every: usize = parse("every").unwrap_or_else(|_| "0".to_string()).parse().map_err(|_| "--every must be an integer".to_string())?;
    let out = parse("out")?;
    if l < 2 || !t_step.is_finite() || t_step == 0.0 || !t_from.is_finite() || !t_to.is_finite() || t_from == t_to && t_step.signum() != 1.0 {
        return Err("invalid lattice or temperature range".to_string());
    }
    if (t_to - t_from) * t_step < 0.0 {
        return Err("--t-step must point from --t-from toward --t-to".to_string());
    }
    if discard == 0 && measure == 0 {
        return Err("--discard and --measure cannot both be zero".to_string());
    }
    Ok(Config { update, l, t_from, t_to, t_step, discard, measure, seed, every, out })
}

fn temperature_grid(from: f64, to: f64, step: f64) -> Vec<f64> {
    let count = (((to - from) / step) + 1e-10).round() as usize;
    let mut grid = Vec::with_capacity(count + 1);
    for i in 0..=count {
        let mut t = from + step * i as f64;
        if (t - to).abs() < 1e-10 {
            t = to;
        }
        grid.push((t * 1e12).round() / 1e12);
    }
    grid
}

#[inline]
fn round6(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

fn writer(path: &Path) -> Result<BufWriter<File>, Box<dyn Error>> {
    Ok(BufWriter::new(OpenOptions::new().create(true).truncate(true).write(true).open(path)?))
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args().map_err(|e| format!("{e}\nusage: {}", usage()))?;
    let grid = temperature_grid(config.t_from, config.t_to, config.t_step);
    let out_dir = Path::new(&config.out);
    fs::create_dir_all(out_dir)?;
    let metadata = RunMetadata {
        l: config.l,
        update: config.update.as_str(),
        t_grid: grid.clone(),
        discard: config.discard,
        measure: config.measure,
        seed: config.seed,
        sample_every: 1,
        time_unit: match config.update { Update::Metropolis => "sweep", Update::Wolff => "cluster_flip" },
    };
    serde_json::to_writer_pretty(writer(&out_dir.join("run.json"))?, &metadata)?;
    let mut series = writer(&out_dir.join("series.jsonl"))?;
    let mut spins_writer = if config.every > 0 { Some(writer(&out_dir.join("spins.jsonl"))?) } else { None };
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut lattice = Lattice::all_up(config.l);
    let mut marked = vec![false; config.l * config.l];
    let mut cumulative_step = 0usize;
    println!("T\tmean_abs_M\t{}", match config.update { Update::Metropolis => "acceptance", Update::Wolff => "mean_cluster_size" });

    for &t in &grid {
        let mut accepted = 0usize;
        let mut proposals = 0usize;
        let mut cluster_total = 0usize;
        for _ in 0..config.discard {
            match config.update {
                Update::Metropolis => { accepted += lattice.metropolis_sweep(t, &mut rng); proposals += config.l * config.l; }
                Update::Wolff => { lattice.wolff_move(t, &mut rng, &mut marked); }
            }
            cumulative_step += 1;
        }
        let mut abs_m_sum = 0.0;
        for sweep in 1..=config.measure {
            let cluster_size = match config.update {
                Update::Metropolis => { accepted += lattice.metropolis_sweep(t, &mut rng); proposals += config.l * config.l; None }
                Update::Wolff => { let c = lattice.wolff_move(t, &mut rng, &mut marked); cluster_total += c; Some(c) }
            };
            cumulative_step += 1;
            let m = lattice.magnetization();
            let e = lattice.energy as f64 / (config.l * config.l) as f64;
            abs_m_sum += m.abs();
            let row = SeriesRow { l: config.l, t, sweep, m: round6(m), e: round6(e), cluster_size };
            serde_json::to_writer(&mut series, &row)?;
            series.write_all(b"\n")?;
            if let Some(ref mut frame_writer) = spins_writer {
                if sweep % config.every == 0 {
                    let frame = SpinFrame { l: config.l, t, sweep: cumulative_step, m: round6(m), spins: lattice.spins.clone() };
                    serde_json::to_writer(&mut *frame_writer, &frame)?;
                    frame_writer.write_all(b"\n")?;
                }
            }
        }
        let mean_abs_m = abs_m_sum / config.measure.max(1) as f64;
        match config.update {
            Update::Metropolis => println!("{t:.6}\t{mean_abs_m:.6}\t{:.6}", accepted as f64 / proposals.max(1) as f64),
            Update::Wolff => println!("{t:.6}\t{mean_abs_m:.6}\t{:.6}", cluster_total as f64 / config.measure.max(1) as f64),
        }
    }
    series.flush()?;
    if let Some(mut f) = spins_writer { f.flush()?; }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_up_energy_is_minus_two_per_site() {
        let lattice = Lattice::all_up(4);
        assert_eq!(lattice.energy, -32);
        assert_eq!(lattice.recompute_energy(), lattice.energy);
    }

    #[test]
    fn incremental_energy_matches_recomputation() {
        let mut lattice = Lattice::all_up(8);
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        for _ in 0..100 {
            lattice.metropolis_sweep(2.3, &mut rng);
            assert_eq!(lattice.energy, lattice.recompute_energy());
        }
    }

    #[test]
    fn wolff_preserves_incremental_energy() {
        let mut lattice = Lattice::all_up(8);
        let mut marked = vec![false; 64];
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        for _ in 0..100 {
            let size = lattice.wolff_move(2.3, &mut rng, &mut marked);
            assert!((1..=64).contains(&size));
            assert_eq!(lattice.energy, lattice.recompute_energy());
        }
    }
}
