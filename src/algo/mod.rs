mod monte;
mod var;
mod vol_surface;
mod rand_matrix;
mod black_scholes;

pub fn dispatch(algo_id: i32, args: &[f32]) -> anyhow::Result<Vec<f32>> {
    match algo_id {
        1 => monte::run_from(args),
        2 => var::run_from(args),
        3 => vol_surface::run_from(args),
        _ => Err(anyhow::anyhow!("Invalid algo_id")),
    }
}

pub fn dispatch_f64(algo_id: i32, args: &[f64]) -> anyhow::Result<Vec<f64>> {
    match algo_id {
        4 => rand_matrix::run_from_f64(args),
        5 => black_scholes::run_from_f64(args),
        _ => Err(anyhow::anyhow!("Invalid algo_id")),
    }
}

pub fn dispatch_cpu(algo_id: i32, args: &[f64]) -> anyhow::Result<Vec<f64>> {
    match algo_id {
        4 => rand_matrix::run_cpu(args),
        _ => Err(anyhow::anyhow!("Invalid algo_id")),
    }
}
