mod monte;
mod var;

pub fn dispatch(algo_id: i32, args: &[f32]) -> anyhow::Result<Vec<f32>> {
    match algo_id {
        1 => monte::run_from(args),
        2 => var::run_from(args),
        _ => Err(anyhow::anyhow!("Invalid algo_id")),
    }
}
