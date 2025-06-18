use crate::cuda::cuda_wrappers::monte_carlo;

pub fn run_from(args: &[f32]) -> anyhow::Result<Vec<f32>> {
    let (num_paths, s0, k, r, sigma, t) = (
        args[0] as usize,
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
    );
    let result = monte_carlo(num_paths, s0, k, r, sigma, t).unwrap_or(-1.0);
    Ok(vec![result])
}
