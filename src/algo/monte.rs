pub fn run_from(args: &[f32]) -> anyhow::Result<Vec<f32>> {
    let (num_paths, s0, k, r, sigma, t) = (
        args[0] as usize,
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
    );
    let result = crate::cuda::monte_carlo_gpu(num_paths, s0, k, r, sigma, t).unwrap_or(-1.0);
    Ok(vec![result])
}
