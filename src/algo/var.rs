pub fn run_from(args: &[f32]) -> anyhow::Result<Vec<f32>> {
    let n = args[0] as usize;
    let returns = &args[1..1 + n];
    let confidence = args[1 + n];
    let holding = args[2 + n];
    
    let v = crate::cuda::var_gpu(returns, confidence, holding)?;
    Ok(vec![v])
}
