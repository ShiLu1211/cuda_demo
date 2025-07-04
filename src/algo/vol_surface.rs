use crate::cuda::vol_surface_gpu;

pub fn run_from(args: &[f32]) -> anyhow::Result<Vec<f32>> {
    let n = args[0] as usize;
    assert_eq!(args.len(), 1 + 5 * n);

    let s = &args[1 .. 1 + n];
    let k = &args[1 + n .. 1 + 2 * n];
    let t = &args[1 + 2 * n .. 1 + 3 * n];
    let r = &args[1 + 3 * n .. 1 + 4 * n];
    let p = &args[1 + 4 * n .. 1 + 5 * n];

    let v = vol_surface_gpu(s, k, t, r, p)?;
    Ok(v)
}