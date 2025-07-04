use crate::CUDA_CTX;
use cust::prelude::*;
use rayon::prelude::*;

pub fn monte_carlo_gpu(
    num_paths: usize,
    s0: f32,
    k: f32,
    r: f32,
    sigma: f32,
    t: f32,
) -> anyhow::Result<f32> {
    if CUDA_CTX.get().is_none() {
        return Err(anyhow::anyhow!("CUDA context not initialized"));
    }

    let ptx = include_str!(concat!(env!("OUT_DIR"), "/monte.ptx"));
    let module = Module::from_ptx(ptx, &[])?;

    let mut results = vec![0.0f32; num_paths];
    let d_results = results.as_slice().as_dbuf()?;

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let kernel = module.get_function("monte_carlo_kernel")?;

    let (_, block_size) = kernel.suggested_launch_configuration(0, 0.into())?;

    let grid_size = (num_paths as u32).div_ceil(block_size);

    unsafe {
        launch!(
            // slices are passed as two parameters, the pointer and the length.
            kernel<<<grid_size, block_size, 0, stream>>>(
                d_results.as_device_ptr(),
                num_paths,
                s0,
                k,
                r,
                sigma,
                t
            )
        )?;
    }

    stream.synchronize()?;

    d_results.copy_to(&mut results)?;

    let mean_payoff: f32 = results.par_iter().sum::<f32>() / num_paths as f32;
    let discounted = (-r * t).exp() * mean_payoff;

    Ok(discounted)
}

pub fn var_gpu(returns: &[f32], confidence: f32, holding: f32) -> anyhow::Result<f32> {
    if CUDA_CTX.get().is_none() {
        return Err(anyhow::anyhow!("CUDA context not initialized"));
    }

    let n = returns.len();
    let mut losses = vec![0.0f32; n];
    let d_returns = returns.as_dbuf()?;
    let d_losses = DeviceBuffer::from_slice(&losses)?;

    // 加载模块、调度
    let ptx = include_str!(concat!(env!("OUT_DIR"), "/var.ptx"));
    let m = Module::from_ptx(ptx, &[])?;
    let s = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let f = m.get_function("var_kernel")?;

    let (_, block_size) = f.suggested_launch_configuration(0, 0.into())?;

    let grid_size = (n as u32).div_ceil(block_size);

    unsafe {
        launch!(f<<<grid_size, block_size, 0, s>>>(
            d_returns.as_device_ptr(),
            n as i32,
            confidence,
            holding,
            d_losses.as_device_ptr()
        ))?;
    }
    s.synchronize()?;

    d_losses.copy_to(&mut losses)?;

    losses.par_sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((1.0 - confidence) * n as f32) as usize;
    Ok(losses[idx])
}

pub fn vol_surface_gpu(
    s: &[f32],
    k: &[f32],
    t: &[f32],
    r: &[f32],
    p: &[f32],
) -> anyhow::Result<Vec<f32>> {
    if CUDA_CTX.get().is_none() {
        return Err(anyhow::anyhow!("CUDA context not initialized"));
    }

    let n = s.len();
    let mut results = vec![0.0f32; n];
    let d_s = s.as_dbuf()?;
    let d_k = k.as_dbuf()?;
    let d_t = t.as_dbuf()?;
    let d_r = r.as_dbuf()?;
    let d_p = p.as_dbuf()?;
    let d_results = DeviceBuffer::from_slice(&results)?;

    // 加载模块、调度
    let ptx = include_str!(concat!(env!("OUT_DIR"), "/vol_surface.ptx"));
    let m = Module::from_ptx(ptx, &[])?;
    let s = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let f = m.get_function("vol_surface_kernel")?;

    let (_, block_size) = f.suggested_launch_configuration(0, 0.into())?;

    let grid_size = (n as u32).div_ceil(block_size);

    unsafe {
        launch!(f<<<grid_size, block_size, 0, s>>>(
            d_s.as_device_ptr(),
            d_k.as_device_ptr(),
            d_t.as_device_ptr(),
            d_r.as_device_ptr(),
            d_p.as_device_ptr(),
            d_results.as_device_ptr(),
            n as i32,
        ))?;
    }
    s.synchronize()?;

    d_results.copy_to(&mut results)?;

    Ok(results)
}
