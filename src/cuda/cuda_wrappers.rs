use crate::CUDA_CTX;
use cust::prelude::*;
use rayon::prelude::*;

pub fn monte_carlo(
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

    let kernel = module.get_function("monte_carlo")?;

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
