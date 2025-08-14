use crate::cuda_init;
use cust::prelude::*;
use rayon::prelude::*;

pub fn run_from(args: &[f32]) -> anyhow::Result<Vec<f32>> {
    let (num_paths, s0, k, r, sigma, t) = (
        args[0] as usize,
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
    );
    let result = monte_carlo_gpu(num_paths, s0, k, r, sigma, t).unwrap_or(-1.0);
    Ok(vec![result])
}

pub fn monte_carlo_gpu(
    num_paths: usize,
    s0: f32,
    k: f32,
    r: f32,
    sigma: f32,
    t: f32,
) -> anyhow::Result<f32> {
    cuda_init();

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
