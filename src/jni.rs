use std::sync::OnceLock;

use cust::prelude::*;
use jni::{
    JNIEnv,
    objects::JObject,
    sys::{jfloat, jint},
};
use rayon::prelude::*;

static CUDA_CTX: OnceLock<Context> = OnceLock::new();

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_erayt_cuda_GpuInterface_quickInit(_env: JNIEnv, _: JObject) {
    CUDA_CTX.get_or_init(|| {
        cust::init(CudaFlags::empty()).expect("Failed to initialize CUDA");
        let device = Device::get_device(0).expect("No CUDA device found");
        Context::new(device).expect("Failed to create CUDA context")
    });
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_erayt_cuda_GpuInterface_monteCarlo(
    _env: JNIEnv,
    _: JObject,
    num_paths: jint,
    s0: jfloat,
    k: jfloat,
    r: jfloat,
    sigma: jfloat,
    t: jfloat,
) -> jfloat {
    let results = match monte_carlo(num_paths as usize, s0, k, r, sigma, t) {
        Ok(results) => results,
        Err(err) => panic!("Failed to run Monte Carlo simulation: {}", err),
    };

    results as jfloat
}

fn monte_carlo(
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

    let mut results = vec![0.0f32; num_paths];
    let d_results = results.as_slice().as_dbuf()?;

    let ptx = include_str!(concat!(env!("OUT_DIR"), "/monte_carlo.ptx"));
    let module = Module::from_ptx(ptx, &[])?;

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
