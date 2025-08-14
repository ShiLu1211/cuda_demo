use crate::cuda_init;
use cust::prelude::*;

pub fn run_from_f64(args: &[f64]) -> anyhow::Result<Vec<f64>> {
    let args_f64 = &args[0..5];
    let num_paths = args[5] as usize;
    let num_steps = args[6] as usize;

    // 3. 后面的数据 reshape 成 paths × steps
    let rand_matrix = &args[7..];
    let result =
        simulate_prices_gpu(rand_matrix, num_paths, num_steps, args_f64).unwrap_or_default();
    Ok(result)
}

fn simulate_prices_gpu(
    rand_matrix: &[f64],
    num_paths: usize,
    num_steps: usize,
    args: &[f64],
) -> anyhow::Result<Vec<f64>> {
    cuda_init();

    let (s0, r, q, v, t) = (args[0], args[1], args[2], args[3], args[4]);

    let size = num_paths * num_steps;
    let d_price = DeviceBuffer::<f64>::zeroed(size)?;
    let d_random_matrix = rand_matrix.as_dbuf()?;

    // 加载模块、调度
    let ptx = include_str!(concat!(env!("OUT_DIR"), "/rand_matrix.ptx"));
    let m = Module::from_ptx(ptx, &[])?;
    let s = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let f = m.get_function("simulate_prices_kernel")?;

    let (_, block_size) = f.suggested_launch_configuration(0, 0.into())?;
    let grid_size = (num_paths as u32).div_ceil(block_size);

    unsafe {
        launch!(f<<<grid_size, block_size, 0, s>>>(
            d_random_matrix.as_device_ptr(),
            d_price.as_device_ptr(),
            num_paths,
            num_steps,
            s0,
            r,
            q,
            v,
            t,
        ))?;
    }
    s.synchronize()?;

    let mut h_price = vec![0f64; size];
    d_price.copy_to(&mut h_price)?;

    Ok(h_price)
}
