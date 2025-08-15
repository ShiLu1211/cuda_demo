use crate::cuda_init;
use cust::prelude::*;
use rand::{SeedableRng, rngs::SmallRng};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

pub fn run_from_f64(args: &[f64]) -> anyhow::Result<Vec<f64>> {
    let args_f64 = &args[0..5];
    let num_paths = args[5] as usize;
    let num_steps = args[6] as usize;
    let seed = args[7] as usize;

    let result = match rand_and_simulate_gpu(seed, num_paths, num_steps, args_f64) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("{}", e);
            vec![]
        }
    };
    Ok(result)
}

pub fn run_cpu(args: &[f64]) -> anyhow::Result<Vec<f64>> {
    let (s0, r, q, v, t) = (args[0], args[1], args[2], args[3], args[4]);
    let num_paths = args[5] as usize;
    let num_steps = args[6] as usize;

    let result = match rand_and_simulate_parallel(num_paths, num_steps, s0, r, q, v, t) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("{}", e);
            vec![]
        }
    };
    Ok(result)
}

fn rand_and_simulate_gpu(
    seed: usize,
    num_paths: usize,
    num_steps: usize,
    args: &[f64],
) -> anyhow::Result<Vec<f64>> {
    cuda_init();

    let (s0, r, q, v, t) = (args[0], args[1], args[2], args[3], args[4]);

    let size = num_paths * num_steps;
    let d_price = DeviceBuffer::<f64>::zeroed(size)?;

    // 加载模块、调度
    let ptx = include_str!(concat!(env!("OUT_DIR"), "/rand_matrix.ptx"));
    let m = Module::from_ptx(ptx, &[])?;
    let s = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let f = m.get_function("random_and_simulate_kernel")?;

    let (_, block_size) = f.suggested_launch_configuration(0, 0.into())?;
    let grid_size = (num_paths as u32).div_ceil(block_size);

    unsafe {
        launch!(f<<<grid_size, block_size, 0, s>>>(
            d_price.as_device_ptr(),
            num_paths,
            num_steps,
            seed,
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

fn rand_and_simulate_parallel(
    num_paths: usize,
    num_steps: usize,
    s0: f64,
    r: f64,
    q: f64,
    v: f64,
    t: f64,
) -> anyhow::Result<Vec<f64>> {
    let dt = t / num_steps as f64;
    let drift = (r - q - 0.5 * v * v) * dt;
    let vol_dt_sqrt = v * dt.sqrt();

    let normal = Normal::<f64>::new(0.0, 1.0)?;

    // 扁平 price_matrix 长度 = num_paths * num_steps
    let prices: Vec<f64> = (0..num_paths)
        .into_par_iter()
        .flat_map_iter(|_| {
            let mut rng = SmallRng::from_rng(&mut rand::rng()); // 每线程独立RNG
            let mut s = s0;
            let mut row = Vec::with_capacity(num_steps);
            for _ in 0..num_steps {
                let z = normal.sample(&mut rng);
                s *= f64::exp(drift + vol_dt_sqrt * z);
                row.push(s);
            }
            row
        })
        .collect();

    Ok(prices)
}
