use crate::cuda_init;
use cust::prelude::*;
use rand::{SeedableRng, rngs::SmallRng};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

pub fn run_from_f64(args: &[f64]) -> anyhow::Result<Vec<f64>> {
    let (num_paths, num_steps, seed) = (args[0] as usize, args[1] as usize, args[2] as usize);
    let result = generate_random_matrix_gpu(num_paths, num_steps, seed).unwrap_or_default();
    Ok(result)
}

pub fn run_cpu(args: &[f64]) -> anyhow::Result<Vec<f64>> {
    let (num_paths, num_steps, _seed) = (args[0] as usize, args[1] as usize, args[2] as usize);
    let result = generate_random_matrix_parallel(num_paths, num_steps).unwrap_or_default();
    Ok(result)
}

fn generate_random_matrix_gpu(
    num_paths: usize,
    num_steps: usize,
    seed: usize,
) -> anyhow::Result<Vec<f64>> {
    cuda_init();

    let size = num_paths * num_steps;
    let d_random_matrix = DeviceBuffer::<f64>::zeroed(size)?;

    // 加载模块、调度
    let ptx = include_str!(concat!(env!("OUT_DIR"), "/rand_matrix.ptx"));
    let m = Module::from_ptx(ptx, &[])?;
    let s = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let f = m.get_function("generate_random_matrix_kernel")?;

    let (_, block_size) = f.suggested_launch_configuration(0, 0.into())?;
    let grid_size = (num_paths as u32).div_ceil(block_size);

    unsafe {
        launch!(f<<<grid_size, block_size, 0, s>>>(
            d_random_matrix.as_device_ptr(),
            num_paths,
            num_steps,
            seed,
        ))?;
    }
    s.synchronize()?;

    let mut h_random_matrix = vec![0f64; size];
    d_random_matrix.copy_to(&mut h_random_matrix)?;

    Ok(h_random_matrix)
}

fn generate_random_matrix_parallel(rows: usize, cols: usize) -> anyhow::Result<Vec<f64>> {
    let normal = Normal::new(0.0, 1.0)?;

    // 用并行 map 构建每条路径的数据
    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map_iter(|_| {
            // 每个线程用 SmallRng 初始化
            let mut rng = SmallRng::from_rng(&mut rand::rng());
            (0..cols)
                .map(|_| normal.sample(&mut rng))
                .collect::<Vec<f64>>()
        })
        .collect();

    Ok(data)
}
