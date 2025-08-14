use crate::cuda_init;
use cust::prelude::*;
use rayon::prelude::*;

pub fn run_from(args: &[f32]) -> anyhow::Result<Vec<f32>> {
    let n = args[0] as usize;
    let returns = &args[1..1 + n];
    let confidence = args[1 + n];
    let holding = args[2 + n];
    
    let v = var_gpu(returns, confidence, holding)?;
    Ok(vec![v])
}

fn var_gpu(returns: &[f32], confidence: f32, holding: f32) -> anyhow::Result<f32> {
    cuda_init();

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
