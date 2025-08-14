use crate::cuda_init;
use cust::prelude::*;

pub fn run_from(args: &[f32]) -> anyhow::Result<Vec<f32>> {
    let n = args[0] as usize;
    assert_eq!(args.len(), 1 + 5 * n);

    let s = &args[1..1 + n];
    let k = &args[1 + n..1 + 2 * n];
    let t = &args[1 + 2 * n..1 + 3 * n];
    let r = &args[1 + 3 * n..1 + 4 * n];
    let p = &args[1 + 4 * n..1 + 5 * n];

    let v = vol_surface_gpu(s, k, t, r, p)?;
    Ok(v)
}

fn vol_surface_gpu(
    s: &[f32],
    k: &[f32],
    t: &[f32],
    r: &[f32],
    p: &[f32],
) -> anyhow::Result<Vec<f32>> {
    cuda_init();

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
