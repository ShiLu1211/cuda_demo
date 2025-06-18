use cust::prelude::*;
use jni::{
    JNIEnv,
    objects::{JFloatArray, JObject},
    sys::jint,
};
use std::sync::OnceLock;

pub mod algo;
pub mod cuda;

pub static CUDA_CTX: OnceLock<Context> = OnceLock::new();

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_erayt_cuda_GpuInterface_cudaInit(_env: JNIEnv, _: JObject) {
    CUDA_CTX.get_or_init(|| {
        cust::init(CudaFlags::empty()).expect("Failed to initialize CUDA");
        let device = Device::get_device(0).expect("No CUDA device found");
        Context::new(device).expect("Failed to create CUDA context")
    });
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_erayt_cuda_GpuInterface_run<'a>(
    env: JNIEnv<'a>,
    _: JObject<'a>,
    algo_id: jint,
    jargs: JFloatArray<'a>,
) -> JFloatArray<'a> {
    let length = env
        .get_array_length(&jargs)
        .expect("Invalid input array length");
    let mut args = vec![0.0f32; length as usize];
    env.get_float_array_region(&jargs, 0, &mut args)
        .expect("Failed to get float array from Java");

    // println!("Received args (algo_id = {}): {:?}", algo_id, args.len());

    let result = match algo::dispatch(algo_id, &args) {
        Ok(result) => result,
        Err(e) => panic!("Algorithm dispatch failed: {}", e),
    };

    let out = env
        .new_float_array(result.len() as i32)
        .expect("Failed to create output array");
    env.set_float_array_region(&out, 0, &result)
        .expect("Failed to set output array contents");

    out
}
