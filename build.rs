use std::process::Command;
use std::env;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let status = Command::new("nvcc")
        .args(["-ptx", "cuda/computex.cu", "-o"])
        .arg(out_dir.join("computex.ptx"))
        .status()
        .expect("Failed to compile CUDA kernel");

    if !status.success() {
        panic!("nvcc failed");
    }

    println!("cargo:rerun-if-changed=kernel.cu");
}
