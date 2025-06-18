use std::path::PathBuf;
use std::process::Command;
use std::{env, fs};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let cuda_files = fs::read_dir("cuda").unwrap();
    for file in cuda_files {
        let path = file.unwrap().path();
        if path.extension().unwrap() == "cu" {
            let ptx_out = out_dir.join(path.with_extension("ptx").file_name().unwrap());
            Command::new("nvcc")
                .args(["-ptx", path.to_str().unwrap(), "-o"])
                .arg(ptx_out)
                .status()
                .unwrap();
        }
    }

    println!("cargo:rerun-if-changed=kernel.cu");
}
