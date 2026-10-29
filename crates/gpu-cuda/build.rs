use std::path::PathBuf;
use std::process::Command;

// The public source generator is shared with the library.
#[allow(unreachable_pub)]
#[path = "src/source.rs"]
mod source;

// Native code for Turing through Blackwell, plus PTX for additional architectures.
const DEFAULT_ARCHITECTURES: &str = "75,80,86,89,90,100,120";

fn main() {
    println!("cargo:rerun-if-changed=src/source.rs");
    println!("cargo:rerun-if-changed=src/compat.cuh");
    println!("cargo:rerun-if-changed=../gpu-metal/src/kernels.metal");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=BLANKET_CUDA_ARCHS");
    if std::env::var_os("CARGO_FEATURE_PRECOMPILED").is_none() {
        return;
    }
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let input = out.join("blanket.cu");
    let output = out.join("blanket.fatbin");
    std::fs::write(&input, source::shader_source()).expect("cannot write generated CUDA source");
    let architectures = std::env::var("BLANKET_CUDA_ARCHS").unwrap_or_else(|_| DEFAULT_ARCHITECTURES.into());
    let mut architectures: Vec<u16> = architectures
        .split(',')
        .map(|value| value.trim().parse().expect("BLANKET_CUDA_ARCHS must contain comma-separated numeric compute capabilities"))
        .collect();
    architectures.sort_unstable();
    architectures.dedup();
    let nvcc = std::env::var_os("NVCC").unwrap_or_else(|| "nvcc".into());
    let mut command = Command::new(nvcc);
    command.args(["--fatbin", "--std=c++17", "--fmad=false"]);
    for architecture in &architectures {
        command.arg(format!("--generate-code=arch=compute_{architecture},code=sm_{architecture}"));
    }
    let baseline = architectures.first().expect("at least one CUDA architecture is required");
    command.arg(format!("--generate-code=arch=compute_{baseline},code=compute_{baseline}"));
    let result = command
        .arg(&input)
        .arg("--output-file")
        .arg(&output)
        .output()
        .expect("CUDA builds require CUDA 13 nvcc on PATH, or NVCC set to its executable; CPU/Metal builds do not require CUDA");
    assert!(
        result.status.success(),
        "CUDA kernel compilation failed:\n{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(std::fs::metadata(output).is_ok_and(|file| file.len() > 0), "nvcc did not produce a CUDA fatbinary");
}
