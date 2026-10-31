use std::process::Command;

#[test]
fn device_helpers_compile_for_cuda() {
    let output = Command::new("clang++")
        .args(["-std=c++17", "-x", "cuda", "--cuda-device-only", "--cuda-gpu-arch=sm_75", "-nocudainc", "-nocudalib", "-fsyntax-only", "-include"])
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cuda_syntax.cuh"))
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/src/kernels.cu"))
        .output();
    match output {
        Ok(output) => assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => eprintln!("clang++ unavailable; CUDA device syntax check skipped"),
        Err(error) => panic!("cannot run CUDA device syntax check: {error}"),
    }
}

#[test]
#[cfg(unix)]
fn portable_kernels_compile_and_run_on_host() {
    let executable = std::env::temp_dir().join(format!("blanket-cuda-host-{}", std::process::id()));
    let output = Command::new(std::env::var("CXX").unwrap_or_else(|_| "c++".into()))
        .arg("-std=c++17")
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/kernels_host.cpp"))
        .arg("-o")
        .arg(&executable)
        .output()
        .expect("C++ compiler required for CUDA portability checks");
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let status = Command::new(&executable).status().unwrap();
    std::fs::remove_file(executable).unwrap();
    assert!(status.success());
}
