//! CUDA compilation of the shared pixel-operation contracts.

const PRELUDE: &str = include_str!("compat.cuh");

/// Lower address spaces, vector helpers, and entry points for NVRTC.
pub fn shader_source() -> String {
    let mut source = blanket_gpu_metal::SHADERS
        .replace("#include <metal_stdlib>", "")
        .replace("using namespace metal;", "")
        .replace("device const float* v [[buffer(4)]],", "device const float* v [[buffer(4)]]")
        .replace("uint i [[thread_position_in_grid]]) {", ") { uint i = blockIdx.x * blockDim.x + threadIdx.x;");
    for slot in 0..5 {
        source = source.replace(&format!("[[buffer({slot})]]"), "");
    }
    let mut cuda = String::with_capacity(PRELUDE.len() + source.len());
    cuda.push_str(PRELUDE);
    for line in source.lines() {
        let line = if line.starts_with("template ") {
            line.replacen("> ", "> __device__ ", 1)
        } else if line.starts_with("float color_transfer(") || line.starts_with("int lut_mix(") || line.starts_with("float warp_cubic(") {
            format!("__device__ {line}")
        } else {
            line.to_owned()
        };
        let mut rest = line.as_str();
        while !rest.is_empty() {
            let count = rest.bytes().take_while(|c| c.is_ascii_alphanumeric() || *c == b'_').count();
            if count == 0 {
                let character = rest.chars().next().unwrap();
                cuda.push(character);
                rest = &rest[character.len_utf8()..];
            } else {
                let token = &rest[..count];
                cuda.push_str(match token {
                    "device" => "",
                    "constant" => "const",
                    "kernel" => "extern \"C\" __global__",
                    "long" => "metal_long",
                    "ulong" => "metal_ulong",
                    "float3" => "vec3<float>",
                    "uint3" => "vec3<uint>",
                    "int3" => "vec3<int>",
                    "float4" => "vec4<float>",
                    "floor" => "floorf",
                    "round" => "roundf",
                    "trunc" => "truncf",
                    "pow" => "powf",
                    "fma" => "fmaf",
                    "copysign" => "copysignf",
                    _ => token,
                });
                rest = &rest[count..];
            }
        }
        cuda.push('\n');
    }
    cuda
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowers_all_entry_points_and_address_spaces() {
        let source = shader_source();
        assert!(!source.contains("[[buffer("));
        assert!(!source.contains("thread_position_in_grid"));
        assert!(!source.contains("metal_stdlib"));
        assert_eq!(source.matches("uint i = blockIdx.x").count(), 4);
        assert!(source.contains("extern \"C\" __global__ void image_u8"));
        assert!(source.contains("extern \"C\" __global__ void image_f32"));
        assert!(source.contains("CONVERSION_KERNEL(convert_f32_u8"));
        assert!(source.contains("REDUCTION_KERNEL(reduce_u8"));
    }

    #[test]
    #[cfg(unix)]
    fn portable_kernels_compile_and_run_on_host() {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let executable = std::env::temp_dir().join(format!("blanket-cuda-host-{}", std::process::id()));
        let mut compiler = Command::new(std::env::var("CXX").unwrap_or_else(|_| "c++".into()))
            .args(["-std=c++17", "-x", "c++", "-", "-o"])
            .arg(&executable)
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("C++ compiler required for CUDA portability checks");
        let source = format!(
            "#define __device__\n#define __global__\nstruct Dim {{ unsigned x; }}; Dim blockIdx{{0}}, blockDim{{1}}, threadIdx{{0}};\n{}\n{}",
            shader_source(),
            r"
int main() {
    uchar source[4] = {0, 1, 127, 255}, output[4] = {};
    uint parameters[8] = {4, 1, 1, 4, 1, 11, 0, 0};
    float controls[1] = {0};
    for (uint i = 0; i < 4; i++) {
        blockIdx.x = i;
        image_u8(source, output, source, parameters, controls);
        if (output[i] != 255 - source[i]) return 1;
    }
    float floating[4] = {};
    parameters[5] = 14; parameters[6] = 4;
    for (uint i = 0; i < 4; i++) {
        blockIdx.x = i;
        convert_u8_f32(source, floating, source, parameters, controls);
        if (floating[i] != float(source[i])) return 2;
    }
    return 0;
}
"
        );
        compiler.stdin.take().unwrap().write_all(source.as_bytes()).unwrap();
        let output = compiler.wait_with_output().unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let status = Command::new(&executable).status().unwrap();
        std::fs::remove_file(executable).unwrap();
        assert!(status.success());
    }
}
