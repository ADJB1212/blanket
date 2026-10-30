//! CUDA source lowering for the shared pixel-operation contracts.

const PRELUDE: &str = include_str!("compat.cuh");

/// Lower address spaces, vector helpers, and entry points for NVRTC.
pub fn shader_source() -> String {
    let mut source = format!("{}\n{}", blanket_gpu_metal::SHADERS, blanket_gpu_metal::TILE_SHADERS)
        .replace("#include <metal_stdlib>", "")
        .replace("using namespace metal;", "")
        .replace("[[buffer(4)]],", "[[buffer(4)]]");
    let grid = "uint i [[thread_position_in_grid]]) {";
    while let Some(index) = source.find(grid) {
        let prefix = source[..index].trim_end();
        let start = prefix.strip_suffix(',').map_or(index, str::len);
        source.replace_range(start..index + grid.len(), ") { uint i = blockIdx.x * blockDim.x + threadIdx.x;");
    }
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
        assert_eq!(source.matches("uint i = blockIdx.x").count(), 7);
        assert!(source.contains("extern \"C\" __global__ void tile_u8"));
        assert!(source.contains("extern \"C\" __global__ void tile_f32"));
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
            "#include <initializer_list>\n#define __device__\n#define __global__\nstruct Dim {{ unsigned x; }}; Dim blockIdx{{0}}, blockDim{{1}}, threadIdx{{0}};\n{}\n{}",
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
    uint tile[11] = {4, 1, 1, 4, 1, 1, 0, 2, 0, 2, 1};
    float tile_output[4] = {-1, -1, -1, -1};
    for (uint i = 0; i < 3; i++) {
        blockIdx.x = i;
        tile_u8(source, output, tile);
        tile_f32(floating, tile_output, tile);
    }
    if (output[0] != 255 || output[1] != 254 || output[2] != 1 || output[3] != 127) return 3;
    if (tile_output[0] != -1 || tile_output[1] != -1 || tile_output[2] != 1 || tile_output[3] != 127) return 4;
    for (uint channels = 1; channels <= 4; channels++) {
        for (uint width : {1u, 15u, 16u, 17u, 31u, 32u, 33u}) {
            uint height = 17, count = width * height * channels;
            uchar input[2246], result[2246];
            for (uint j = 0; j < count + 2; j++) input[j] = uchar(j * 37 + j / 11);
            for (uint axis = 0; axis < 2; axis++) {
                for (uint radius : {32u, 128u, 10000000u}) {
                    uint scale = BOX_SCALE / (2 * radius + 2);
                    float weights[2] = {float(scale), float((BOX_SCALE - (2 * radius + 1) * scale) / 2)};
                    uint p[8] = {width, height, channels, width, height, 18, radius, axis};
                    for (uint j = 0; j < count + 2; j++) result[j] = 173;
                    for (uint j = 0; j <= count; j++) {
                        blockIdx.x = j;
                        image_u8(input + 1, result + 1, input + 1, p, weights);
                    }
                    if (result[0] != 173 || result[count + 1] != 173) return 5;
                    for (uint j = 0; j < count; j++) {
                        uint band = j % channels, x = (j / channels) % width, y = j / (channels * width);
                        metal_long pos = axis == 0 ? x : y, last = (axis == 0 ? width : height) - 1;
                        uint base = axis == 0 ? y * width * channels + band : x * channels + band;
                        uint stride = axis == 0 ? channels : width * channels, sum = 0;
                        for (metal_long t = max(0ll, pos - radius); t <= min(last, pos + radius); t++)
                            sum += input[1 + base + uint(t) * stride];
                        sum += input[1 + base] * uint(max(0ll, metal_long(radius) - pos));
                        sum += input[1 + base + uint(last) * stride] * uint(max(0ll, pos + radius - last));
                        uint left = base + uint(clamp(pos - radius - 1, 0ll, last)) * stride;
                        uint right = base + uint(clamp(pos + radius + 1, 0ll, last)) * stride;
                        uint fixed = sum * uint(weights[0]) + (uint(input[1 + left]) + uint(input[1 + right])) * uint(weights[1]) + BOX_ROUND;
                        if (result[1 + j] != uchar(fixed >> BOX_BITS)) return 6;
                    }
                }
            }
        }
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
