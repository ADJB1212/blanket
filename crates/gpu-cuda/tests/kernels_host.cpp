#include <initializer_list>

#define __device__
#define __global__
struct Dim {
    unsigned x;
};
Dim blockIdx{0}, blockDim{1}, threadIdx{0};

#include "../src/kernels.cu"

int main() {
    uchar source[4] = {0, 1, 127, 255}, output[4] = {};
    uint parameters[8] = {4, 1, 1, 4, 1, 11, 0, 0};
    float controls[1] = {0};
    for (uint i = 0; i < 4; i++) {
        blockIdx.x = i;
        image_u8(source, output, source, parameters, controls);
        if (output[i] != 255 - source[i])
            return 1;
    }
    float floating[4] = {};
    parameters[5] = 14;
    parameters[6] = 4;
    for (uint i = 0; i < 4; i++) {
        blockIdx.x = i;
        convert_u8_f32(source, floating, source, parameters, controls);
        if (floating[i] != float(source[i]))
            return 2;
    }
    uint tile[11] = {4, 1, 1, 4, 1, 1, 0, 2, 0, 2, 1};
    float tile_output[4] = {-1, -1, -1, -1};
    for (uint i = 0; i < 3; i++) {
        blockIdx.x = i;
        tile_u8(source, output, tile);
        tile_f32(floating, tile_output, tile);
    }
    if (output[0] != 255 || output[1] != 254 || output[2] != 1 || output[3] != 127)
        return 3;
    if (tile_output[0] != -1 || tile_output[1] != -1 || tile_output[2] != 1 || tile_output[3] != 127)
        return 4;
    for (uint channels = 1; channels <= 4; channels++) {
        for (uint width : {1u, 15u, 16u, 17u, 31u, 32u, 33u}) {
            uint height = 17, count = width * height * channels;
            uchar input[2246], result[2246];
            for (uint j = 0; j < count + 2; j++)
                input[j] = uchar(j * 37 + j / 11);
            for (uint axis = 0; axis < 2; axis++) {
                for (uint radius : {32u, 128u, 10000000u}) {
                    uint scale = BOX_SCALE / (2 * radius + 2);
                    float weights[2] = {float(scale), float((BOX_SCALE - (2 * radius + 1) * scale) / 2)};
                    uint p[8] = {width, height, channels, width, height, 18, radius, axis};
                    for (uint j = 0; j < count + 2; j++)
                        result[j] = 173;
                    for (uint j = 0; j <= count; j++) {
                        blockIdx.x = j;
                        image_u8(input + 1, result + 1, input + 1, p, weights);
                    }
                    if (result[0] != 173 || result[count + 1] != 173)
                        return 5;
                    for (uint j = 0; j < count; j++) {
                        uint band = j % channels, x = (j / channels) % width, y = j / (channels * width);
                        int64 pos = axis == 0 ? x : y, last = (axis == 0 ? width : height) - 1;
                        uint base = axis == 0 ? y * width * channels + band : x * channels + band;
                        uint stride = axis == 0 ? channels : width * channels, sum = 0;
                        for (int64 t = max(0ll, pos - radius); t <= min(last, pos + radius); t++)
                            sum += input[1 + base + uint(t) * stride];
                        sum += input[1 + base] * uint(max(0ll, int64(radius) - pos));
                        sum += input[1 + base + uint(last) * stride] * uint(max(0ll, pos + radius - last));
                        uint left = base + uint(clamp(pos - radius - 1, 0ll, last)) * stride;
                        uint right = base + uint(clamp(pos + radius + 1, 0ll, last)) * stride;
                        uint fixed = sum * uint(weights[0]) + (uint(input[1 + left]) + uint(input[1 + right])) * uint(weights[1]) + BOX_ROUND;
                        if (result[1 + j] != uchar(fixed >> BOX_BITS))
                            return 6;
                    }
                }
            }
        }
    }
    return 0;
}
