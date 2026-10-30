#include <initializer_list>
#include <vector>

#define __device__
#define __global__
struct Dim {
    unsigned x;
};
Dim blockIdx{0}, blockDim{1}, threadIdx{0};

#include "../src/kernels.cu"

template <typename T> bool statistics_match_serial() {
    for (uint channels = 1; channels <= 4; channels++) {
        for (uint pixels : {1u, 37u, 1031u}) {
            std::vector<T> source(pixels * channels + 2);
            for (uint i = 0; i < source.size(); i++)
                source[i] = T((i * 37 + i / 11) % 16);
            for (uint kind : {0u, 1u}) {
                if (kind == 1 && !is_same_v<T, uchar>)
                    continue;
                uint fields = kind == 1 ? HISTOGRAM_BINS : STATISTICS_FIELDS;
                uint p[8] = {pixels, 1, channels, fields, channels, 29, kind, 0};
                float controls[1] = {0};
                std::vector<float> expected(fields * channels);
                for (uint i = 0; i < channels; i++)
                    process(source.data() + 1, reinterpret_cast<T *>(expected.data()), source.data() + 1, p, controls, i);
                for (uint blocks : {1u, 2u, 7u}) {
                    p[7] = blocks;
                    std::vector<float> partials(fields * channels * blocks + 2, -123);
                    std::vector<float> output(fields * channels + 2, -123);
                    for (uint i = 0; i <= channels * blocks; i++)
                        process(source.data() + 1, reinterpret_cast<T *>(partials.data() + 1), source.data() + 1, p, controls, i);
                    for (uint i = 0; i <= fields * channels; i++) {
                        blockIdx.x = i;
                        finish_statistics(partials.data() + 1, output.data() + 1, p);
                    }
                    if (partials.front() != -123 || partials.back() != -123 || output.front() != -123 || output.back() != -123)
                        return false;
                    for (uint i = 0; i < expected.size(); i++) {
                        if (kind == 1 || i % fields == fields - 1) {
                            if (as_type<uint>(output[i + 1]) != as_type<uint>(expected[i]))
                                return false;
                        } else if (abs(output[i + 1] - expected[i]) > max(0.001f, abs(expected[i]) * 1e-5f))
                            return false;
                    }
                }
            }
        }
    }
    return true;
}

int main() {
    if (!statistics_match_serial<uchar>() || !statistics_match_serial<float>())
        return 7;
    {
        float samples[3] = {1, NAN, INFINITY}, partials[21], output[7];
        uint p[8] = {3, 1, 1, 7, 1, 29, 0, 3};
        float controls[1] = {0};
        for (uint i = 0; i < 3; i++)
            process(samples, partials, samples, p, controls, i);
        blockIdx.x = 0;
        finish_statistics(partials, output, p);
        if (!isnan(output[0]) || !isnan(output[1]) || !isnan(output[2]) || as_type<uint>(output[6]) != 3)
            return 8;
    }
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
