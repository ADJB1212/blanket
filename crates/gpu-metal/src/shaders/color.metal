// Shared luminance, transfer, LUT, and image-mode conversion routines.

// Compute luminance with byte fixed-point rounding or float color weights.
template <typename T> float luminance(device const T *src, uint base, uint channels) {
    if (channels < 3)
        return float(src[base]);
    if (is_same_v<T, uchar>) {
        return float((uint(src[base]) * LUMA_RED + uint(src[base + 1]) * LUMA_GREEN + uint(src[base + 2]) * LUMA_BLUE + (1u << (LUMA_BITS - 1))) >>
                     LUMA_BITS);
    }
    return dot(float3(src[base], src[base + 1], src[base + 2]), float3(0.299f, 0.587f, 0.114f));
}

// Encode or decode signed sRGB samples without clipping their magnitude.
float color_transfer(float value, bool decode) {
    float magnitude = abs(value), result;
    if (decode)
        result = magnitude <= 0.04045f ? magnitude / 12.92f : pow((magnitude + 0.055f) / 1.055f, 2.4f);
    else
        result = magnitude <= 0.0031308f ? magnitude * 12.92f : 1.055f * pow(magnitude, 1.0f / 2.4f) - 0.055f;
    return copysign(result, value);
}

// Interpolate signed LUT samples with the CPU fixed-point scale.
int lut_mix(int a, int b, int shift) {
    return (a * (LUT_BLEND_SCALE - shift) + b * shift) >> LUT_BLEND_BITS;
}

// Map one color channel through a trilinear three-dimensional LUT.
template <typename T> void map_color_lut(device const T *src, device T *dst, constant uint *p, device const float *v, uint i) {
    uint channels = p[2], table_channels = p[6], output_mode = p[7];
    uint dc = output_mode == 2 || output_mode == 5 ? 3 : 4;
    if (i >= p[3] * p[4] * dc)
        return;
    uint band = i % dc, base = (i / dc) * channels;
    if (band >= table_channels) {
        dst[i] = src[base + band];
        return;
    }
    uint3 sizes = uint3(v[0], v[1], v[2]), cell;
    int3 shifts;
    float3 fractions;
    for (uint axis = 0; axis < 3; axis++) {
        if (is_same_v<T, uchar>) {
            uint position = uint(src[base + axis]) * uint(v[3 + axis]);
            cell[axis] = min(position >> LUT_POSITION_BITS, sizes[axis] - 2);
            shifts[axis] = int((position & ((1u << LUT_POSITION_BITS) - 1)) >> (LUT_POSITION_BITS - LUT_BLEND_BITS));
        } else {
            float sample = float(src[base + axis]);
            float position = (isnan(sample) ? 0.0f : clamp(sample, 0.0f, 1.0f)) * float(sizes[axis] - 1);
            cell[axis] = min(uint(position), sizes[axis] - 2);
            fractions[axis] = position - float(cell[axis]);
        }
    }
    uint index = LUT_HEADER_LEN + ((cell.z * sizes.y + cell.y) * sizes.x + cell.x) * table_channels + band;
    uint row_stride = sizes.x * table_channels, plane_stride = sizes.y * row_stride;
    if (is_same_v<T, uchar>) {
        int planes[2];
        for (uint z = 0; z < 2; z++) {
            uint start = index + z * plane_stride;
            int low = lut_mix(int(v[start]), int(v[start + table_channels]), shifts.x);
            int high = lut_mix(int(v[start + row_stride]), int(v[start + row_stride + table_channels]), shifts.x);
            planes[z] = lut_mix(low, high, shifts.y);
        }
        int value = (lut_mix(planes[0], planes[1], shifts.z) + (1 << (LUT_SAMPLE_BITS - 1))) >> LUT_SAMPLE_BITS;
        dst[i] = T(clamp(value, 0, 255));
    } else {
        float planes[2];
        for (uint z = 0; z < 2; z++) {
            uint start = index + z * plane_stride;
            float low = mix(v[start], v[start + table_channels], fractions.x);
            float high = mix(v[start + row_stride], v[start + row_stride + table_channels], fractions.x);
            planes[z] = mix(low, high, fractions.y);
        }
        dst[i] = T(mix(planes[0], planes[1], fractions.z));
    }
}

// Convert one channel between grayscale, color, alpha, and sample types.
template <typename S, typename D> void convert_image(device const S *src, device D *dst, constant uint *p, uint i) {
    uint c = p[2], mode = p[6];
    uint dc = (mode == 0 || mode == 4) ? 1 : (mode == 1 ? 2 : ((mode == 2 || mode == 5) ? 3 : 4));
    if (i >= p[3] * p[4] * dc)
        return;
    uint band = i % dc, base = (i / dc) * c;
    bool source_alpha = c == 2 || c == 4, output_alpha = dc == 2 || dc == 4;
    float value;
    if (output_alpha && band == dc - 1)
        value = source_alpha ? float(src[base + c - 1]) : (is_same_v<S, uchar> ? 255.0f : 1.0f);
    else if (dc < 3)
        value = luminance(src, base, c);
    else
        value = float(src[base + (c < 3 ? 0 : band)]);
    if (is_same_v<D, float>) {
        if (mode == 4) {
            if (is_same_v<S, float> && c >= 3)
                value *= 255.0f;
        } else if (is_same_v<S, uchar> || c == 1) {
            if (!(output_alpha && band == dc - 1 && !source_alpha))
                value /= 255.0f;
            else
                value = 1.0f;
        }
        dst[i] = D(value);
    } else if (is_same_v<S, float>) {
        if (c == 1) {
            value = output_alpha && band == dc - 1 ? 255.0f : value;
            dst[i] = D(isnan(value) ? 0.0f : clamp(trunc(value), 0.0f, 255.0f));
        } else
            dst[i] = store_sample<D>((isnan(value) ? 0.0f : clamp(value, 0.0f, 1.0f)) * 255.0f);
    } else
        dst[i] = D(value);
}

#define CONVERSION_KERNEL(NAME, SRC, DST)                                                                                                            \
    /* Convert one sample between the selected source and destination types. */                                                                      \
    kernel void NAME(device const SRC *s [[buffer(0)]],                                                                                              \
                     device DST *d [[buffer(1)]],                                                                                                    \
                     device const SRC *o [[buffer(2)]],                                                                                              \
                     constant uint *p [[buffer(3)]],                                                                                                 \
                     device const float *v [[buffer(4)]],                                                                                            \
                     uint i [[thread_position_in_grid]]) {                                                                                           \
        convert_image(s, d, p, i);                                                                                                                   \
    }
CONVERSION_KERNEL(convert_u8_u8, uchar, uchar)
CONVERSION_KERNEL(convert_u8_f32, uchar, float)
CONVERSION_KERNEL(convert_f32_u8, float, uchar)
CONVERSION_KERNEL(convert_f32_f32, float, float)
