#include "common.cuh"

// Match CPU luminance conversion and fractional box-filter precision.
const uint LUMA_RED = 19595, LUMA_GREEN = 38470, LUMA_BLUE = 7471, LUMA_BITS = 16;
const uint BOX_BITS = 24, BOX_SCALE = 1u << BOX_BITS, BOX_ROUND = BOX_SCALE / 2;
// Rank validation limits each neighborhood to 7x7 samples.
const uint MAX_RANK_SAMPLES = 49;
// Keep reduction blocks bounded and byte luminance sums exact across all passes.
const uint REDUCTION_BLOCK_SIZE = 256;
// CPU 3D LUTs use signed samples with 6 fractional bits and 15-bit interpolation.
const uint LUT_SAMPLE_BITS = 6, LUT_POSITION_BITS = 18, LUT_BLEND_BITS = 15;
const int LUT_BLEND_SCALE = 1 << LUT_BLEND_BITS;
const uint LUT_HEADER_LEN = 6;
// Statistics store extrema, four raw power sums, and the exact pixel count.
const uint STATISTICS_FIELDS = 7, MOMENT_ORDERS = 4, HISTOGRAM_BINS = 256;
// Resize coefficients use the CPU kernel's 22-bit fixed-point scale.
const uint RESIZE_BITS = 22, RESIZE_SCALE = 1u << RESIZE_BITS, RESIZE_ROUND = RESIZE_SCALE / 2;
const uint BLUR_OUTPUTS = 16, SLIDING_BLUR_RADIUS = 32;

// Byte sums can slide exactly, including modular overflow at extreme radii.
template <typename T, typename S>
__device__ __forceinline__ void sliding_box_blur(const T *__restrict__ src, T *__restrict__ dst, const uint *__restrict__ p,
                                                 const float *__restrict__ v, uint i, uint channels) {
    uint width = p[0], height = p[1], radius = p[6];
    bool horizontal = p[7] == 0;
    uint length = horizontal ? width : height, lines = horizontal ? height : width;
    uint blocks = (length + BLUR_OUTPUTS - 1) / BLUR_OUTPUTS;
    if (i >= blocks * lines * channels)
        return;
    uint band = i % channels, lane = i / channels;
    uint line = horizontal ? lane / blocks : lane % lines;
    uint position = (horizontal ? lane % blocks : lane / lines) * BLUR_OUTPUTS;
    uint last = length - 1;
    uint base = horizontal ? line * width * channels + band : line * channels + band;
    uint stride = horizontal ? channels : width * channels;
    uint low = position > radius ? position - radius : 0, high = min(last, position + radius);
    S sum = S(0);
    for (uint t = low; t <= high; t++)
        sum += S(src[base + t * stride]);
    sum += S(src[base]) * (radius > position ? radius - position : 0);
    sum += S(src[base + last * stride]) * (position + radius > last ? position + radius - last : 0);
    uint end = min(position + BLUR_OUTPUTS, length);
    uint v0 = uint(v[0]), v1 = uint(v[1]);
    for (; position < end; position++) {
        uint left = base + (position > radius ? position - radius - 1 : 0) * stride;
        uint right = base + min(position + radius + 1, last) * stride;
        S r_val = S(src[right]);
        S fixed = sum * v0 + (S(src[left]) + r_val) * v1 + BOX_ROUND;
        dst[base + position * stride] = T(fixed >> BOX_BITS);
        uint outgoing = base + (position > radius ? position - radius : 0) * stride;
        sum = sum - S(src[outgoing]) + r_val;
    }
}

template <typename T> __device__ __forceinline__ float luminance(const T *__restrict__ src, uint base, uint channels) {
    if (channels < 3)
        return float(src[base]);
    if (is_same_v<T, uchar>) {
        return float((uint(src[base]) * LUMA_RED + uint(src[base + 1]) * LUMA_GREEN + uint(src[base + 2]) * LUMA_BLUE + (1u << (LUMA_BITS - 1))) >>
                     LUMA_BITS);
    }
    return dot(vec3<float>(src[base], src[base + 1], src[base + 2]), vec3<float>(0.299f, 0.587f, 0.114f));
}

__device__ __forceinline__ float color_transfer(float value, bool decode) {
    float magnitude = fabsf(value), result;
    if (decode)
        result = magnitude <= 0.04045f ? magnitude / 12.92f : powf((magnitude + 0.055f) / 1.055f, 2.4f);
    else
        result = magnitude <= 0.0031308f ? magnitude * 12.92f : 1.055f * powf(magnitude, 1.0f / 2.4f) - 0.055f;
    return copysignf(result, value);
}

__device__ __forceinline__ int lut_mix(int a, int b, int shift) {
    return (a * (LUT_BLEND_SCALE - shift) + b * shift) >> LUT_BLEND_BITS;
}

__device__ __forceinline__ float warp_cubic(vec4<float> v, float t) {
    float a = -v[0] + v[1] - v[2] + v[3];
    float b = 2.0f * v[0] - 2.0f * v[1] + v[2] - v[3];
    float c = -v[0] + v[2];
    return ((a * t + b) * t + c) * t + v[1];
}

template <typename T>
__device__ __forceinline__ void map_color_lut(const T *__restrict__ src, T *__restrict__ dst, const uint *__restrict__ p, const float *__restrict__ v,
                                              uint i) {
    uint channels = p[2], table_channels = p[6], output_mode = p[7];
    uint dc = output_mode == 2 || output_mode == 5 ? 3 : 4;
    if (i >= p[3] * p[4] * dc)
        return;
    uint band = i % dc, base = (i / dc) * channels;
    if (band >= table_channels) {
        dst[i] = src[base + band];
        return;
    }
    uint sz_x = uint(v[0]), sz_y = uint(v[1]), sz_z = uint(v[2]);
    uint row_stride = sz_x * table_channels, plane_stride = sz_y * row_stride;

    if (is_same_v<T, uchar>) {
        uint pos_x = uint(src[base + 0]) * uint(v[3]);
        uint pos_y = uint(src[base + 1]) * uint(v[4]);
        uint pos_z = uint(src[base + 2]) * uint(v[5]);

        uint cell_x = min(pos_x >> LUT_POSITION_BITS, sz_x - 2);
        uint cell_y = min(pos_y >> LUT_POSITION_BITS, sz_y - 2);
        uint cell_z = min(pos_z >> LUT_POSITION_BITS, sz_z - 2);

        int shift_x = int((pos_x & ((1u << LUT_POSITION_BITS) - 1)) >> (LUT_POSITION_BITS - LUT_BLEND_BITS));
        int shift_y = int((pos_y & ((1u << LUT_POSITION_BITS) - 1)) >> (LUT_POSITION_BITS - LUT_BLEND_BITS));
        int shift_z = int((pos_z & ((1u << LUT_POSITION_BITS) - 1)) >> (LUT_POSITION_BITS - LUT_BLEND_BITS));

        uint index = LUT_HEADER_LEN + ((cell_z * sz_y + cell_y) * sz_x + cell_x) * table_channels + band;
        int planes[2];
#pragma unroll
        for (uint z = 0; z < 2; z++) {
            uint start = index + z * plane_stride;
            int low = lut_mix(int(v[start]), int(v[start + table_channels]), shift_x);
            int high = lut_mix(int(v[start + row_stride]), int(v[start + row_stride + table_channels]), shift_x);
            planes[z] = lut_mix(low, high, shift_y);
        }
        int value = (lut_mix(planes[0], planes[1], shift_z) + (1 << (LUT_SAMPLE_BITS - 1))) >> LUT_SAMPLE_BITS;
        dst[i] = T(clamp(value, 0, 255));
    } else {
        float s_x = float(src[base + 0]);
        float s_y = float(src[base + 1]);
        float s_z = float(src[base + 2]);

        float pos_x = (isnan(s_x) ? 0.0f : clamp(s_x, 0.0f, 1.0f)) * float(sz_x - 1);
        float pos_y = (isnan(s_y) ? 0.0f : clamp(s_y, 0.0f, 1.0f)) * float(sz_y - 1);
        float pos_z = (isnan(s_z) ? 0.0f : clamp(s_z, 0.0f, 1.0f)) * float(sz_z - 1);

        uint cell_x = min(uint(pos_x), sz_x - 2);
        uint cell_y = min(uint(pos_y), sz_y - 2);
        uint cell_z = min(uint(pos_z), sz_z - 2);

        float frac_x = pos_x - float(cell_x);
        float frac_y = pos_y - float(cell_y);
        float frac_z = pos_z - float(cell_z);

        uint index = LUT_HEADER_LEN + ((cell_z * sz_y + cell_y) * sz_x + cell_x) * table_channels + band;
        float planes[2];
#pragma unroll
        for (uint z = 0; z < 2; z++) {
            uint start = index + z * plane_stride;
            float low = mix(v[start], v[start + table_channels], frac_x);
            float high = mix(v[start + row_stride], v[start + row_stride + table_channels], frac_x);
            planes[z] = mix(low, high, frac_y);
        }
        dst[i] = T(mix(planes[0], planes[1], frac_z));
    }
}

template <typename T>
__device__ __forceinline__ void reduce_luminance(const T *__restrict__ src, float *__restrict__ dst, const uint *__restrict__ p, uint i) {
    if ((i & 3u) != 0 || i >= p[3] * 4)
        return;
    uint block = i >> 2, first = block * REDUCTION_BLOCK_SIZE;
    uint last = min(first + REDUCTION_BLOCK_SIZE, p[0] * p[1]), stage = p[6], c = p[2];

    if (stage == 0) {
        uint64 integer_sum = 0;
        for (uint pixel = first; pixel < last; pixel++)
            integer_sum += uint64(luminance(src, pixel * c, c));
        dst[i] = as_type<float>(uint(integer_sum));
        dst[i + 1] = as_type<float>(uint(integer_sum >> 32));
    } else if (stage == 1) {
        float sum = 0;
        for (uint pixel = first; pixel < last; pixel++)
            sum += luminance(src, pixel * c, c);
        dst[i] = sum;
        dst[i + 1] = 0;
    } else if (stage == 2) {
        uint64 integer_sum = 0;
        const uint *words = reinterpret_cast<const uint *>(src);
        for (uint pixel = first; pixel < last; pixel++)
            integer_sum += uint64(words[pixel * 4]) | (uint64(words[pixel * 4 + 1]) << 32);
        dst[i] = as_type<float>(uint(integer_sum));
        dst[i + 1] = as_type<float>(uint(integer_sum >> 32));
    } else {
        float sum = 0;
        for (uint pixel = first; pixel < last; pixel++)
            sum += float(src[pixel * 4]);
        dst[i] = sum;
        dst[i + 1] = 0;
    }
    dst[i + 2] = 0;
    dst[i + 3] = 0;
}

template <typename T> __device__ __forceinline__ float read_sample(const T *__restrict__ p, uint i) {
    return float(p[i]);
}
template <typename T> __device__ __forceinline__ T store_sample(float v) {
    return T(v);
}
template <> __device__ __forceinline__ uchar store_sample<uchar>(float v) {
    return uchar(clamp(floorf(v + 0.5f), 0.0f, 255.0f));
}

template <typename T>
__device__ void process(const T *__restrict__ src, T *__restrict__ dst, const T *__restrict__ other, const uint *__restrict__ p,
                        const float *__restrict__ v, uint i) {
    uint sw = p[0], sh = p[1], c = p[2], dw = p[3], dh = p[4], op = p[5], k = p[6];
    if (op == 18 && is_same_v<T, uchar> && k >= SLIDING_BLUR_RADIUS) {
        sliding_box_blur<T, uint>(src, dst, p, v, i, c);
        return;
    }
    if (op == 29) {
        uint blocks = max(p[7], 1u), pixels = sw * sh;
        if (i >= c * blocks)
            return;
        uint band = i / blocks, block = i % blocks;
        uint chunk = (pixels + blocks - 1) / blocks;
        uint first = min(block * chunk, pixels), last = min(first + chunk, pixels);
        float *output = reinterpret_cast<float *>(dst);
        if (k == 1) {
            uint *hist_out = reinterpret_cast<uint *>(output + i * HISTOGRAM_BINS);
            for (uint bin = 0; bin < HISTOGRAM_BINS; bin++)
                hist_out[bin] = 0;
            for (uint pixel = first; pixel < last; pixel++)
                hist_out[uint(src[pixel * c + band])]++;
        } else {
            float low = INFINITY, high = -INFINITY;
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            bool nan = false;
            for (uint pixel = first; pixel < last; pixel++) {
                float value = float(src[pixel * c + band]);
                nan = nan || isnan(value);
                low = min(low, value);
                high = max(high, value);
                float p2 = value * value;
                s0 += value;
                s1 += p2;
                s2 += p2 * value;
                s3 += p2 * value * value;
            }
            float *out_stat = output + i * STATISTICS_FIELDS;
            out_stat[0] = nan ? NAN : low;
            out_stat[1] = nan ? NAN : high;
            out_stat[2] = s0;
            out_stat[3] = s1;
            out_stat[4] = s2;
            out_stat[5] = s3;
            out_stat[6] = as_type<float>(last - first);
        }
        return;
    }
    if (op == 15) {
        if (i < dw * dh)
            dst[i] = src[i * c + k];
        return;
    }
    if (op == 25) {
        map_color_lut(src, dst, p, v, i);
        return;
    }
    if (i >= dw * dh * c)
        return;
    uint band, pixel_idx, x, y;
    if (c == 4) {
        band = i & 3u;
        pixel_idx = i >> 2;
    } else if (c == 1) {
        band = 0;
        pixel_idx = i;
    } else if (c == 2) {
        band = i & 1u;
        pixel_idx = i >> 1;
    } else {
        pixel_idx = i / c;
        band = i - pixel_idx * c;
    }
    y = pixel_idx / dw;
    x = pixel_idx - y * dw;

    if (op == 30) {
        float value = float(src[i]);
        value = isnan(value) ? 0.0f : clamp(value, 0.0f, 1.0f);
        float v0 = v[0];
        dst[i] = T(floorf(value * v0 + 0.5f) / v0);
        return;
    }
    if (op == 28) {
        float mask = float(other[pixel_idx]) / v[0];
        mask = isnan(mask) ? 0.0f : clamp(mask, 0.0f, 1.0f);
        if (p[7] == 1)
            mask = 1.0f - mask;
        dst[i] = mask == 0.0f ? T(0) : T(float(src[i]) * mask);
        return;
    }
    if (op == 26 || op == 27) {
        float sx, sy;
        if (op == 27) {
            const float *coordinates = reinterpret_cast<const float *>(other);
            sx = coordinates[pixel_idx * 3];
            sy = coordinates[pixel_idx * 3 + 1];
        } else {
            float px = float(x) + 0.5f, py = float(y) + 0.5f;
            float denominator = v[6] * px + v[7] * py + 1.0f;
            sx = (v[0] * px + v[1] * py + v[2]) / denominator;
            sy = (v[3] * px + v[4] * py + v[5]) / denominator;
        }
        if (!isfinite(sx) || !isfinite(sy) || sx < 0.0f || sy < 0.0f || sx >= float(sw) || sy >= float(sh)) {
            dst[i] = store_sample<T>(v[8 + band]);
            return;
        }
        if (k == 0) {
            dst[i] = src[(uint(sy) * sw + uint(sx)) * c + band];
            return;
        }
        sx -= 0.5f;
        sy -= 0.5f;
        float flx = floorf(sx), fly = floorf(sy);
        int ix = int(flx), iy = int(fly);
        float fx = sx - flx, fy = sy - fly, result = 0.0f;
        int max_x = int(sw) - 1, max_y = int(sh) - 1;
        if (k == 2) {
            vec4<float> rows;
#pragma unroll
            for (int ky = -1; ky <= 2; ky++) {
                uint row_offset = uint(clamp(iy + ky, 0, max_y)) * sw;
                vec4<float> taps;
#pragma unroll
                for (int kx = -1; kx <= 2; kx++) {
                    uint index = (row_offset + uint(clamp(ix + kx, 0, max_x))) * c + band;
                    taps[kx + 1] = float(src[index]);
                }
                rows[ky + 1] = warp_cubic(taps, fx);
            }
            result = warp_cubic(rows, fy);
            dst[i] = store_sample<T>(is_same_v<T, uchar> ? floorf(result) : result);
            return;
        }
        uint x0 = uint(clamp(ix, 0, max_x)) * c + band;
        uint x1 = uint(clamp(ix + 1, 0, max_x)) * c + band;
        uint y0 = uint(clamp(iy, 0, max_y)) * sw * c;
        uint y1 = uint(clamp(iy + 1, 0, max_y)) * sw * c;
        float wx0 = 1.0f - fx, wx1 = fx;
        float row0 = float(src[y0 + x0]) * wx0 + float(src[y0 + x1]) * wx1;
        float row1 = float(src[y1 + x0]) * wx0 + float(src[y1 + x1]) * wx1;
        result = row0 * (1.0f - fy) + row1 * fy;
        dst[i] = store_sample<T>(is_same_v<T, uchar> ? floorf(result) : result);
        return;
    }
    if (op == 23 || op == 24) {
        float value = float(src[i]);
        if (op == 23 && band < 3)
            value = color_transfer(value, k == 0);
        if (op == 24 && band == c - 1)
            value = isnan(value) ? 0.0f : clamp(value, 0.0f, 1.0f);
        dst[i] = T(value);
        return;
    }
    if (op == 22) {
        float original = float(src[i]), base = original;
        if ((k != 2 && (c == 2 || c == 4) && band == c - 1) ||
            (k == 2 && (x == 0 || y == 0 || x + 1 >= sw || y + 1 >= sh || (c == 4 && band == c - 1)))) {
            dst[i] = src[i];
            return;
        }
        if (k == 0)
            base = luminance(src, i - band, c);
        else if (k == 1) {
            if (is_same_v<T, uchar>) {
                const uint *words = reinterpret_cast<const uint *>(other);
                uint64 total = uint64(words[0]) | (uint64(words[1]) << 32);
                base = float((total * 2 + p[7]) / (uint64(p[7]) * 2));
            } else
                base = float(other[0]) / float(p[7]);
        } else {
            float sum = original * 4.0f;
#pragma unroll
            for (int ky = -1; ky <= 1; ky++) {
                uint row_base = (y + ky) * sw + x;
#pragma unroll
                for (int kx = -1; kx <= 1; kx++) {
                    sum += float(src[(row_base + kx) * c + band]);
                }
            }
            base = is_same_v<T, uchar> ? float((uint(sum) + 6u) / 13u) : sum / 13.0f;
        }
        float v0 = v[0];
        float value = v0 == 0.0f ? base : fmaf(v0, original - base, base);
        if (is_same_v<T, uchar>)
            value = floorf(value);
        dst[i] = store_sample<T>(value);
        return;
    }
    if (op == 16 || op == 17) {
        dst[i] = op == 16 ? (band == k ? other[pixel_idx] : src[i]) : (band == c - 1 ? store_sample<T>(v[0]) : src[i]);
        return;
    }
    if (op == 18) {
        bool horiz = p[7] == 0;
        int pos = int(horiz ? x : y), last = int(horiz ? sw : sh) - 1, ik = int(k);
        int low = max(0, pos - ik), high = min(last, pos + ik);
        uint repeat_low = uint(max(0, ik - pos)), repeat_high = uint(max(0, pos + ik - last));
        uint first = horiz ? y * sw * c + band : x * c + band;
        uint stride = horiz ? c : sw * c;
        uint left = first + uint(clamp(pos - ik - 1, 0, last)) * stride;
        uint right = first + uint(clamp(pos + ik + 1, 0, last)) * stride;

        if (is_same_v<T, uchar>) {
            uint integer_sum = 0;
            for (int t = low; t <= high; t++)
                integer_sum += uint(src[first + uint(t) * stride]);
            integer_sum += uint(src[first]) * repeat_low + uint(src[first + uint(last) * stride]) * repeat_high;
            uint fixed = integer_sum * uint(v[0]) + (uint(src[left]) + uint(src[right])) * uint(v[1]) + BOX_ROUND;
            dst[i] = T(fixed >> BOX_BITS);
        } else {
            float sum = 0.0f;
            for (int t = low; t <= high; t++)
                sum += float(src[first + uint(t) * stride]);
            if (repeat_low)
                sum += float(src[first]) * float(repeat_low);
            if (repeat_high)
                sum += float(src[first + uint(last) * stride]) * float(repeat_high);
            float v0 = v[0], v1 = v[1];
            float value = v0 == 0.0f ? 0.0f : sum * (v0 / float(BOX_SCALE));
            if (v1 != 0.0f)
                value += (float(src[left]) + float(src[right])) * (v1 / float(BOX_SCALE));
            dst[i] = T(value);
        }
        return;
    }
    if (op == 19) {
        uint rank = p[7], total_k = k * k;
        int half_k = int(k >> 1), max_x = int(sw) - 1, max_y = int(sh) - 1;
        if (rank == 0 || rank == total_k - 1) {
            float extreme = 0.0f;
            uint count = 0;
            for (uint ky = 0; ky < k; ky++) {
                uint sy = uint(clamp(int(y) + int(ky) - half_k, 0, max_y)) * sw;
                for (uint kx = 0; kx < k; kx++) {
                    uint sx = uint(clamp(int(x) + int(kx) - half_k, 0, max_x));
                    float value = read_sample(src, (sy + sx) * c + band);
                    if (!is_same_v<T, uchar> && isnan(value)) {
                        dst[i] = T(value);
                        return;
                    }
                    extreme = count == 0 ? value : (rank == 0 ? min(extreme, value) : max(extreme, value));
                    count++;
                }
            }
            dst[i] = T(extreme);
            return;
        }
        if (is_same_v<T, uchar>) {
            uchar samples[MAX_RANK_SAMPLES];
            uint count = 0;
            for (uint ky = 0; ky < k; ky++) {
                uint sy = uint(clamp(int(y) + int(ky) - half_k, 0, max_y)) * sw;
                for (uint kx = 0; kx < k; kx++) {
                    uint sx = uint(clamp(int(x) + int(kx) - half_k, 0, max_x));
                    uchar value = src[(sy + sx) * c + band];
                    uint j = min(count, rank);
                    if (count > rank && value >= samples[rank]) {
                        count++;
                        continue;
                    }
                    while (j > 0 && samples[j - 1] > value) {
                        samples[j] = samples[j - 1];
                        j--;
                    }
                    samples[j] = value;
                    count++;
                }
            }
            dst[i] = T(samples[rank]);
        } else {
            float samples[MAX_RANK_SAMPLES];
            uint count = 0;
            for (uint ky = 0; ky < k; ky++) {
                uint sy = uint(clamp(int(y) + int(ky) - half_k, 0, max_y)) * sw;
                for (uint kx = 0; kx < k; kx++) {
                    uint sx = uint(clamp(int(x) + int(kx) - half_k, 0, max_x));
                    float value = read_sample(src, (sy + sx) * c + band);
                    if (isnan(value)) {
                        dst[i] = T(value);
                        return;
                    }
                    uint j = min(count, rank);
                    if (count > rank && value >= samples[rank]) {
                        count++;
                        continue;
                    }
                    while (j > 0 && samples[j - 1] > value) {
                        samples[j] = samples[j - 1];
                        j--;
                    }
                    samples[j] = value;
                    count++;
                }
            }
            dst[i] = T(samples[rank]);
        }
        return;
    }
    if (op == 7) {
        int sx = int(x) + int(k), sy = int(y) + int(p[7]);
        dst[i] = (uint(sx) < sw && uint(sy) < sh) ? src[(uint(sy) * sw + uint(sx)) * c + band] : T(0);
        return;
    }
    if (op == 8) {
        uint sx = x, sy = y;
        switch (k) {
        case 0:
            sx = sw - 1 - x;
            break;
        case 1:
            sy = sh - 1 - y;
            break;
        case 2:
            sx = sw - 1 - y;
            sy = x;
            break;
        case 3:
            sx = sw - 1 - x;
            sy = sh - 1 - y;
            break;
        case 4:
            sx = y;
            sy = sh - 1 - x;
            break;
        case 5:
            sx = y;
            sy = x;
            break;
        case 6:
            sx = sw - 1 - y;
            sy = sh - 1 - x;
            break;
        }
        dst[i] = src[(sy * sw + sx) * c + band];
        return;
    }
    if (op == 9) {
        uint sx = uint((int64(x) - int(k) + sw) % sw), sy = uint((int64(y) - int(p[7]) + sh) % sh);
        dst[i] = src[(sy * sw + sx) * c + band];
        return;
    }
    float a = op == 0 ? 0.0f : read_sample(src, i), out = a;
    float maxval = is_same_v<T, uchar> ? 255.0f : 1.0f;
    if (op == 0) {
        bool horiz = p[7] == 0;
        uint pos = horiz ? x : y;
        uint base = pos * k, start = uint(v[base]), count = uint(v[base + 1]);
        uint first = horiz ? (y * sw + start) * c + band : (start * sw + x) * c + band;
        uint stride = horiz ? c : sw * c;
        const float *weights = v + base + 2;
        if (is_same_v<T, uchar>) {
            int fixed = RESIZE_ROUND;
            for (uint t = 0; t < count; t++)
                fixed += int(src[first + t * stride]) * int(roundf(weights[t] * float(RESIZE_SCALE)));
            out = float(fixed >> RESIZE_BITS);
        } else {
            out = 0.0f;
            for (uint t = 0; t < count; t++)
                out += read_sample(src, first + t * stride) * weights[t];
        }
    } else if (op == 1) {
        uint r = k >> 1;
        if (x >= r && y >= r && x + r < sw && y + r < sh) {
            out = v[1];
            const float *kw = v + 2;
            for (uint ky = 0; ky < k; ky++) {
                float subtotal = 0.0f;
                uint row_base = ((y + r - ky) * sw + x - r) * c + band;
                for (uint kx = 0; kx < k; kx++)
                    subtotal += read_sample(src, row_base + kx * c) * kw[kx];
                out += subtotal;
                kw += k;
            }
        }
    } else if (op == 2) {
        out = a + (read_sample(other, i) - a) * v[0];
        if (is_same_v<T, uchar>)
            out = floorf(out);
    } else if (op == 3) {
        uint base = i - band;
        float da = read_sample(src, base + c - 1) / maxval, sa = read_sample(other, base + c - 1) / maxval;
        da = isnan(da) ? 0.0f : clamp(da, 0.0f, 1.0f);
        sa = isnan(sa) ? 0.0f : clamp(sa, 0.0f, 1.0f);
        if (is_same_v<T, uchar> && sa == 0.0f) {
            dst[i] = src[i];
            return;
        }
        if (sa == 1.0f) {
            dst[i] = band == c - 1 ? T(maxval) : other[i];
            return;
        }
        float alpha = sa + da * (1.0f - sa);
        out = band == c - 1
                  ? alpha * maxval
                  : (alpha == 0.0f ? 0.0f : ((sa == 0.0f ? 0.0f : read_sample(other, i) * sa) + (da == 0.0f ? 0.0f : a * da) * (1.0f - sa)) / alpha);
    } else if (op == 4 || op == 5) {
        float raw = read_sample(src, i - band + c - 1);
        float alpha = isnan(raw) ? 0.0f : clamp(raw / maxval, 0.0f, 1.0f);
        out = band == c - 1 ? alpha * maxval
                            : (alpha == 0.0f ? ((op == 5 && is_same_v<T, uchar>) ? a : 0.0f)
                                             : (op == 4 ? a * alpha : (is_same_v<T, uchar> ? a * maxval / raw : a / alpha)));
        if (op == 5 && is_same_v<T, uchar>)
            out = floorf(out);
    } else if (op == 10) {
        if (!(c == 4 && band == 3)) {
            out = a * v[0];
            if (is_same_v<T, uchar>)
                out = floorf(out);
        }
    } else if (op == 11) {
        out = maxval - a;
    } else if (op == 12) {
        out = v[(band << 8) + uint(a)];
    } else if (op == 20) {
        float difference = a - read_sample(other, i);
        if (fabsf(difference) > v[0]) {
            if (is_same_v<T, uchar>)
                out = float(int(a) + (int(difference) * int(k)) / 100);
            else
                out = a + difference * float(int(k)) / 100.0f;
        }
    } else if (op == 13) {
        float b = read_sample(other, i);
        if (is_same_v<T, uchar>) {
            switch (k) {
            case 0:
                out = fabsf(a - b);
                break;
            case 1:
                out = float((uint(a) * uint(b)) / 255u);
                break;
            case 2:
                out = float(255u - ((255u - uint(a)) * (255u - uint(b))) / 255u);
                break;
            case 3:
                out = max(a, b);
                break;
            case 4:
                out = min(a, b);
                break;
            case 5:
                out = (a + b) / v[0] + v[1];
                if (!isfinite(out) || out >= 2147483648.0f || out < -2147483648.0f)
                    out = 0.0f;
                out = floorf(out);
                break;
            case 6:
                out = (a - b) / v[0] + v[1];
                if (!isfinite(out) || out >= 2147483648.0f || out < -2147483648.0f)
                    out = 0.0f;
                out = floorf(out);
                break;
            }
        } else {
            switch (k) {
            case 0:
                out = fabsf(a - b);
                break;
            case 1:
                out = a * b / maxval;
                break;
            case 2:
                out = maxval - (maxval - a) * (maxval - b) / maxval;
                break;
            case 3:
                out = (isnan(a) || isnan(b)) ? NAN : max(a, b);
                break;
            case 4:
                out = (isnan(a) || isnan(b)) ? NAN : min(a, b);
                break;
            case 5:
                out = (a + b) / v[0] + v[1];
                break;
            case 6:
                out = (a - b) / v[0] + v[1];
                break;
            }
        }
    }
    dst[i] = store_sample<T>(out);
}

template <typename S, typename D>
__device__ __forceinline__ void convert_image(const S *__restrict__ src, D *__restrict__ dst, const uint *__restrict__ p, uint i) {
    uint c = p[2], mode = p[6];
    uint dc = (mode == 0 || mode == 4) ? 1 : (mode == 1 ? 2 : ((mode == 2 || mode == 5) ? 3 : 4));
    if (i >= p[3] * p[4] * dc)
        return;
    uint band = i % dc, base = (i / dc) * c;
    bool source_alpha = c == 2 || c == 4, output_alpha = dc == 2 || dc == 4;
    float value;
    if (output_alpha && band == dc - 1)
        value = source_alpha ? float(src[base + c - 1]) : (is_same_v<S, uchar> ? 255.0f : 1.0f);
    else if (mode == 4 && is_same_v<S, uchar> && c >= 3)
        value = float(uint(src[base]) * 299u + uint(src[base + 1]) * 587u + uint(src[base + 2]) * 114u) / 1000.0f;
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
            dst[i] = D(isnan(value) ? 0.0f : clamp(truncf(value), 0.0f, 255.0f));
        } else
            dst[i] = store_sample<D>((isnan(value) ? 0.0f : clamp(value, 0.0f, 1.0f)) * 255.0f);
    } else
        dst[i] = D(value);
}

#define CONVERSION_KERNEL(NAME, SRC, DST)                                                                                                            \
    extern "C" __global__ void NAME(const SRC *src, DST *dst, const SRC *other, const uint *p, const float *v) {                                     \
        uint i = blockIdx.x * blockDim.x + threadIdx.x;                                                                                              \
        convert_image(src, dst, p, i);                                                                                                               \
    }
CONVERSION_KERNEL(convert_u8_u8, uchar, uchar)
CONVERSION_KERNEL(convert_u8_f32, uchar, float)
CONVERSION_KERNEL(convert_f32_u8, float, uchar)
CONVERSION_KERNEL(convert_f32_f32, float, float)

#define REDUCTION_KERNEL(NAME, SRC)                                                                                                                  \
    extern "C" __global__ void NAME(const SRC *src, float *dst, const SRC *other, const uint *p, const float *v) {                                   \
        uint i = blockIdx.x * blockDim.x + threadIdx.x;                                                                                              \
        reduce_luminance(src, dst, p, i);                                                                                                            \
    }
REDUCTION_KERNEL(reduce_u8, uchar)
REDUCTION_KERNEL(reduce_f32, float)

extern "C" __global__ void finish_statistics(const float *__restrict__ src, float *__restrict__ dst, const uint *__restrict__ p) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x, blocks = p[7];
    if (p[6] == 1) {
        if (i >= p[2] * HISTOGRAM_BINS)
            return;
        uint band = i >> 8, bin = i & 255u, count = 0;
        const float *src_ptr = src + (band * blocks * HISTOGRAM_BINS) + bin;
        for (uint block = 0; block < blocks; block++)
            count += as_type<uint>(src_ptr[block * HISTOGRAM_BINS]);
        dst[i] = as_type<float>(count);
        return;
    }
    if (i >= p[2])
        return;
    float low = INFINITY, high = -INFINITY;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    bool nan = false;
    const float *partial = src + i * blocks * STATISTICS_FIELDS;
    for (uint block = 0; block < blocks; block++, partial += STATISTICS_FIELDS) {
        float p_low = partial[0];
        nan = nan || isnan(p_low);
        low = min(low, p_low);
        high = max(high, partial[1]);
        s0 += partial[2];
        s1 += partial[3];
        s2 += partial[4];
        s3 += partial[5];
    }
    float *result = dst + i * STATISTICS_FIELDS;
    result[0] = nan ? NAN : low;
    result[1] = nan ? NAN : high;
    result[2] = s0;
    result[3] = s1;
    result[4] = s2;
    result[5] = s3;
    result[6] = as_type<float>(p[0] * p[1]);
}

extern "C" __global__ void image_u8(const uchar *src, uchar *dst, const uchar *other, const uint *p, const float *v) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    process(src, dst, other, p, v, i);
}
extern "C" __global__ void image_f32(const float *src, float *dst, const float *other, const uint *p, const float *v) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    process(src, dst, other, p, v, i);
}

template <typename T> __device__ __forceinline__ void copy_tile(const T *__restrict__ src, T *__restrict__ dst, const uint *__restrict__ p, uint i) {
    uint channels = p[2], width = p[9], height = p[10];
    if (i >= width * height * channels)
        return;
    uint cw = channels * width;
    uint y = i / cw;
    uint rem = i - y * cw;
    uint x = rem / channels;
    uint band = rem - x * channels;
    dst[((y + p[8]) * p[3] + x + p[7]) * channels + band] = src[((y + p[6]) * p[0] + x + p[5]) * channels + band];
}
extern "C" __global__ void tile_u8(const uchar *src, uchar *dst, const uint *p) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    copy_tile(src, dst, p, i);
}
extern "C" __global__ void tile_f32(const float *src, float *dst, const uint *p) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    copy_tile(src, dst, p, i);
}