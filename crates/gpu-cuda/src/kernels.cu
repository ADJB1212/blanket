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
template <typename T, typename S> __device__ void sliding_box_blur(const T *src, T *dst, const uint *p, const float *v, uint i, uint channels) {
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
    for (; position < end; position++) {
        uint left = base + (position > radius ? position - radius - 1 : 0) * stride;
        uint right = base + min(position + radius + 1, last) * stride;
        S fixed = sum * uint(v[0]) + (S(src[left]) + S(src[right])) * uint(v[1]) + BOX_ROUND;
        dst[base + position * stride] = T(fixed >> BOX_BITS);
        uint outgoing = base + (position > radius ? position - radius : 0) * stride;
        sum = sum - S(src[outgoing]) + S(src[right]);
    }
}

template <typename T> __device__ float luminance(const T *src, uint base, uint channels) {
    if (channels < 3)
        return float(src[base]);
    if (is_same_v<T, uchar>) {
        return float((uint(src[base]) * LUMA_RED + uint(src[base + 1]) * LUMA_GREEN + uint(src[base + 2]) * LUMA_BLUE + (1u << (LUMA_BITS - 1))) >>
                     LUMA_BITS);
    }
    return dot(vec3<float>(src[base], src[base + 1], src[base + 2]), vec3<float>(0.299f, 0.587f, 0.114f));
}

__device__ float color_transfer(float value, bool decode) {
    float magnitude = abs(value), result;
    if (decode)
        result = magnitude <= 0.04045f ? magnitude / 12.92f : powf((magnitude + 0.055f) / 1.055f, 2.4f);
    else
        result = magnitude <= 0.0031308f ? magnitude * 12.92f : 1.055f * powf(magnitude, 1.0f / 2.4f) - 0.055f;
    return copysignf(result, value);
}

__device__ int lut_mix(int a, int b, int shift) {
    return (a * (LUT_BLEND_SCALE - shift) + b * shift) >> LUT_BLEND_BITS;
}

__device__ float warp_cubic(vec4<float> v, float t) {
    float a = -v[0] + v[1] - v[2] + v[3];
    float b = 2 * v[0] - 2 * v[1] + v[2] - v[3];
    float c = -v[0] + v[2];
    return ((a * t + b) * t + c) * t + v[1];
}

template <typename T> __device__ void map_color_lut(const T *src, T *dst, const uint *p, const float *v, uint i) {
    uint channels = p[2], table_channels = p[6], output_mode = p[7];
    uint dc = output_mode == 2 || output_mode == 5 ? 3 : 4;
    if (i >= p[3] * p[4] * dc)
        return;
    uint band = i % dc, base = (i / dc) * channels;
    if (band >= table_channels) {
        dst[i] = src[base + band];
        return;
    }
    vec3<uint> sizes = vec3<uint>(v[0], v[1], v[2]), cell;
    vec3<int> shifts;
    vec3<float> fractions;
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

template <typename T> __device__ void reduce_luminance(const T *src, float *dst, const uint *p, uint i) {
    if (i % 4 != 0 || i >= p[3] * 4)
        return;
    uint block = i / 4, first = block * REDUCTION_BLOCK_SIZE;
    uint last = min(first + REDUCTION_BLOCK_SIZE, p[0] * p[1]), stage = p[6];
    uint64 integer_sum = 0;
    float sum = 0;
    for (uint pixel = first; pixel < last; pixel++) {
        if (stage == 0)
            integer_sum += uint64(luminance(src, pixel * p[2], p[2]));
        else if (stage == 1)
            sum += luminance(src, pixel * p[2], p[2]);
        else if (stage == 2) {
            const uint *words = reinterpret_cast<const uint *>(src);
            integer_sum += uint64(words[pixel * 4]) | (uint64(words[pixel * 4 + 1]) << 32);
        } else
            sum += float(src[pixel * 4]);
    }
    if (stage == 0 || stage == 2) {
        dst[i] = as_type<float>(uint(integer_sum));
        dst[i + 1] = as_type<float>(uint(integer_sum >> 32));
    } else {
        dst[i] = sum;
        dst[i + 1] = 0;
    }
    dst[i + 2] = 0;
    dst[i + 3] = 0;
}

template <typename T> __device__ float read_sample(const T *p, uint i) {
    return float(p[i]);
}
template <typename T> __device__ T store_sample(float v) {
    return T(v);
}
template <> __device__ uchar store_sample<uchar>(float v) {
    return uchar(clamp(floorf(v + 0.5f), 0.0f, 255.0f));
}

template <typename T> __device__ void process(const T *src, T *dst, const T *other, const uint *p, const float *v, uint i) {
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
            uint counts[HISTOGRAM_BINS] = {0};
            for (uint pixel = first; pixel < last; pixel++)
                counts[uint(src[pixel * c + band])]++;
            for (uint bin = 0; bin < HISTOGRAM_BINS; bin++)
                output[i * HISTOGRAM_BINS + bin] = as_type<float>(counts[bin]);
        } else {
            float low = INFINITY, high = -INFINITY;
            float sums[MOMENT_ORDERS] = {0};
            bool nan = false;
            for (uint pixel = first; pixel < last; pixel++) {
                float value = float(src[pixel * c + band]);
                nan = nan || isnan(value);
                low = min(low, value);
                high = max(high, value);
                float power = value;
                for (uint order = 0; order < MOMENT_ORDERS; order++) {
                    sums[order] += power;
                    power *= value;
                }
            }
            output[i * STATISTICS_FIELDS] = nan ? NAN : low;
            output[i * STATISTICS_FIELDS + 1] = nan ? NAN : high;
            for (uint order = 0; order < MOMENT_ORDERS; order++)
                output[i * STATISTICS_FIELDS + 2 + order] = sums[order];
            output[i * STATISTICS_FIELDS + STATISTICS_FIELDS - 1] = as_type<float>(last - first);
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
    uint band = i % c, x = (i / c) % dw, y = i / (c * dw);
    if (op == 30) {
        float value = float(src[i]);
        value = isnan(value) ? 0.0f : clamp(value, 0.0f, 1.0f);
        dst[i] = T(floorf(value * v[0] + 0.5f) / v[0]);
        return;
    }
    if (op == 28) {
        float mask = float(other[i / c]) / v[0];
        mask = isnan(mask) ? 0.0f : clamp(mask, 0.0f, 1.0f);
        if (p[7] == 1)
            mask = 1.0f - mask;
        dst[i] = mask == 0 ? T(0) : T(float(src[i]) * mask);
        return;
    }
    if (op == 26 || op == 27) {
        float px = float(x) + 0.5f, py = float(y) + 0.5f;
        float denominator = v[6] * px + v[7] * py + 1;
        float sx = (v[0] * px + v[1] * py + v[2]) / denominator;
        float sy = (v[3] * px + v[4] * py + v[5]) / denominator;
        if (op == 27) {
            const float *coordinates = reinterpret_cast<const float *>(other);
            sx = coordinates[(i / c) * 3];
            sy = coordinates[(i / c) * 3 + 1];
        }
        if (!isfinite(sx) || !isfinite(sy) || sx < 0 || sy < 0 || sx >= sw || sy >= sh) {
            dst[i] = store_sample<T>(v[8 + band]);
            return;
        }
        if (k == 0) {
            dst[i] = src[(uint(sy) * sw + uint(sx)) * c + band];
            return;
        }
        sx -= 0.5f;
        sy -= 0.5f;
        int ix = int(floorf(sx)), iy = int(floorf(sy));
        float fx = sx - floorf(sx), fy = sy - floorf(sy), result = 0;
        if (k == 2) {
            vec4<float> rows;
            for (int ky = -1; ky <= 2; ky++) {
                vec4<float> taps;
                for (int kx = -1; kx <= 2; kx++) {
                    uint index = (uint(clamp(iy + ky, 0, int(sh) - 1)) * sw + uint(clamp(ix + kx, 0, int(sw) - 1))) * c + band;
                    taps[kx + 1] = float(src[index]);
                }
                rows[ky + 1] = warp_cubic(taps, fx);
            }
            result = warp_cubic(rows, fy);
            dst[i] = store_sample<T>(is_same_v<T, uchar> ? floorf(result) : result);
            return;
        }
        int first = 0, last = 1;
        for (int ky = first; ky <= last; ky++) {
            float wy = ky == 0 ? 1 - fy : fy;
            float row = 0;
            for (int kx = first; kx <= last; kx++) {
                float wx = kx == 0 ? 1 - fx : fx;
                uint index = (uint(clamp(iy + ky, 0, int(sh) - 1)) * sw + uint(clamp(ix + kx, 0, int(sw) - 1))) * c + band;
                row += float(src[index]) * wx;
            }
            result += row * wy;
        }
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
            float sum = original * 4;
            for (uint ky = 0; ky < 3; ky++)
                for (uint kx = 0; kx < 3; kx++) {
                    sum += float(src[((y + ky - 1) * sw + x + kx - 1) * c + band]);
                }
            base = is_same_v<T, uchar> ? float((uint(sum) + 6) / 13) : sum / 13;
        }
        float value = v[0] == 0 ? base : fmaf(v[0], original - base, base);
        if (is_same_v<T, uchar>)
            value = floorf(value);
        dst[i] = store_sample<T>(value);
        return;
    }
    if (op == 16 || op == 17) {
        dst[i] = op == 16 ? (band == k ? other[i / c] : src[i]) : (band == c - 1 ? store_sample<T>(v[0]) : src[i]);
        return;
    }
    if (op == 18) {
        int64 pos = p[7] == 0 ? x : y, last = (p[7] == 0 ? sw : sh) - 1;
        int64 low = max(0l, pos - int64(k)), high = min(last, pos + int64(k));
        uint repeat_low = uint(max(0l, int64(k) - pos)), repeat_high = uint(max(0l, pos + int64(k) - last));
        uint first = p[7] == 0 ? y * sw * c + band : x * c + band;
        uint stride = p[7] == 0 ? c : sw * c;
        float sum = 0;
        uint integer_sum = 0;
        for (int64 t = low; t <= high; t++) {
            float value = read_sample(src, first + uint(t) * stride);
            if (is_same_v<T, uchar>)
                integer_sum += uint(value);
            else
                sum += value;
        }
        if (is_same_v<T, uchar>) {
            integer_sum += uint(src[first]) * repeat_low + uint(src[first + uint(last) * stride]) * repeat_high;
        } else {
            if (repeat_low)
                sum += float(src[first]) * float(repeat_low);
            if (repeat_high)
                sum += float(src[first + uint(last) * stride]) * float(repeat_high);
        }
        uint left = first + uint(clamp(pos - int64(k) - 1, 0l, last)) * stride;
        uint right = first + uint(clamp(pos + int64(k) + 1, 0l, last)) * stride;
        if (is_same_v<T, uchar>) {
            uint fixed = integer_sum * uint(v[0]) + (uint(src[left]) + uint(src[right])) * uint(v[1]) + BOX_ROUND;
            dst[i] = T(fixed >> BOX_BITS);
        } else {
            float value = v[0] == 0 ? 0 : sum * (v[0] / float(BOX_SCALE));
            if (v[1] != 0)
                value += (float(src[left]) + float(src[right])) * (v[1] / float(BOX_SCALE));
            dst[i] = T(value);
        }
        return;
    }
    if (op == 19) {
        float samples[MAX_RANK_SAMPLES];
        uint count = 0, rank = p[7];
        float extreme = 0;
        for (uint ky = 0; ky < k; ky++) {
            uint sy = uint(clamp(int64(y) + ky - k / 2, 0l, int64(sh) - 1));
            for (uint kx = 0; kx < k; kx++) {
                uint sx = uint(clamp(int64(x) + kx - k / 2, 0l, int64(sw) - 1));
                float value = read_sample(src, (sy * sw + sx) * c + band);
                if (isnan(value)) {
                    dst[i] = T(value);
                    return;
                }
                if (rank == 0 || rank == k * k - 1) {
                    extreme = count == 0 ? value : (rank == 0 ? min(extreme, value) : max(extreme, value));
                } else {
                    // Values above the requested rank cannot affect the result.
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
                }
                count++;
            }
        }
        dst[i] = T((rank == 0 || rank == k * k - 1) ? extreme : samples[rank]);
        return;
    }
    if (op == 7) {
        int64 sx = int64(x) + int(k), sy = int64(y) + int(p[7]);
        dst[i] = (sx >= 0 && sy >= 0 && sx < sw && sy < sh) ? src[(uint(sy) * sw + uint(sx)) * c + band] : T(0);
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
        uint pos = p[7] == 0 ? x : y;
        uint base = pos * k, start = uint(v[base]), count = uint(v[base + 1]);
        out = 0;
        int fixed = RESIZE_ROUND;
        uint first = p[7] == 0 ? (y * sw + start) * c + band : (start * sw + x) * c + band;
        uint stride = p[7] == 0 ? c : sw * c;
        for (uint t = 0; t < count; t++) {
            uint idx = first + t * stride;
            if (is_same_v<T, uchar>)
                fixed += int(src[idx]) * int(roundf(v[base + 2 + t] * float(RESIZE_SCALE)));
            else
                out += read_sample(src, idx) * v[base + 2 + t];
        }
        if (is_same_v<T, uchar>)
            out = float(fixed >> RESIZE_BITS);
    } else if (op == 1) {
        uint r = k / 2;
        if (x >= r && y >= r && x + r < sw && y + r < sh) {
            out = v[1];
            for (uint ky = 0; ky < k; ky++) {
                float subtotal = 0;
                for (uint kx = 0; kx < k; kx++)
                    subtotal += read_sample(src, ((y + r - ky) * sw + x + kx - r) * c + band) * v[2 + ky * k + kx];
                out += subtotal;
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
        float alpha = sa + da * (1 - sa);
        out = band == c - 1 ? alpha * maxval
                            : (alpha == 0 ? 0 : ((sa == 0 ? 0 : read_sample(other, i) * sa) + (da == 0 ? 0 : a * da) * (1 - sa)) / alpha);
    } else if (op == 4 || op == 5) {
        float raw = read_sample(src, i - band + c - 1);
        float alpha = isnan(raw) ? 0.0f : clamp(raw / maxval, 0.0f, 1.0f);
        out = band == c - 1 ? alpha * maxval
                            : (alpha == 0 ? ((op == 5 && is_same_v<T, uchar>) ? a : 0.0f)
                                          : (op == 4 ? a * alpha : (is_same_v<T, uchar> ? a * maxval / raw : a / alpha)));
        if (op == 5 && is_same_v<T, uchar>)
            out = floorf(out);
    } else if (op == 10) {
        if (!(c == 4 && band == c - 1)) {
            out = a * v[0];
            if (is_same_v<T, uchar>)
                out = floorf(out);
        }
    } else if (op == 11) {
        out = maxval - a;
    } else if (op == 12) {
        out = v[band * 256 + uint(a)];
    } else if (op == 20) {
        float difference = a - read_sample(other, i);
        if (abs(difference) > v[0]) {
            if (is_same_v<T, uchar>)
                out = float(int64(a) + int64(difference) * int(k) / 100);
            else
                out = a + difference * float(int(k)) / 100.0f;
        }
    } else if (op == 13) {
        float b = read_sample(other, i);
        switch (k) {
        case 0:
            out = abs(a - b);
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
        if (is_same_v<T, uchar>) {
            if (k == 1)
                out = float(uint(a) * uint(b) / 255);
            else if (k == 2)
                out = float(255 - (255 - uint(a)) * (255 - uint(b)) / 255);
            else if (k >= 5) {
                if (!isfinite(out) || out >= 2147483648.0f || out < -2147483648.0f)
                    out = 0;
                out = floorf(out);
            }
        }
    }
    dst[i] = store_sample<T>(out);
}

template <typename S, typename D> __device__ void convert_image(const S *src, D *dst, const uint *p, uint i) {
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

extern "C" __global__ void finish_statistics(const float *src, float *dst, const uint *p) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x, blocks = p[7];
    if (p[6] == 1) {
        if (i >= p[2] * HISTOGRAM_BINS)
            return;
        uint band = i / HISTOGRAM_BINS, bin = i % HISTOGRAM_BINS, count = 0;
        for (uint block = 0; block < blocks; block++)
            count += as_type<uint>(src[(band * blocks + block) * HISTOGRAM_BINS + bin]);
        dst[i] = as_type<float>(count);
        return;
    }
    if (i >= p[2])
        return;
    float low = INFINITY, high = -INFINITY, sums[MOMENT_ORDERS] = {0};
    bool nan = false;
    for (uint block = 0; block < blocks; block++) {
        const float *partial = src + (i * blocks + block) * STATISTICS_FIELDS;
        nan = nan || isnan(partial[0]);
        low = min(low, partial[0]);
        high = max(high, partial[1]);
        for (uint order = 0; order < MOMENT_ORDERS; order++)
            sums[order] += partial[2 + order];
    }
    float *result = dst + i * STATISTICS_FIELDS;
    result[0] = nan ? NAN : low;
    result[1] = nan ? NAN : high;
    for (uint order = 0; order < MOMENT_ORDERS; order++)
        result[2 + order] = sums[order];
    result[STATISTICS_FIELDS - 1] = as_type<float>(p[0] * p[1]);
}

extern "C" __global__ void image_u8(const uchar *src, uchar *dst, const uchar *other, const uint *p, const float *v) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    process(src, dst, other, p, v, i);
}
extern "C" __global__ void image_f32(const float *src, float *dst, const float *other, const uint *p, const float *v) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    process(src, dst, other, p, v, i);
}

template <typename T> __device__ void copy_tile(const T *src, T *dst, const uint *p, uint i) {
    uint channels = p[2], width = p[9], height = p[10];
    if (i >= width * height * channels)
        return;
    uint band = i % channels, x = (i / channels) % width, y = i / (channels * width);
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
