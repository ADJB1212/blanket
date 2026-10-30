#include <metal_stdlib>
using namespace metal;

#ifdef BLANKET_METAL_SPECIALIZE
// Specialization removes unrelated operations and their register requirements.
constant uint SPECIALIZED_OPERATION [[function_constant(0)]];
constant uint SPECIALIZED_CHANNELS [[function_constant(1)]];
#endif

// Match CPU luminance conversion and fractional box-filter precision.
constant uint LUMA_RED = 19595, LUMA_GREEN = 38470, LUMA_BLUE = 7471, LUMA_BITS = 16;
constant uint BOX_BITS = 24, BOX_SCALE = 1u << BOX_BITS, BOX_ROUND = BOX_SCALE / 2;
// Rank validation limits each neighborhood to 7x7 samples.
constant uint MAX_RANK_SAMPLES = 49;
// Keep reduction blocks bounded and byte luminance sums exact across all passes.
constant uint REDUCTION_BLOCK_SIZE = 256;
// CPU 3D LUTs use signed samples with 6 fractional bits and 15-bit interpolation.
constant uint LUT_SAMPLE_BITS = 6, LUT_POSITION_BITS = 18, LUT_BLEND_BITS = 15;
constant int LUT_BLEND_SCALE = 1 << LUT_BLEND_BITS;
constant uint LUT_HEADER_LEN = 6;
// Statistics store extrema, four raw power sums, and the exact pixel count.
constant uint STATISTICS_FIELDS = 7, MOMENT_ORDERS = 4, HISTOGRAM_BINS = 256;
// Resize coefficients use the CPU kernel's 22-bit fixed-point scale.
constant uint RESIZE_BITS = 22, RESIZE_SCALE = 1u << RESIZE_BITS, RESIZE_ROUND = RESIZE_SCALE / 2;
constant uint BLUR_OUTPUTS = 16, SLIDING_BLUR_RADIUS = 32;

// Byte sums can slide exactly, including modular overflow at extreme radii.
template <typename T, typename S>
void sliding_box_blur(device const T *src, device T *dst, constant uint *p, device const float *v, uint i, uint channels) {
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

template <typename T> float luminance(device const T *src, uint base, uint channels) {
    if (channels < 3)
        return float(src[base]);
    if (is_same_v<T, uchar>) {
        return float((uint(src[base]) * LUMA_RED + uint(src[base + 1]) * LUMA_GREEN + uint(src[base + 2]) * LUMA_BLUE + (1u << (LUMA_BITS - 1))) >>
                     LUMA_BITS);
    }
    return dot(float3(src[base], src[base + 1], src[base + 2]), float3(0.299f, 0.587f, 0.114f));
}

float color_transfer(float value, bool decode) {
    float magnitude = abs(value), result;
    if (decode)
        result = magnitude <= 0.04045f ? magnitude / 12.92f : pow((magnitude + 0.055f) / 1.055f, 2.4f);
    else
        result = magnitude <= 0.0031308f ? magnitude * 12.92f : 1.055f * pow(magnitude, 1.0f / 2.4f) - 0.055f;
    return copysign(result, value);
}

int lut_mix(int a, int b, int shift) {
    return (a * (LUT_BLEND_SCALE - shift) + b * shift) >> LUT_BLEND_BITS;
}

float warp_cubic(float4 v, float t) {
    float a = -v[0] + v[1] - v[2] + v[3];
    float b = 2 * v[0] - 2 * v[1] + v[2] - v[3];
    float c = -v[0] + v[2];
    return ((a * t + b) * t + c) * t + v[1];
}

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

template <typename T> void reduce_luminance(device const T *src, device float *dst, constant uint *p, uint i) {
    if (i % 4 != 0 || i >= p[3] * 4)
        return;
    uint block = i / 4, first = block * REDUCTION_BLOCK_SIZE;
    uint last = min(first + REDUCTION_BLOCK_SIZE, p[0] * p[1]), stage = p[6];
    ulong integer_sum = 0;
    float sum = 0;
    for (uint pixel = first; pixel < last; pixel++) {
        if (stage == 0)
            integer_sum += ulong(luminance(src, pixel * p[2], p[2]));
        else if (stage == 1)
            sum += luminance(src, pixel * p[2], p[2]);
        else if (stage == 2) {
            device const uint *words = reinterpret_cast<device const uint *>(src);
            integer_sum += ulong(words[pixel * 4]) | (ulong(words[pixel * 4 + 1]) << 32);
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

template <typename T> float read_sample(device const T *p, uint i) {
    return float(p[i]);
}
template <typename T> T store_sample(float v) {
    return T(v);
}
template <> uchar store_sample<uchar>(float v) {
    return uchar(clamp(floor(v + 0.5f), 0.0f, 255.0f));
}

template <typename T> void process(device const T *src, device T *dst, device const T *other, constant uint *p, device const float *v, uint i) {
    uint sw = p[0], sh = p[1], c = p[2], dw = p[3], dh = p[4], op = p[5], k = p[6];
#ifdef BLANKET_METAL_SPECIALIZE
    op = SPECIALIZED_OPERATION;
    c = SPECIALIZED_CHANNELS;
#endif
    if (op == 18 && is_same_v<T, uchar> && k >= SLIDING_BLUR_RADIUS) {
        sliding_box_blur<T, uint>(src, dst, p, v, i, c);
        return;
    }
    if (op == 29) {
        if (i >= c)
            return;
        device float *output = reinterpret_cast<device float *>(dst);
        if (k == 1) {
            uint counts[HISTOGRAM_BINS] = {0};
            for (uint pixel = 0; pixel < sw * sh; pixel++)
                counts[uint(src[pixel * c + i])]++;
            for (uint bin = 0; bin < HISTOGRAM_BINS; bin++)
                output[i * HISTOGRAM_BINS + bin] = as_type<float>(counts[bin]);
        } else {
            float low = INFINITY, high = -INFINITY;
            float sums[MOMENT_ORDERS] = {0};
            bool nan = false;
            for (uint pixel = 0; pixel < sw * sh; pixel++) {
                float value = float(src[pixel * c + i]);
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
            output[i * STATISTICS_FIELDS + STATISTICS_FIELDS - 1] = as_type<float>(sw * sh);
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
        dst[i] = T(floor(value * v[0] + 0.5f) / v[0]);
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
            device const float *coordinates = reinterpret_cast<device const float *>(other);
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
        int ix = int(floor(sx)), iy = int(floor(sy));
        float fx = sx - floor(sx), fy = sy - floor(sy), result = 0;
        if (k == 2) {
            float4 rows;
            for (int ky = -1; ky <= 2; ky++) {
                float4 taps;
                for (int kx = -1; kx <= 2; kx++) {
                    uint index = (uint(clamp(iy + ky, 0, int(sh) - 1)) * sw + uint(clamp(ix + kx, 0, int(sw) - 1))) * c + band;
                    taps[kx + 1] = float(src[index]);
                }
                rows[ky + 1] = warp_cubic(taps, fx);
            }
            result = warp_cubic(rows, fy);
            dst[i] = store_sample<T>(is_same_v<T, uchar> ? floor(result) : result);
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
        dst[i] = store_sample<T>(is_same_v<T, uchar> ? floor(result) : result);
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
                device const uint *words = reinterpret_cast<device const uint *>(other);
                ulong total = ulong(words[0]) | (ulong(words[1]) << 32);
                base = float((total * 2 + p[7]) / (ulong(p[7]) * 2));
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
        float value = v[0] == 0 ? base : fma(v[0], original - base, base);
        if (is_same_v<T, uchar>)
            value = floor(value);
        dst[i] = store_sample<T>(value);
        return;
    }
    if (op == 16 || op == 17) {
        dst[i] = op == 16 ? (band == k ? other[i / c] : src[i]) : (band == c - 1 ? store_sample<T>(v[0]) : src[i]);
        return;
    }
    if (op == 18) {
        long pos = p[7] == 0 ? x : y, last = (p[7] == 0 ? sw : sh) - 1;
        long low = max(0l, pos - long(k)), high = min(last, pos + long(k));
        uint repeat_low = uint(max(0l, long(k) - pos)), repeat_high = uint(max(0l, pos + long(k) - last));
        uint first = p[7] == 0 ? y * sw * c + band : x * c + band;
        uint stride = p[7] == 0 ? c : sw * c;
        float sum = 0;
        uint integer_sum = 0;
        for (long t = low; t <= high; t++) {
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
        uint left = first + uint(clamp(pos - long(k) - 1, 0l, last)) * stride;
        uint right = first + uint(clamp(pos + long(k) + 1, 0l, last)) * stride;
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
            uint sy = uint(clamp(long(y) + ky - k / 2, 0l, long(sh) - 1));
            for (uint kx = 0; kx < k; kx++) {
                uint sx = uint(clamp(long(x) + kx - k / 2, 0l, long(sw) - 1));
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
        long sx = long(x) + int(k), sy = long(y) + int(p[7]);
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
        uint sx = x >= k ? x - k : x + sw - k, sy = y >= p[7] ? y - p[7] : y + sh - p[7];
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
                fixed += int(src[idx]) * int(round(v[base + 2 + t] * float(RESIZE_SCALE)));
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
            out = floor(out);
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
            out = floor(out);
    } else if (op == 10) {
        if (!(c == 4 && band == c - 1)) {
            out = a * v[0];
            if (is_same_v<T, uchar>)
                out = floor(out);
        }
    } else if (op == 11) {
        out = maxval - a;
    } else if (op == 12) {
        out = v[band * 256 + uint(a)];
    } else if (op == 20) {
        float difference = a - read_sample(other, i);
        if (abs(difference) > v[0]) {
            if (is_same_v<T, uchar>)
                out = float(long(a) + long(difference) * int(k) / 100);
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
                out = floor(out);
            }
        }
    }
    dst[i] = store_sample<T>(out);
}

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

#define REDUCTION_KERNEL(NAME, SRC)                                                                                                                  \
    kernel void NAME(device const SRC *s [[buffer(0)]],                                                                                              \
                     device float *d [[buffer(1)]],                                                                                                  \
                     device const SRC *o [[buffer(2)]],                                                                                              \
                     constant uint *p [[buffer(3)]],                                                                                                 \
                     device const float *v [[buffer(4)]],                                                                                            \
                     uint i [[thread_position_in_grid]]) {                                                                                           \
        reduce_luminance(s, d, p, i);                                                                                                                \
    }
REDUCTION_KERNEL(reduce_u8, uchar)
REDUCTION_KERNEL(reduce_f32, float)

kernel void image_u8(device const uchar *s [[buffer(0)]], device uchar *d [[buffer(1)]], device const uchar *o [[buffer(2)]],
                     constant uint *p [[buffer(3)]], device const float *v [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    process(s, d, o, p, v, i);
}
kernel void image_f32(device const float *s [[buffer(0)]], device float *d [[buffer(1)]], device const float *o [[buffer(2)]],
                      constant uint *p [[buffer(3)]], device const float *v [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    process(s, d, o, p, v, i);
}

#ifdef BLANKET_METAL_SPECIALIZE
uchar4 rank_samples(device const uchar *src, uint first, uint width, uint height, int dx, int dy) {
    uint c = SPECIALIZED_CHANNELS;
    uint row = width * c, y = first / row;
    uint x = (first % row) / c, last = ((first + 3) % row) / c;
    if (first / row == (first + 3) / row && int(x) + dx >= 0 && int(last) + dx < int(width) && int(y) + dy >= 0 && int(y) + dy < int(height))
        return uchar4(*reinterpret_cast<device const packed_uchar4 *>(src + int(first) + dy * int(row) + dx * int(c)));
    uchar4 result;
    for (uint lane = 0; lane < 4; lane++) {
        uint sample = first + lane;
        uint sx = uint(clamp(int(sample / c % width) + dx, 0, int(width) - 1));
        uint sy = uint(clamp(int(sample / row) + dy, 0, int(height) - 1));
        result[lane] = src[(sy * width + sx) * c + sample % c];
    }
    return result;
}

void rank_swap(thread uchar4 &a, thread uchar4 &b) {
    uchar4 low = min(a, b);
    b = max(a, b);
    a = low;
}

uchar4 median9(thread uchar4 *s) {
    rank_swap(s[1], s[2]);
    rank_swap(s[4], s[5]);
    rank_swap(s[7], s[8]);
    rank_swap(s[0], s[1]);
    rank_swap(s[3], s[4]);
    rank_swap(s[6], s[7]);
    rank_swap(s[1], s[2]);
    rank_swap(s[4], s[5]);
    rank_swap(s[7], s[8]);
    rank_swap(s[0], s[3]);
    rank_swap(s[5], s[8]);
    rank_swap(s[4], s[7]);
    rank_swap(s[3], s[6]);
    rank_swap(s[1], s[4]);
    rank_swap(s[2], s[5]);
    rank_swap(s[4], s[7]);
    rank_swap(s[4], s[2]);
    rank_swap(s[6], s[4]);
    rank_swap(s[4], s[2]);
    return s[4];
}

template <typename P, typename I, typename F>
void filter_pixels(device const P *src, device P *dst, constant uint *p, device const float *v, uint i) {
    if (i >= p[3] * p[4])
        return;
    uint x = i % p[3], y = i / p[3];
    if (SPECIALIZED_OPERATION == 0) {
        uint base = (p[7] == 0 ? x : y) * p[6];
        uint start = uint(v[base]), taps = uint(v[base + 1]);
        uint source = p[7] == 0 ? y * p[0] + start : start * p[0] + x;
        uint stride = p[7] == 0 ? 1 : p[0];
        I fixed = I(RESIZE_ROUND);
        for (uint t = 0; t < taps; t++)
            fixed += I(src[source + t * stride]) * int(round(v[base + 2 + t] * float(RESIZE_SCALE)));
        dst[i] = P(clamp(F(fixed >> RESIZE_BITS), 0.0f, 255.0f));
    } else {
        uint size = p[6], radius = size / 2;
        P result = src[i];
        if (x >= radius && y >= radius && x + radius < p[0] && y + radius < p[1]) {
            F sum = F(v[1]);
            for (uint ky = 0; ky < size; ky++) {
                F subtotal = F(0);
                for (uint kx = 0; kx < size; kx++)
                    subtotal += F(src[(y + radius - ky) * p[0] + x + kx - radius]) * v[2 + ky * size + kx];
                sum += subtotal;
            }
            result = P(clamp(floor(sum + 0.5f), 0.0f, 255.0f));
        }
        dst[i] = result;
    }
}

template <typename P, typename U> void box_pixels(device const P *src, device P *dst, constant uint *p, device const float *v, uint i) {
    if (i >= p[3] * p[4])
        return;
    uint x = i % p[3], y = i / p[3], radius = p[6];
    long pos = p[7] == 0 ? x : y, last = (p[7] == 0 ? p[0] : p[1]) - 1;
    long low = max(0l, pos - long(radius)), high = min(last, pos + long(radius));
    uint repeat_low = uint(max(0l, long(radius) - pos)), repeat_high = uint(max(0l, pos + long(radius) - last));
    uint source = p[7] == 0 ? y * p[0] : x, stride = p[7] == 0 ? 1 : p[0];
    U sum = U(0);
    for (long t = low; t <= high; t++)
        sum += U(src[source + uint(t) * stride]);
    sum += U(src[source]) * repeat_low + U(src[source + uint(last) * stride]) * repeat_high;
    uint left = source + uint(clamp(pos - long(radius) - 1, 0l, last)) * stride;
    uint right = source + uint(clamp(pos + long(radius) + 1, 0l, last)) * stride;
    U fixed = sum * uint(v[0]) + (U(src[left]) + U(src[right])) * uint(v[1]) + BOX_ROUND;
    dst[i] = P(fixed >> BOX_BITS);
}

kernel void color_u8(device const uchar *s [[buffer(0)]], device uchar *d [[buffer(1)]], constant uint *p [[buffer(3)]],
                     device const float *v [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (SPECIALIZED_CHANNELS == 3) {
        if (SPECIALIZED_OPERATION == 18)
            box_pixels<packed_uchar3, uint3>(reinterpret_cast<device const packed_uchar3 *>(s), reinterpret_cast<device packed_uchar3 *>(d), p, v, i);
        else
            filter_pixels<packed_uchar3, int3, float3>(
                reinterpret_cast<device const packed_uchar3 *>(s), reinterpret_cast<device packed_uchar3 *>(d), p, v, i);
    } else {
        if (SPECIALIZED_OPERATION == 18)
            box_pixels<uchar2, uint2>(reinterpret_cast<device const uchar2 *>(s), reinterpret_cast<device uchar2 *>(d), p, v, i);
        else
            filter_pixels<uchar2, int2, float2>(reinterpret_cast<device const uchar2 *>(s), reinterpret_cast<device uchar2 *>(d), p, v, i);
    }
}

// Full vectors avoid byte stores and per-sample coordinate calculations.
kernel void packed_u8(device const uchar *s [[buffer(0)]], device uchar *d [[buffer(1)]], device const uchar *o [[buffer(2)]],
                      constant uint *p [[buffer(3)]], device const float *v [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    constexpr uint PACKED_SAMPLES = 4;
    uint first = i * PACKED_SAMPLES, count = p[3] * p[4] * p[2];
    if (first + PACKED_SAMPLES > count) {
        for (uint sample = first; sample < count; sample++)
            process(s, d, o, p, v, sample);
        return;
    }
    if (SPECIALIZED_OPERATION == 19) {
        uint size = p[6], rank = p[7];
        uchar4 result;
        if (rank == 0 || rank == size * size - 1) {
            result = uchar4(rank == 0 ? 255 : 0);
            for (uint y = 0; y < size; y++)
                for (uint x = 0; x < size; x++) {
                    uchar4 value = rank_samples(s, first, p[0], p[1], int(x) - int(size / 2), int(y) - int(size / 2));
                    result = rank == 0 ? min(result, value) : max(result, value);
                }
        } else if (size == 3) {
            uchar4 samples[9];
#pragma unroll
            for (uint y = 0; y < 3; y++) {
#pragma unroll
                for (uint x = 0; x < 3; x++)
                    samples[y * 3 + x] = rank_samples(s, first, p[0], p[1], int(x) - 1, int(y) - 1);
            }
            if (rank == 4) {
                result = median9(samples);
            } else {
#pragma unroll
                for (uint phase = 0; phase < 9; phase++) {
#pragma unroll
                    for (uint j = phase % 2; j < 8; j += 2) {
                        rank_swap(samples[j], samples[j + 1]);
                    }
                }
                result = samples[rank];
            }
        } else {
            for (uint lane = 0; lane < PACKED_SAMPLES; lane++)
                process(s, d, o, p, v, first + lane);
            return;
        }
        reinterpret_cast<device uchar4 *>(d)[i] = result;
        return;
    }
    if (SPECIALIZED_OPERATION == 0) {
        filter_pixels<uchar4, int4, float4>(reinterpret_cast<device const uchar4 *>(s), reinterpret_cast<device uchar4 *>(d), p, v, i);
        return;
    }
    if (SPECIALIZED_OPERATION == 18) {
        if (SPECIALIZED_CHANNELS != 4) {
            int radius = int(p[6]);
            bool horizontal = p[7] == 0;
            uint4 sum = uint4(0);
            for (int delta = -radius; delta <= radius; delta++)
                sum += uint4(rank_samples(s, first, p[0], p[1], horizontal ? delta : 0, horizontal ? 0 : delta));
            uint4 fixed = sum * uint(v[0]) + BOX_ROUND;
            if (v[1] != 0) {
                uint4 left = uint4(rank_samples(s, first, p[0], p[1], horizontal ? -radius - 1 : 0, horizontal ? 0 : -radius - 1));
                uint4 right = uint4(rank_samples(s, first, p[0], p[1], horizontal ? radius + 1 : 0, horizontal ? 0 : radius + 1));
                fixed += (left + right) * uint(v[1]);
            }
            reinterpret_cast<device uchar4 *>(d)[i] = uchar4(fixed >> BOX_BITS);
            return;
        }
        box_pixels<uchar4, uint4>(reinterpret_cast<device const uchar4 *>(s), reinterpret_cast<device uchar4 *>(d), p, v, i);
        return;
    }
    if (SPECIALIZED_OPERATION == 1) {
        filter_pixels<uchar4, int4, float4>(reinterpret_cast<device const uchar4 *>(s), reinterpret_cast<device uchar4 *>(d), p, v, i);
        return;
    }
    uchar4 bytes = reinterpret_cast<device const uchar4 *>(s)[i];
    if (SPECIALIZED_OPERATION == 12) {
        uchar4 result;
        for (uint lane = 0; lane < PACKED_SAMPLES; lane++)
            result[lane] = uchar(v[((first + lane) % SPECIALIZED_CHANNELS) * 256 + uint(bytes[lane])]);
        reinterpret_cast<device uchar4 *>(d)[i] = result;
        return;
    }
    if (SPECIALIZED_OPERATION == 13) {
        uint4 a = uint4(bytes), b = uint4(reinterpret_cast<device const uchar4 *>(o)[i]);
        uint kind = p[6];
        float4 result;
        switch (kind) {
        case 0:
            result = float4(max(a, b) - min(a, b));
            break;
        case 1:
            result = float4(a * b / 255);
            break;
        case 2:
            result = float4(255 - (255 - a) * (255 - b) / 255);
            break;
        case 3:
            result = float4(max(a, b));
            break;
        case 4:
            result = float4(min(a, b));
            break;
        default:
            result = (kind == 5 ? float4(a + b) : float4(int4(a) - int4(b))) / v[0] + v[1];
            for (uint lane = 0; lane < PACKED_SAMPLES; lane++)
                if (!isfinite(result[lane]) || result[lane] >= 2147483648.0f || result[lane] < -2147483648.0f)
                    result[lane] = 0;
            result = floor(result);
        }
        reinterpret_cast<device uchar4 *>(d)[i] = uchar4(clamp(result, 0.0f, 255.0f));
        return;
    }
    if (SPECIALIZED_OPERATION == 6) {
        reinterpret_cast<device uchar4 *>(d)[i] = bytes;
        return;
    }
    if (SPECIALIZED_OPERATION == 11) {
        reinterpret_cast<device uchar4 *>(d)[i] = uchar4(255) - bytes;
        return;
    }
    float4 a = float4(bytes), result;
    if (SPECIALIZED_OPERATION == 2) {
        float4 b = float4(reinterpret_cast<device const uchar4 *>(o)[i]);
        result = a + (b - a) * v[0];
    } else {
        result = a * v[0];
        if (SPECIALIZED_CHANNELS == 4)
            result.w = a.w;
    }
    reinterpret_cast<device uchar4 *>(d)[i] = uchar4(clamp(floor(result), 0.0f, 255.0f));
}
#endif
