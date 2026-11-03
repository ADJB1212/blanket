// Scalar filters and packed byte-filter helpers.

// Blur consecutive byte samples with an exact sliding-window sum.
template <typename T, typename S>
void sliding_box_blur(device const T *src, device T *dst, constant uint *p, device const float *v, uint i, uint channels) {
    uint width = p[0], height = p[1], radius = p[6];
    bool horizontal = p[7] == 0;
    uint length = horizontal ? width : height, lines = horizontal ? height : width;
    uint blocks = (length + BLUR_OUTPUTS - 1) >> 4; // BLUR_OUTPUTS == 16
    if (i >= blocks * lines * channels)
        return;
    uint lane, band;
    decompose_channel(i, channels, lane, band);
    uint line, block_idx;
    if (horizontal) {
        line = lane / blocks;
        block_idx = lane - line * blocks;
    } else {
        block_idx = lane / lines;
        line = lane - block_idx * lines;
    }
    uint position = block_idx << 4;
    uint last = length - 1;
    uint base = horizontal ? line * width * channels + band : line * channels + band;
    uint stride = horizontal ? channels : width * channels;
    uint low = position > radius ? position - radius : 0, high = min(last, position + radius);

    S sum = S(0);
    for (uint t = low; t <= high; t++)
        sum += S(src[base + t * stride]);
    if (radius > position)
        sum += S(src[base]) * (radius - position);
    if (position + radius > last)
        sum += S(src[base + last * stride]) * (position + radius - last);

    uint w0 = uint(v[0]), w1 = uint(v[1]);
    uint end = min(position + BLUR_OUTPUTS, length);
    for (; position < end; position++) {
        uint left = base + (position > radius ? position - radius - 1 : 0) * stride;
        uint right = base + min(position + radius + 1, last) * stride;
        S right_val = S(src[right]);
        S fixed = sum * w0 + (S(src[left]) + right_val) * w1 + BOX_ROUND;
        dst[base + position * stride] = T(fixed >> BOX_BITS);
        uint outgoing = base + (position > radius ? position - radius : 0) * stride;
        sum = sum - S(src[outgoing]) + right_val;
    }
}

// Apply a fractional box window to one sample with replicated borders.
template <typename T> void box_blur_sample(device const T *src, device T *dst, constant uint *p, device const float *v, uint i, uint c) {
    uint sw = p[0], sh = p[1], dw = p[3], k = p[6];
    uint band, x, y;
    decompose_3d(i, c, dw, band, x, y);
    bool horiz = (p[7] == 0);
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
            sum += read_sample(src, first + uint(t) * stride);
        if (repeat_low)
            sum += float(src[first]) * float(repeat_low);
        if (repeat_high)
            sum += float(src[first + uint(last) * stride]) * float(repeat_high);
        float w0 = v[0], w1 = v[1];
        float value = (w0 == 0.0f) ? 0.0f : sum * (w0 * (1.0f / float(BOX_SCALE)));
        if (w1 != 0.0f)
            value += (float(src[left]) + float(src[right])) * (w1 * (1.0f / float(BOX_SCALE)));
        dst[i] = T(value);
    }
}

// Select a neighborhood rank while preserving NaN propagation.
template <typename T> void rank_sample(device const T *src, device T *dst, constant uint *p, uint i, uint c) {
    uint sw = p[0], sh = p[1], dw = p[3], k = p[6], rank = p[7];
    uint band, x, y;
    decompose_3d(i, c, dw, band, x, y);
    float samples[MAX_RANK_SAMPLES];
    uint count = 0, max_rank = k * k - 1;
    bool is_extreme = (rank == 0 || rank == max_rank);
    float extreme = 0.0f;
    int half_k = int(k >> 1), max_x = int(sw) - 1, max_y = int(sh) - 1;
    int base_y = int(y) - half_k, base_x = int(x) - half_k;

    for (uint ky = 0; ky < k; ky++) {
        uint row_offset = uint(clamp(base_y + int(ky), 0, max_y)) * sw;
        for (uint kx = 0; kx < k; kx++) {
            uint sx = uint(clamp(base_x + int(kx), 0, max_x));
            float value = read_sample(src, (row_offset + sx) * c + band);
            if (!is_same_v<T, uchar> && isnan(value)) {
                dst[i] = T(value);
                return;
            }
            if (is_extreme) {
                extreme = (count == 0) ? value : (rank == 0 ? min(extreme, value) : max(extreme, value));
            } else {
                if (count > rank && value >= samples[rank]) {
                    count++;
                    continue;
                }
                uint j = min(count, rank);
                while (j > 0 && samples[j - 1] > value) {
                    samples[j] = samples[j - 1];
                    j--;
                }
                samples[j] = value;
            }
            count++;
        }
    }
    dst[i] = T(is_extreme ? extreme : samples[rank]);
}

// Accumulate one resize sample in the original tap order and precision.
template <typename T> float resize_sample(device const T *src, constant uint *p, device const float *v, uint i, uint c) {
    uint sw = p[0], dw = p[3], k = p[6];
    uint band, x, y;
    decompose_3d(i, c, dw, band, x, y);
    bool horiz = (p[7] == 0);
    uint pos = horiz ? x : y;
    uint base = pos * k, start = uint(v[base]), count = uint(v[base + 1]);
    uint first = horiz ? (y * sw + start) * c + band : (start * sw + x) * c + band;
    uint stride = horiz ? c : sw * c;
    device const float *weights = v + base + 2;

    if (is_same_v<T, uchar>) {
        int fixed = RESIZE_ROUND;
        for (uint t = 0; t < count; t++)
            fixed += int(src[first + t * stride]) * int(round(weights[t] * float(RESIZE_SCALE)));
        return float(fixed >> RESIZE_BITS);
    } else {
        float out = 0.0f;
        for (uint t = 0; t < count; t++)
            out = fma(read_sample(src, first + t * stride), weights[t], out);
        return out;
    }
}

// Convolve one interior sample and copy samples along the image border.
template <typename T> float convolve_sample(device const T *src, constant uint *p, device const float *v, uint i, uint c) {
    uint sw = p[0], sh = p[1], dw = p[3], k = p[6];
    uint band, x, y;
    decompose_3d(i, c, dw, band, x, y);
    uint r = k >> 1;
    if (x >= r && y >= r && x + r < sw && y + r < sh) {
        float out = v[1];
        uint row_stride = sw * c;
        device const T *row_ptr = src + ((y + r) * sw + (x - r)) * c + band;
        device const float *k_ptr = v + 2;
        for (uint ky = 0; ky < k; ky++) {
            float subtotal = 0.0f;
            for (uint kx = 0; kx < k; kx++)
                subtotal = fma(float(row_ptr[kx * c]), k_ptr[kx], subtotal);
            out += subtotal;
            row_ptr -= row_stride;
            k_ptr += k;
        }
        return out;
    }
    return read_sample(src, i);
}

#ifdef BLANKET_METAL_SPECIALIZE
// Load four neighboring bytes, clamping each sample at image edges.
inline uchar4 rank_samples(device const uchar *src, uint first, uint width, uint height, int dx, int dy) {
    const uint c = SPECIALIZED_CHANNELS;
    uint row = width * c;
    uint y = first / row;
    uint rem = first - y * row;
    uint x = rem / c;
    uint last_rem = first + 3 - y * row;
    if (last_rem < row && int(x) + dx >= 0 && int(last_rem / c) + dx < int(width) && int(y) + dy >= 0 && int(y) + dy < int(height))
        return uchar4(*reinterpret_cast<device const packed_uchar4 *>(src + int(first) + dy * int(row) + dx * int(c)));
    uchar4 result;
    int max_w = int(width) - 1, max_h = int(height) - 1;
#pragma unroll
    for (uint lane = 0; lane < 4; lane++) {
        uint sample = first + lane;
        uint sy_raw, s_rem;
        decompose_2d(sample, row, s_rem, sy_raw);
        uint sx_raw, band;
        decompose_channel(s_rem, c, sx_raw, band);
        uint sx = uint(clamp(int(sx_raw) + dx, 0, max_w));
        uint sy = uint(clamp(int(sy_raw) + dy, 0, max_h));
        result[lane] = src[(sy * width + sx) * c + band];
    }
    return result;
}

// Sort two packed sample vectors independently by lane.
inline void rank_swap(thread uchar4 &a, thread uchar4 &b) {
    uchar4 low = min(a, b);
    b = max(a, b);
    a = low;
}

// Select the median of nine packed samples with a sorting network.
inline uchar4 median9(thread uchar4 *s) {
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

// Resize or convolve one packed pixel with CPU-compatible rounding.
template <typename P, typename I, typename F, bool Floating = false>
void filter_pixels(device const P *src, device P *dst, constant uint *p, device const float *v, uint i) {
    uint dw = p[3], dh = p[4];
    if (i >= dw * dh)
        return;
    uint x, y;
    decompose_2d(i, dw, x, y);
    uint sw = p[0];
    if (SPECIALIZED_OPERATION == OP_RESIZE) {
        bool horiz = (p[7] == 0);
        uint base = (horiz ? x : y) * p[6];
        uint start = uint(v[base]), taps = uint(v[base + 1]);
        uint source = horiz ? y * sw + start : start * sw + x;
        uint stride = horiz ? 1u : sw;
        device const float *weights = v + base + 2;
        if (Floating) {
            F sum = F(0);
            for (uint t = 0; t < taps; t++)
                sum = fma(F(src[source + t * stride]), weights[t], sum);
            dst[i] = P(sum);
        } else {
            I fixed = I(RESIZE_ROUND);
            for (uint t = 0; t < taps; t++)
                fixed += I(src[source + t * stride]) * int(round(weights[t] * float(RESIZE_SCALE)));
            dst[i] = P(clamp(F(fixed >> RESIZE_BITS), 0.0f, 255.0f));
        }
    } else {
        uint size = p[6], radius = size >> 1;
        if (x >= radius && y >= radius && x + radius < sw && y + radius < p[1]) {
            F sum = F(v[1]);
            device const P *row_ptr = src + (y + radius) * sw + (x - radius);
            device const float *k_ptr = v + 2;
            for (uint ky = 0; ky < size; ky++) {
                F subtotal = F(0);
                for (uint kx = 0; kx < size; kx++)
                    subtotal = fma(F(row_ptr[kx]), k_ptr[kx], subtotal);
                sum += subtotal;
                row_ptr -= sw;
                k_ptr += size;
            }
            dst[i] = Floating ? P(sum) : P(clamp(floor(sum + 0.5f), 0.0f, 255.0f));
        } else {
            dst[i] = src[i];
        }
    }
}

// Apply a fractional box window to one packed byte pixel.
template <typename P, typename U> void box_pixels(device const P *src, device P *dst, constant uint *p, device const float *v, uint i) {
    uint dw = p[3], dh = p[4];
    if (i >= dw * dh)
        return;
    uint x, y;
    decompose_2d(i, dw, x, y);
    int radius = int(p[6]);
    bool horiz = (p[7] == 0);
    int pos = int(horiz ? x : y), last = int(horiz ? p[0] : p[1]) - 1;
    int low = max(0, pos - radius), high = min(last, pos + radius);
    uint repeat_low = uint(max(0, radius - pos)), repeat_high = uint(max(0, pos + radius - last));
    uint source = horiz ? y * p[0] : x, stride = horiz ? 1u : p[0];
    U sum = U(0);
    for (int t = low; t <= high; t++)
        sum += U(src[source + uint(t) * stride]);
    sum += U(src[source]) * repeat_low + U(src[source + uint(last) * stride]) * repeat_high;
    uint left = source + uint(clamp(pos - radius - 1, 0, last)) * stride;
    uint right = source + uint(clamp(pos + radius + 1, 0, last)) * stride;
    U fixed = sum * uint(v[0]) + (U(src[left]) + U(src[right])) * uint(v[1]) + BOX_ROUND;
    dst[i] = P(fixed >> BOX_BITS);
}

#endif