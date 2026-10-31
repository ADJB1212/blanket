// Scalar filters and packed byte-filter helpers.

// Blur consecutive byte samples with an exact sliding-window sum.
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

// Apply a fractional box window to one sample with replicated borders.
template <typename T> void box_blur_sample(device const T *src, device T *dst, constant uint *p, device const float *v, uint i, uint c) {
    uint sw = p[0], sh = p[1], dw = p[3], k = p[6];
    uint band = i % c, x = (i / c) % dw, y = i / (c * dw);
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

// Select a neighborhood rank while preserving NaN propagation.
template <typename T> void rank_sample(device const T *src, device T *dst, constant uint *p, uint i, uint c) {
    uint sw = p[0], sh = p[1], dw = p[3], k = p[6];
    uint band = i % c, x = (i / c) % dw, y = i / (c * dw);
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

// Accumulate one resize sample in the original tap order and precision.
template <typename T> float resize_sample(device const T *src, constant uint *p, device const float *v, uint i, uint c) {
    uint sw = p[0], dw = p[3], k = p[6];
    uint band = i % c, x = (i / c) % dw, y = i / (c * dw);
    float out = 0;
    uint pos = p[7] == 0 ? x : y;
    uint base = pos * k, start = uint(v[base]), count = uint(v[base + 1]);
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
    return out;
}

// Convolve one interior sample and copy samples along the image border.
template <typename T> float convolve_sample(device const T *src, constant uint *p, device const float *v, uint i, uint c) {
    uint sw = p[0], sh = p[1], dw = p[3], k = p[6];
    uint band = i % c, x = (i / c) % dw, y = i / (c * dw);
    float out = read_sample(src, i);
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
    return out;
}

#ifdef BLANKET_METAL_SPECIALIZE
// Load four neighboring bytes, clamping each sample at image edges.
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

// Sort two packed sample vectors independently by lane.
void rank_swap(thread uchar4 &a, thread uchar4 &b) {
    uchar4 low = min(a, b);
    b = max(a, b);
    a = low;
}

// Select the median of nine packed samples with a sorting network.
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

// Resize or convolve one packed pixel with CPU-compatible rounding.
template <typename P, typename I, typename F, bool Floating = false>
void filter_pixels(device const P *src, device P *dst, constant uint *p, device const float *v, uint i) {
    if (i >= p[3] * p[4])
        return;
    uint x = i % p[3], y = i / p[3];
    if (SPECIALIZED_OPERATION == OP_RESIZE) {
        uint base = (p[7] == 0 ? x : y) * p[6];
        uint start = uint(v[base]), taps = uint(v[base + 1]);
        uint source = p[7] == 0 ? y * p[0] + start : start * p[0] + x;
        uint stride = p[7] == 0 ? 1 : p[0];
        if (Floating) {
            F sum = F(0);
            for (uint t = 0; t < taps; t++)
                sum += F(src[source + t * stride]) * v[base + 2 + t];
            dst[i] = P(sum);
        } else {
            I fixed = I(RESIZE_ROUND);
            for (uint t = 0; t < taps; t++)
                fixed += I(src[source + t * stride]) * int(round(v[base + 2 + t] * float(RESIZE_SCALE)));
            dst[i] = P(clamp(F(fixed >> RESIZE_BITS), 0.0f, 255.0f));
        }
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
            result = Floating ? P(sum) : P(clamp(floor(sum + 0.5f), 0.0f, 255.0f));
        }
        dst[i] = result;
    }
}

// Apply a fractional box window to one packed byte pixel.
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

#endif
