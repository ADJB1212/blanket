// Scalar histogram, moment, and luminance reduction fallbacks.

// Reduce one luminance block while preserving exact byte-sum carries.
template <typename T> void reduce_luminance_scalar(device const T *src, device float *dst, constant uint *p, uint i) {
    if ((i & 3u) != 0 || i >= (p[3] << 2))
        return;
    uint block = i >> 2, first = block << 8; // REDUCTION_BLOCK_SIZE == 256
    uint last = min(first + REDUCTION_BLOCK_SIZE, p[0] * p[1]), stage = p[6], c = p[2];

    float2 out_lo = float2(0.0f);
    if (stage == 0) {
        // Max sum for 256 bytes is 65,280 -> fits in 32-bit uint without 64-bit ulong emulation
        uint sum = 0;
        for (uint pixel = first; pixel < last; pixel++)
            sum += uint(luminance(src, pixel * c, c));
        out_lo.x = as_type<float>(sum);
    } else if (stage == 1) {
        float sum = 0.0f;
        for (uint pixel = first; pixel < last; pixel++)
            sum += luminance(src, pixel * c, c);
        out_lo.x = sum;
    } else if (stage == 2) {
        device const uint2 *words = reinterpret_cast<device const uint2 *>(src);
        ulong integer_sum = 0;
        for (uint pixel = first; pixel < last; pixel++) {
            uint2 w = words[pixel << 1];
            integer_sum += ulong(w.x) | (ulong(w.y) << 32);
        }
        out_lo = float2(as_type<float>(uint(integer_sum)), as_type<float>(uint(integer_sum >> 32)));
    } else {
        float sum = 0.0f;
        for (uint pixel = first; pixel < last; pixel++)
            sum += float(src[pixel << 2]);
        out_lo.x = sum;
    }
    reinterpret_cast<device float4 *>(dst)[block] = float4(out_lo.x, out_lo.y, 0.0f, 0.0f);
}

// Compute one channel's histogram or moments for the scalar fallback.
template <typename T> void channel_statistics_scalar(device const T *src, device T *dst, constant uint *p, uint i, uint c) {
    if (i >= c)
        return;
    uint pixels = p[0] * p[1], k = p[6];
    device float *output = reinterpret_cast<device float *>(dst);
    device const T *chan_src = src + i;

    if (k == 1) {
        uint counts[HISTOGRAM_BINS] = {0};
        for (uint pixel = 0; pixel < pixels; pixel++)
            counts[uint(chan_src[pixel * c])]++;
        device float *hist_out = output + (i << 8); // HISTOGRAM_BINS == 256
        for (uint bin = 0; bin < HISTOGRAM_BINS; bin++)
            hist_out[bin] = as_type<float>(counts[bin]);
    } else {
        float low = INFINITY, high = -INFINITY;
        float4 sums = float4(0.0f);
        for (uint pixel = 0; pixel < pixels; pixel++) {
            float value = float(chan_src[pixel * c]);
            low = fmin(low, value);
            high = fmax(high, value);
            float sq = value * value;
            sums += float4(value, sq, sq * value, sq * sq);
        }
        // NaN propagates naturally through sums.x; only check once after the loop
        bool nan = !is_same_v<T, uchar> && isnan(sums.x);
        device float *stat_out = output + i * STATISTICS_FIELDS;
        stat_out[0] = nan ? NAN : low;
        stat_out[1] = nan ? NAN : high;
        stat_out[2] = sums.x;
        stat_out[3] = sums.y;
        stat_out[4] = sums.z;
        stat_out[5] = sums.w;
        stat_out[6] = as_type<float>(pixels);
    }
}

#define REDUCTION_KERNEL(NAME, SRC)                                                                                                                  \
    /* Dispatch the scalar luminance reduction fallback for this sample type. */                                                                     \
    kernel void NAME(device const SRC *s [[buffer(0)]],                                                                                              \
                     device float *d [[buffer(1)]],                                                                                                  \
                     device const SRC *o [[buffer(2)]],                                                                                              \
                     constant uint *p [[buffer(3)]],                                                                                                 \
                     device const float *v [[buffer(4)]],                                                                                            \
                     uint i [[thread_position_in_grid]]) {                                                                                           \
        reduce_luminance_scalar(s, d, p, i);                                                                                                         \
    }
REDUCTION_KERNEL(reduce_u8, uchar)
REDUCTION_KERNEL(reduce_f32, float)