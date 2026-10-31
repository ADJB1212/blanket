// Scalar histogram, moment, and luminance reduction fallbacks.

// Reduce one luminance block while preserving exact byte-sum carries.
template <typename T> void reduce_luminance_scalar(device const T *src, device float *dst, constant uint *p, uint i) {
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

// Compute one channel's histogram or moments for the scalar fallback.
template <typename T> void channel_statistics_scalar(device const T *src, device T *dst, constant uint *p, uint i, uint c) {
    uint sw = p[0], sh = p[1], k = p[6];
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
