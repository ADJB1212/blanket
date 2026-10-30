// One threadgroup reduces each channel; partials hold extrema, power sums, and NaN.
constant uint MAX_SIMDGROUPS = 32, STATISTICS_PARTIALS = 7;

template <typename T>
void channel_statistics(device const T *src, device float *output, constant uint *p, threadgroup atomic_uint *counts, threadgroup float *partials,
                        uint band, uint member, uint threads, uint lane, uint group, uint groups) {
    uint c = p[2], pixels = p[0] * p[1];
    if (p[6] == 1) {
        for (uint bin = member; bin < HISTOGRAM_BINS; bin += threads)
            atomic_store_explicit(&counts[bin], 0, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint pixel = member; pixel < pixels; pixel += threads)
            atomic_fetch_add_explicit(&counts[uint(src[pixel * c + band])], 1, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint bin = member; bin < HISTOGRAM_BINS; bin += threads)
            output[band * HISTOGRAM_BINS + bin] = as_type<float>(atomic_load_explicit(&counts[bin], memory_order_relaxed));
        return;
    }
    float low = INFINITY, high = -INFINITY;
    float4 sums = float4(0);
    bool nan = false;
    for (uint pixel = member; pixel < pixels; pixel += threads) {
        float value = float(src[pixel * c + band]);
        nan = nan || isnan(value);
        low = min(low, value);
        high = max(high, value);
        float square = value * value, cube = square * value;
        sums += float4(value, square, cube, cube * value);
    }
    low = simd_min(low);
    high = simd_max(high);
    sums = simd_sum(sums);
    nan = simd_any(nan);
    if (lane == 0) {
        threadgroup float *partial = partials + group * STATISTICS_PARTIALS;
        partial[0] = low;
        partial[1] = high;
        for (uint order = 0; order < MOMENT_ORDERS; order++)
            partial[2 + order] = sums[order];
        partial[6] = nan ? 1.0f : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (member != 0)
        return;
    for (uint index = 1; index < groups; index++) {
        threadgroup float *partial = partials + index * STATISTICS_PARTIALS;
        low = min(low, partial[0]);
        high = max(high, partial[1]);
        sums += float4(partial[2], partial[3], partial[4], partial[5]);
        nan = nan || partial[6] != 0;
    }
    device float *result = output + band * STATISTICS_FIELDS;
    result[0] = nan ? NAN : low;
    result[1] = nan ? NAN : high;
    for (uint order = 0; order < MOMENT_ORDERS; order++)
        result[2 + order] = sums[order];
    result[STATISTICS_FIELDS - 1] = as_type<float>(pixels);
}

#define STATISTICS_KERNEL(NAME, SRC)                                                                                                                 \
    kernel void NAME(device const SRC *s [[buffer(0)]],                                                                                              \
                     device float *d [[buffer(1)]],                                                                                                  \
                     constant uint *p [[buffer(3)]],                                                                                                 \
                     uint band [[threadgroup_position_in_grid]],                                                                                     \
                     uint member [[thread_index_in_threadgroup]],                                                                                    \
                     uint threads [[threads_per_threadgroup]],                                                                                       \
                     uint lane [[thread_index_in_simdgroup]],                                                                                        \
                     uint group [[simdgroup_index_in_threadgroup]],                                                                                  \
                     uint groups [[simdgroups_per_threadgroup]]) {                                                                                   \
        threadgroup atomic_uint counts[HISTOGRAM_BINS];                                                                                              \
        threadgroup float partials[MAX_SIMDGROUPS * STATISTICS_PARTIALS];                                                                            \
        channel_statistics(s, d, p, counts, partials, band, member, threads, lane, group, groups);                                                   \
    }
STATISTICS_KERNEL(statistics_u8, uchar)
STATISTICS_KERNEL(statistics_f32, float)
