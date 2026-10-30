// Threadgroups partition channels; partials hold extrema, power sums, and NaN.
constant uint MAX_SIMDGROUPS = 32, STATISTICS_PARTIALS = 7;

template <typename T>
void channel_statistics(device const T *src, device float *output, constant uint *p, threadgroup atomic_uint *counts, threadgroup float *partials,
                        uint band, uint block, uint member, uint threads, uint lane, uint group, uint groups) {
    uint c = p[2], pixels = p[0] * p[1];
    uint blocks = p[7];
    output += (band * blocks + block) * (p[6] == 1 ? HISTOGRAM_BINS : STATISTICS_FIELDS);
    if (p[6] == 1) {
        for (uint bin = member; bin < HISTOGRAM_BINS; bin += threads)
            atomic_store_explicit(&counts[bin], 0, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint pixel = block * threads + member; pixel < pixels; pixel += threads * blocks)
            atomic_fetch_add_explicit(&counts[uint(src[pixel * c + band])], 1, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint bin = member; bin < HISTOGRAM_BINS; bin += threads)
            output[bin] = as_type<float>(atomic_load_explicit(&counts[bin], memory_order_relaxed));
        return;
    }
    float low = INFINITY, high = -INFINITY;
    float4 sums = float4(0);
    bool nan = false;
    for (uint pixel = block * threads + member; pixel < pixels; pixel += threads * blocks) {
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
    device float *result = output;
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
                     uint position [[threadgroup_position_in_grid]],                                                                                 \
                     uint member [[thread_index_in_threadgroup]],                                                                                    \
                     uint threads [[threads_per_threadgroup]],                                                                                       \
                     uint lane [[thread_index_in_simdgroup]],                                                                                        \
                     uint group [[simdgroup_index_in_threadgroup]],                                                                                  \
                     uint groups [[simdgroups_per_threadgroup]]) {                                                                                   \
        threadgroup atomic_uint counts[HISTOGRAM_BINS];                                                                                              \
        threadgroup float partials[MAX_SIMDGROUPS * STATISTICS_PARTIALS];                                                                            \
        channel_statistics(s, d, p, counts, partials, position % p[2], position / p[2], member, threads, lane, group, groups);                       \
    }
STATISTICS_KERNEL(statistics_u8, uchar)
STATISTICS_KERNEL(statistics_f32, float)

kernel void finish_statistics(device const float *src [[buffer(0)]], device float *dst [[buffer(1)]], constant uint *p [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
    uint blocks = p[7];
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
    float low = INFINITY, high = -INFINITY;
    float4 sums = float4(0);
    bool nan = false;
    for (uint block = 0; block < blocks; block++) {
        device const float *partial = src + (i * blocks + block) * STATISTICS_FIELDS;
        nan = nan || isnan(partial[0]);
        low = min(low, partial[0]);
        high = max(high, partial[1]);
        sums += float4(partial[2], partial[3], partial[4], partial[5]);
    }
    device float *result = dst + i * STATISTICS_FIELDS;
    result[0] = nan ? NAN : low;
    result[1] = nan ? NAN : high;
    for (uint order = 0; order < MOMENT_ORDERS; order++)
        result[2 + order] = sums[order];
    result[STATISTICS_FIELDS - 1] = as_type<float>(p[0] * p[1]);
}

template <typename T> void parallel_luminance(device const T *src, device float *dst, constant uint *p, uint i, uint lanes, uint lane) {
    uint block = i / lanes, first = block * REDUCTION_BLOCK_SIZE;
    uint last = min(first + REDUCTION_BLOCK_SIZE, p[0] * p[1]), stage = p[6];
    ulong integer_sum = 0;
    float sum = 0;
    for (uint pixel = first + lane; pixel < last; pixel += lanes) {
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
    uint low = uint(integer_sum), high = uint(integer_sum >> 32);
    if (stage == 0 || stage == 2) {
        for (uint delta = lanes / 2; delta > 0; delta /= 2) {
            uint next_low = simd_shuffle_down(low, delta), next_high = simd_shuffle_down(high, delta);
            uint combined = low + next_low;
            high += next_high + uint(combined < low);
            low = combined;
        }
    } else
        sum = simd_sum(sum);
    if (lane == 0) {
        uint output = block * 4;
        dst[output] = stage == 0 || stage == 2 ? as_type<float>(low) : sum;
        dst[output + 1] = stage == 0 || stage == 2 ? as_type<float>(high) : 0;
        dst[output + 2] = 0;
        dst[output + 3] = 0;
    }
}

#define LUMINANCE_KERNEL(NAME, TYPE)                                                                                                                 \
    kernel void NAME(device const TYPE *src [[buffer(0)]],                                                                                           \
                     device float *dst [[buffer(1)]],                                                                                                \
                     constant uint *p [[buffer(3)]],                                                                                                 \
                     uint i [[thread_position_in_grid]],                                                                                             \
                     uint lanes [[threads_per_simdgroup]],                                                                                           \
                     uint lane [[thread_index_in_simdgroup]]) {                                                                                      \
        parallel_luminance(src, dst, p, i, lanes, lane);                                                                                             \
    }
LUMINANCE_KERNEL(luminance_u8, uchar)
LUMINANCE_KERNEL(luminance_f32, float)
