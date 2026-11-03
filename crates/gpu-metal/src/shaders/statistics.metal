// Threadgroups partition channels; partials hold extrema, power sums, and NaN.
constant uint MAX_SIMDGROUPS = 32, STATISTICS_PARTIALS = 7;

// Reduce a channel partition into histogram counts or statistical moments.
template <typename T, bool Histogram>
void channel_statistics(device const T *src, device float *output, constant uint *p, threadgroup atomic_uint *counts, threadgroup float *partials,
                        uint band, uint block, uint member, uint threads, uint lane, uint group, uint groups, uint lanes) {
    uint c = p[2], pixels = p[0] * p[1];
    uint blocks = p[7];
    uint stride = threads * blocks;
    device const T *chan_src = src + band;
    output += (band * blocks + block) * (Histogram ? HISTOGRAM_BINS : STATISTICS_FIELDS);

    if (Histogram) {
        for (uint bin = member; bin < HISTOGRAM_BINS; bin += threads)
            atomic_store_explicit(&counts[bin], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint first = block * threads + group * lanes;
        for (; first < pixels; first += stride) {
            uint pixel = first + lane;
            uint value = pixel < pixels ? uint(chan_src[pixel * c]) : 0u;
            uint leader = simd_broadcast_first(value);
            // Keep mixed groups on the simple atomic path for the remaining input.
            if (!simd_all(pixel >= pixels || value == leader))
                break;
            if (lane == 0)
                atomic_fetch_add_explicit(&counts[leader], min(lanes, pixels - first), memory_order_relaxed);
        }
        for (uint pixel = first + lane; pixel < pixels; pixel += stride)
            atomic_fetch_add_explicit(&counts[uint(chan_src[pixel * c])], 1u, memory_order_relaxed);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint bin = member; bin < HISTOGRAM_BINS; bin += threads)
            output[bin] = as_type<float>(atomic_load_explicit(&counts[bin], memory_order_relaxed));
        return;
    }

    float low = INFINITY, high = -INFINITY;
    float4 sums = float4(0.0f);
    for (uint pixel = block * threads + member; pixel < pixels; pixel += stride) {
        float value = float(chan_src[pixel * c]);
        low = fmin(low, value);
        high = fmax(high, value);
        float sq = value * value;
        sums += float4(value, sq, sq * value, sq * sq);
    }
    bool nan = !is_same_v<T, uchar> && isnan(sums.x);
    low = simd_min(low);
    high = simd_max(high);
    sums = simd_sum(sums);
    nan = !is_same_v<T, uchar> && simd_any(nan);

    if (lane == 0) {
        threadgroup float *partial = partials + group * STATISTICS_PARTIALS;
        partial[0] = low;
        partial[1] = high;
        partial[2] = sums.x;
        partial[3] = sums.y;
        partial[4] = sums.z;
        partial[5] = sums.w;
        partial[6] = nan ? 1.0f : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (member != 0)
        return;

    for (uint index = 1; index < groups; index++) {
        threadgroup float *partial = partials + index * STATISTICS_PARTIALS;
        low = fmin(low, partial[0]);
        high = fmax(high, partial[1]);
        sums += float4(partial[2], partial[3], partial[4], partial[5]);
        nan = nan || (partial[6] != 0.0f);
    }
    output[0] = nan ? NAN : low;
    output[1] = nan ? NAN : high;
    output[2] = sums.x;
    output[3] = sums.y;
    output[4] = sums.z;
    output[5] = sums.w;
    output[STATISTICS_FIELDS - 1] = as_type<float>(pixels);
}

#define STATISTICS_KERNEL(NAME, SRC, HISTOGRAM)                                                                                                      \
    /* Dispatch a channel partition with only the required shared scratch storage. */                                                                \
    kernel void NAME(device const SRC *s [[buffer(0)]],                                                                                              \
                     device float *d [[buffer(1)]],                                                                                                  \
                     constant uint *p [[buffer(3)]],                                                                                                 \
                     uint position [[threadgroup_position_in_grid]],                                                                                 \
                     uint member [[thread_index_in_threadgroup]],                                                                                    \
                     uint threads [[threads_per_threadgroup]],                                                                                       \
                     uint lane [[thread_index_in_simdgroup]],                                                                                        \
                     uint group [[simdgroup_index_in_threadgroup]],                                                                                  \
                     uint groups [[simdgroups_per_threadgroup]],                                                                                     \
                     uint lanes [[threads_per_simdgroup]]) {                                                                                         \
        threadgroup atomic_uint counts[HISTOGRAM ? HISTOGRAM_BINS : 1];                                                                              \
        threadgroup float partials[HISTOGRAM ? 1 : MAX_SIMDGROUPS * STATISTICS_PARTIALS];                                                            \
        uint block, band;                                                                                                                            \
        decompose_channel(position, p[2], block, band);                                                                                              \
        channel_statistics<SRC, HISTOGRAM>(s, d, p, counts, partials, band, block, member, threads, lane, group, groups, lanes);                     \
    }
STATISTICS_KERNEL(statistics_u8, uchar, false)
STATISTICS_KERNEL(statistics_f32, float, false)
STATISTICS_KERNEL(histogram_u8, uchar, true)

// Merge channel partitions while preserving exact counts and NaN extrema.
kernel void finish_statistics(device const float *src [[buffer(0)]], device float *dst [[buffer(1)]], constant uint *p [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
    uint blocks = p[7];
    if (p[6] == 1) {
        if (i >= (p[2] << 8)) // HISTOGRAM_BINS == 256
            return;
        uint band = i >> 8, bin = i & 255u, count = 0;
        device const float *bin_src = src + (band * blocks << 8) + bin;
        for (uint block = 0; block < blocks; block++)
            count += as_type<uint>(bin_src[block << 8]);
        dst[i] = as_type<float>(count);
        return;
    }
    if (i >= p[2])
        return;
    float low = INFINITY, high = -INFINITY;
    float4 sums = float4(0.0f);
    bool nan = false;
    device const float *base_partial = src + i * blocks * STATISTICS_FIELDS;
    for (uint block = 0; block < blocks; block++) {
        device const float *partial = base_partial + block * STATISTICS_FIELDS;
        float p0 = partial[0];
        nan = nan || isnan(p0);
        low = fmin(low, p0);
        high = fmax(high, partial[1]);
        sums += float4(partial[2], partial[3], partial[4], partial[5]);
    }
    device float *result = dst + i * STATISTICS_FIELDS;
    result[0] = nan ? NAN : low;
    result[1] = nan ? NAN : high;
    result[2] = sums.x;
    result[3] = sums.y;
    result[4] = sums.z;
    result[5] = sums.w;
    result[STATISTICS_FIELDS - 1] = as_type<float>(p[0] * p[1]);
}

// Reduce one luminance block cooperatively, carrying byte sums across words.
template <typename T> void parallel_luminance(device const T *src, device float *dst, constant uint *p, uint i, uint lanes, uint lane) {
    uint block = i >> ctz(lanes), first = block << 8; // REDUCTION_BLOCK_SIZE == 256
    uint last = min(first + REDUCTION_BLOCK_SIZE, p[0] * p[1]), stage = p[6], c = p[2];

    if (stage == 0) {
        uint sum = 0;
        for (uint pixel = first + lane; pixel < last; pixel += lanes)
            sum += uint(luminance(src, pixel * c, c));
        // A block contains at most 256 byte luminances, so no carry is possible.
        sum = simd_sum(sum);
        if (lane == 0)
            reinterpret_cast<device float4 *>(dst)[block] = float4(as_type<float>(sum), 0.0f, 0.0f, 0.0f);
        return;
    }
    if (stage == 2) {
        device const uint2 *words = reinterpret_cast<device const uint2 *>(src);
        ulong integer_sum = 0;
        for (uint pixel = first + lane; pixel < last; pixel += lanes) {
            uint2 w = words[pixel << 1];
            integer_sum += ulong(w.x) | (ulong(w.y) << 32);
        }
        uint low = uint(integer_sum), high = uint(integer_sum >> 32);
        for (uint delta = lanes >> 1; delta > 0; delta >>= 1) {
            uint next_low = simd_shuffle_down(low, delta), next_high = simd_shuffle_down(high, delta);
            uint combined = low + next_low;
            high += next_high + uint(combined < low);
            low = combined;
        }
        if (lane == 0)
            reinterpret_cast<device float4 *>(dst)[block] = float4(as_type<float>(low), as_type<float>(high), 0.0f, 0.0f);
        return;
    }

    float sum = 0.0f;
    if (stage == 1) {
        for (uint pixel = first + lane; pixel < last; pixel += lanes)
            sum += luminance(src, pixel * c, c);
    } else {
        for (uint pixel = first + lane; pixel < last; pixel += lanes)
            sum += float(src[pixel << 2]);
    }
    sum = simd_sum(sum);
    if (lane == 0)
        reinterpret_cast<device float4 *>(dst)[block] = float4(sum, 0.0f, 0.0f, 0.0f);
}

#define LUMINANCE_KERNEL(NAME, TYPE)                                                                                                                 \
    /* Dispatch a SIMD-group luminance reduction for this sample type. */                                                                            \
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