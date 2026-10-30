template <typename S> S blur_sum(S value) {
    return simd_sum(value);
}
template <> uint2 blur_sum(uint2 value) {
    return uint2(simd_sum(value.x), simd_sum(value.y));
}
template <> uint3 blur_sum(uint3 value) {
    return uint3(simd_sum(value.x), simd_sum(value.y), simd_sum(value.z));
}
template <> uint4 blur_sum(uint4 value) {
    return uint4(simd_sum(value.x), simd_sum(value.y), simd_sum(value.z), simd_sum(value.w));
}
template <typename S> S blur_prefix(S value) {
    return simd_prefix_exclusive_sum(value);
}
template <> uint2 blur_prefix(uint2 value) {
    return uint2(simd_prefix_exclusive_sum(value.x), simd_prefix_exclusive_sum(value.y));
}
template <> uint3 blur_prefix(uint3 value) {
    return uint3(simd_prefix_exclusive_sum(value.x), simd_prefix_exclusive_sum(value.y), simd_prefix_exclusive_sum(value.z));
}
template <> uint4 blur_prefix(uint4 value) {
    return uint4(simd_prefix_exclusive_sum(value.x),
                 simd_prefix_exclusive_sum(value.y),
                 simd_prefix_exclusive_sum(value.z),
                 simd_prefix_exclusive_sum(value.w));
}

// One shared window and an exclusive scan produce a SIMD group's outputs.
template <typename T, typename S>
void scan_box_blur(device const T *src, device T *dst, constant uint *p, device const float *v, uint i, uint lanes) {
    uint width = p[0], height = p[1], radius = p[6];
    bool horizontal = p[7] == 0;
    uint length = horizontal ? width : height, lines = horizontal ? height : width;
    uint blocks = (length + lanes - 1) / lanes, lane = i % lanes, group = i / lanes;
    uint line = horizontal ? group / blocks : group % lines;
    uint first = (horizontal ? group % blocks : group / lines) * lanes;
    uint last = length - 1, position = first + lane;
    uint base = horizontal ? line * width : line;
    uint stride = horizontal ? 1 : width;
    uint low = first > radius ? first - radius : 0, high = min(last, first + radius);
    S sum = S(0);
    for (uint t = low + lane; t <= high; t += lanes)
        sum += S(src[base + t * stride]);
    if (lane == 0) {
        sum += S(src[base]) * (radius > first ? radius - first : 0);
        sum += S(src[base + last * stride]) * (first + radius > last ? first + radius - last : 0);
    }
    sum = blur_sum(sum);
    uint outgoing = base + min(position > radius ? position - radius : 0, last) * stride;
    uint right = base + min(position + radius + 1, last) * stride;
    sum += blur_prefix(S(src[right]) - S(src[outgoing]));
    if (position < length) {
        uint left = base + (position > radius ? position - radius - 1 : 0) * stride;
        S fixed = sum * uint(v[0]) + (S(src[left]) + S(src[right])) * uint(v[1]) + BOX_ROUND;
        dst[base + position * stride] = T(fixed >> BOX_BITS);
    }
}

#define SCAN_BLUR_KERNEL(NAME, PIXEL, SUM)                                                                                                           \
    kernel void NAME(device const PIXEL *src [[buffer(0)]],                                                                                          \
                     device PIXEL *dst [[buffer(1)]],                                                                                                \
                     constant uint *p [[buffer(3)]],                                                                                                 \
                     device const float *v [[buffer(4)]],                                                                                            \
                     uint i [[thread_position_in_grid]],                                                                                             \
                     uint lanes [[threads_per_simdgroup]]) {                                                                                         \
        scan_box_blur<PIXEL, SUM>(src, dst, p, v, i, lanes);                                                                                         \
    }
SCAN_BLUR_KERNEL(scan_blur_l_u8, uchar, uint)
SCAN_BLUR_KERNEL(scan_blur_la_u8, uchar2, uint2)
SCAN_BLUR_KERNEL(scan_blur_rgb_u8, packed_uchar3, uint3)
SCAN_BLUR_KERNEL(scan_blur_rgba_u8, uchar4, uint4)
