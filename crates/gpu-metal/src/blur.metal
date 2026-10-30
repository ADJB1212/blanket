template <typename S> S blur_sum(S value) {
    return simd_sum(value);
}
template <> uint4 blur_sum(uint4 value) {
    return uint4(simd_sum(value.x), simd_sum(value.y), simd_sum(value.z), simd_sum(value.w));
}
template <typename S> S blur_prefix(S value) {
    return simd_prefix_exclusive_sum(value);
}
template <> uint4 blur_prefix(uint4 value) {
    return uint4(simd_prefix_exclusive_sum(value.x),
                 simd_prefix_exclusive_sum(value.y),
                 simd_prefix_exclusive_sum(value.z),
                 simd_prefix_exclusive_sum(value.w));
}

// One shared window and an exclusive scan produce a SIMD group's outputs.
template <typename T, typename S>
void scan_box_blur(device const T *src, device T *dst, constant uint *p, device const float *v, uint i, uint lanes, uint channels) {
    uint width = p[0], height = p[1], radius = p[6];
    bool horizontal = p[7] == 0;
    uint length = horizontal ? width : height, lines = horizontal ? height : width;
    uint blocks = (length + lanes - 1) / lanes, lane = i % lanes, group = i / lanes;
    uint band = group % channels, index = group / channels;
    uint line = horizontal ? index / blocks : index % lines;
    uint first = (horizontal ? index % blocks : index / lines) * lanes;
    uint last = length - 1, position = first + lane;
    uint base = horizontal ? line * width * channels + band : line * channels + band;
    uint stride = horizontal ? channels : width * channels;
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

kernel void scan_blur_u8(device const uchar *src [[buffer(0)]], device uchar *dst [[buffer(1)]], constant uint *p [[buffer(3)]],
                         device const float *v [[buffer(4)]], uint i [[thread_position_in_grid]], uint lanes [[threads_per_simdgroup]]) {
    if (p[2] == 4)
        scan_box_blur<uchar4, uint4>(reinterpret_cast<device const uchar4 *>(src), reinterpret_cast<device uchar4 *>(dst), p, v, i, lanes, 1);
    else
        scan_box_blur<uchar, uint>(src, dst, p, v, i, lanes, p[2]);
}
