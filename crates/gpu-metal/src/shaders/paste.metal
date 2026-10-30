// Paste a clipped pixel through byte or float mask coverage.
template <typename T, typename F, bool Floating = false>
void paste_image(device const T *background, device T *dst, device const T *source, device const uchar *mask, constant uint *p, uint i,
                 uint channels) {
    uint width = p[0], height = p[1];
    if (i >= width * height * channels)
        return;
    uint pixel = i / channels, band = i % channels;
    long x = long(pixel % width) - as_type<int>(p[5]);
    long y = long(pixel / width) - as_type<int>(p[6]);
    if (x < 0 || y < 0 || x >= p[3] || y >= p[4]) {
        dst[i] = background[i];
        return;
    }
    uint source_pixel = uint(y) * p[3] + uint(x);
    float coverage = 1;
    if (p[7] == 1)
        coverage = float(mask[source_pixel]) / 255.0f;
    else if (p[7] == 2) {
        coverage = reinterpret_cast<device const float *>(mask)[source_pixel];
        coverage = isnan(coverage) ? 0.0f : clamp(coverage, 0.0f, 1.0f);
    }
    F a = F(background[i]), b = F(source[source_pixel * channels + band]);
    F value = (coverage == 1 ? F(0) : a * (1 - coverage)) + (coverage == 0 ? F(0) : b * coverage);
    if (Floating)
        dst[i] = T(value);
    else
        dst[i] = T(clamp(floor(value + 0.5f), 0.0f, 255.0f));
}

// Dispatch byte paste with the packed pixel type for its channel count.
kernel void paste_u8(device const uchar *a [[buffer(0)]], device uchar *dst [[buffer(1)]], device const uchar *b [[buffer(2)]],
                     constant uint *p [[buffer(3)]], device const uchar *mask [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if (p[2] == 4)
        paste_image<uchar4, float4>(reinterpret_cast<device const uchar4 *>(a),
                                    reinterpret_cast<device uchar4 *>(dst),
                                    reinterpret_cast<device const uchar4 *>(b),
                                    mask,
                                    p,
                                    i,
                                    1);
    else if (p[2] == 3)
        paste_image<packed_uchar3, float3>(reinterpret_cast<device const packed_uchar3 *>(a),
                                           reinterpret_cast<device packed_uchar3 *>(dst),
                                           reinterpret_cast<device const packed_uchar3 *>(b),
                                           mask,
                                           p,
                                           i,
                                           1);
    else if (p[2] == 2)
        paste_image<uchar2, float2>(reinterpret_cast<device const uchar2 *>(a),
                                    reinterpret_cast<device uchar2 *>(dst),
                                    reinterpret_cast<device const uchar2 *>(b),
                                    mask,
                                    p,
                                    i,
                                    1);
    else
        paste_image<uchar, float>(a, dst, b, mask, p, i, 1);
}

// Dispatch float paste with the packed pixel type for its channel count.
kernel void paste_f32(device const float *a [[buffer(0)]], device float *dst [[buffer(1)]], device const float *b [[buffer(2)]],
                      constant uint *p [[buffer(3)]], device const uchar *mask [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if (p[2] == 4)
        paste_image<float4, float4, true>(reinterpret_cast<device const float4 *>(a),
                                          reinterpret_cast<device float4 *>(dst),
                                          reinterpret_cast<device const float4 *>(b),
                                          mask,
                                          p,
                                          i,
                                          1);
    else if (p[2] == 3)
        paste_image<packed_float3, float3, true>(reinterpret_cast<device const packed_float3 *>(a),
                                                 reinterpret_cast<device packed_float3 *>(dst),
                                                 reinterpret_cast<device const packed_float3 *>(b),
                                                 mask,
                                                 p,
                                                 i,
                                                 1);
    else if (p[2] == 2)
        paste_image<float2, float2, true>(reinterpret_cast<device const float2 *>(a),
                                          reinterpret_cast<device float2 *>(dst),
                                          reinterpret_cast<device const float2 *>(b),
                                          mask,
                                          p,
                                          i,
                                          1);
    else
        paste_image<float, float, true>(a, dst, b, mask, p, i, 1);
}
