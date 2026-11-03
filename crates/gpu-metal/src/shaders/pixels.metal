// Copy one packed pixel through crop, transpose, or wrapped-offset geometry.
template <typename P> void geometry_pixels(device const P *src, device P *dst, constant uint *p, uint i) {
    uint sw = p[0], sh = p[1], width = p[3], height = p[4];
    if (i >= width * height)
        return;
    uint x, y;
    decompose_2d(i, width, x, y);
    uint sx = x, sy = y;
    if (SPECIALIZED_OPERATION == OP_CROP) {
        int cx = int(x) + as_type<int>(p[6]), cy = int(y) + as_type<int>(p[7]);
        dst[i] = (cx >= 0 && cy >= 0 && uint(cx) < sw && uint(cy) < sh) ? src[uint(cy) * sw + uint(cx)] : P(0);
        return;
    }
    if (SPECIALIZED_OPERATION == OP_TRANSPOSE) {
        switch (p[6]) {
        case 0:
            sx = sw - 1 - x;
            break;
        case 1:
            sy = sh - 1 - y;
            break;
        case 2:
            sx = sw - 1 - y;
            sy = x;
            break;
        case 3:
            sx = sw - 1 - x;
            sy = sh - 1 - y;
            break;
        case 4:
            sx = y;
            sy = sh - 1 - x;
            break;
        case 5:
            sx = y;
            sy = x;
            break;
        case 6:
            sx = sw - 1 - y;
            sy = sh - 1 - x;
            break;
        }
    } else {
        uint p6 = p[6], p7 = p[7];
        sx = x >= p6 ? x - p6 : x + sw - p6;
        sy = y >= p7 ? y - p7 : y + sh - p7;
    }
    dst[i] = src[sy * sw + sx];
}

// Composite or change alpha association for one packed pixel.
template <typename P, typename F, bool Floating, uint Channels>
void alpha_pixels(device const P *src, device P *dst, device const P *other, constant uint *p, uint i) {
    if (i >= p[3] * p[4])
        return;
    F a = F(src[i]), result;
    constexpr float maxval = Floating ? 1.0f : 255.0f;
    constexpr float inv_maxval = 1.0f / maxval;
    float raw = float(a[Channels - 1]);
    float da = isnan(raw) ? 0.0f : saturate(raw * inv_maxval);
    if (SPECIALIZED_OPERATION == OP_ALPHA_COMPOSITE) {
        F b = F(other[i]);
        float sa = float(b[Channels - 1]) * inv_maxval;
        sa = isnan(sa) ? 0.0f : saturate(sa);
        if (!Floating && sa == 0.0f) {
            dst[i] = src[i];
            return;
        }
        float inv_sa = 1.0f - sa;
        float alpha = fma(da, inv_sa, sa);
        result = (sa == 1.0f) ? b : (alpha == 0.0f ? F(0) : ((sa == 0.0f ? F(0) : b * sa) + (da == 0.0f ? F(0) : a * da) * inv_sa) / alpha);
        result[Channels - 1] = alpha * maxval;
    } else {
        result = (da == 0.0f) ? ((SPECIALIZED_OPERATION == OP_UNPREMULTIPLY && !Floating) ? a : F(0))
                              : (SPECIALIZED_OPERATION == OP_PREMULTIPLY ? a * da : (Floating ? a / da : (a * maxval) / raw));
        result[Channels - 1] = da * maxval;
        if (SPECIALIZED_OPERATION == OP_UNPREMULTIPLY && !Floating)
            result = floor(result);
    }
    if (Floating)
        dst[i] = P(result);
    else
        dst[i] = P(clamp(floor(result + 0.5f), 0.0f, 255.0f));
}

// Resize or convolve one float pixel while retaining its sample range.
kernel void filter_f32(device const float *src [[buffer(0)]], device float *dst [[buffer(1)]], constant uint *p [[buffer(3)]],
                       device const float *v [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (SPECIALIZED_CHANNELS == 1)
        filter_pixels<float, int, float, true>(src, dst, p, v, i);
    else if (SPECIALIZED_CHANNELS == 2)
        filter_pixels<float2, int2, float2, true>(reinterpret_cast<device const float2 *>(src), reinterpret_cast<device float2 *>(dst), p, v, i);
    else if (SPECIALIZED_CHANNELS == 3)
        filter_pixels<packed_float3, int3, float3, true>(
            reinterpret_cast<device const packed_float3 *>(src), reinterpret_cast<device packed_float3 *>(dst), p, v, i);
    else
        filter_pixels<float4, int4, float4, true>(reinterpret_cast<device const float4 *>(src), reinterpret_cast<device float4 *>(dst), p, v, i);
}

// Share coordinates and interpolation weights across a float pixel.
template <typename P, typename F, uint Channels>
void warp_pixels(device const P *src, device P *dst, device const P *other, constant uint *p, device const float *v, uint i) {
    uint dw = p[3], dh = p[4];
    if (i >= dw * dh)
        return;
    uint sw = p[0], sh = p[1];
    float sx, sy;
    if (SPECIALIZED_OPERATION == OP_REMAP) {
        device const float *coordinates = reinterpret_cast<device const float *>(other);
        sx = coordinates[i * 3];
        sy = coordinates[i * 3 + 1];
    } else {
        uint x, y;
        decompose_2d(i, dw, x, y);
        float px = float(x) + 0.5f, py = float(y) + 0.5f;
        float inv_denom = 1.0f / fma(v[6], px, fma(v[7], py, 1.0f));
        sx = fma(v[0], px, fma(v[1], py, v[2])) * inv_denom;
        sy = fma(v[3], px, fma(v[4], py, v[5])) * inv_denom;
    }
    if (!isfinite(sx) || !isfinite(sy) || sx < 0.0f || sy < 0.0f || sx >= float(sw) || sy >= float(sh)) {
        F fill;
#pragma unroll
        for (uint band = 0; band < Channels; band++)
            fill[band] = v[8 + band];
        dst[i] = P(fill);
        return;
    }
    uint k = p[6];
    if (k == 0) {
        dst[i] = src[uint(sy) * sw + uint(sx)];
        return;
    }
    sx -= 0.5f;
    sy -= 0.5f;
    float fl_x = floor(sx), fl_y = floor(sy);
    int ix = int(fl_x), iy = int(fl_y);
    float fx = sx - fl_x, fy = sy - fl_y;
    int max_x = int(sw) - 1, max_y = int(sh) - 1;
    F result = F(0);

    if (k == 2) {
        uint x_off[4] = {uint(clamp(ix - 1, 0, max_x)), uint(clamp(ix, 0, max_x)), uint(clamp(ix + 1, 0, max_x)), uint(clamp(ix + 2, 0, max_x))};
        F samples[4][4];
#pragma unroll
        for (int ky = 0; ky < 4; ky++) {
            device const P *row_ptr = src + uint(clamp(iy + ky - 1, 0, max_y)) * sw;
#pragma unroll
            for (int kx = 0; kx < 4; kx++)
                samples[ky][kx] = F(row_ptr[x_off[kx]]);
        }
#pragma unroll
        for (uint band = 0; band < Channels; band++) {
            float4 rows;
#pragma unroll
            for (int ky = 0; ky < 4; ky++) {
                float4 taps = float4(samples[ky][0][band], samples[ky][1][band], samples[ky][2][band], samples[ky][3][band]);
                rows[ky] = warp_cubic(taps, fx);
            }
            result[band] = warp_cubic(rows, fy);
        }
    } else {
        uint x0 = uint(clamp(ix, 0, max_x)), x1 = uint(clamp(ix + 1, 0, max_x));
        uint y0 = uint(clamp(iy, 0, max_y)) * sw, y1 = uint(clamp(iy + 1, 0, max_y)) * sw;
        F s00 = F(src[y0 + x0]), s01 = F(src[y0 + x1]);
        F s10 = F(src[y1 + x0]), s11 = F(src[y1 + x1]);
        float wx0 = 1.0f - fx, wy0 = 1.0f - fy;
#pragma unroll
        for (uint band = 0; band < Channels; band++) {
            float r0 = s00[band] * wx0 + s01[band] * fx;
            float r1 = s10[band] * wx0 + s11[band] * fx;
            result[band] = r0 * wy0 + r1 * fy;
        }
    }
    dst[i] = P(result);
}

// Dispatch a packed pixel to geometry, resampling, or alpha processing.
template <typename P, typename F, bool Floating, uint Channels>
void process_pixel(device const P *src, device P *dst, device const P *other, constant uint *p, device const float *v, uint i) {
    if (Floating && (SPECIALIZED_OPERATION == OP_WARP || SPECIALIZED_OPERATION == OP_REMAP))
        warp_pixels<P, F, Channels>(src, dst, other, p, v, i);
    else if (SPECIALIZED_OPERATION == OP_ALPHA_COMPOSITE || SPECIALIZED_OPERATION == OP_PREMULTIPLY || SPECIALIZED_OPERATION == OP_UNPREMULTIPLY)
        alpha_pixels<P, F, Floating, Channels>(src, dst, other, p, i);
    else
        geometry_pixels(src, dst, p, i);
}

#define PIXEL_KERNEL(NAME, SAMPLE, TWO, THREE, FOUR, FLOATING)                                                                                       \
    kernel void NAME(device const SAMPLE *src [[buffer(0)]],                                                                                         \
                     device SAMPLE *dst [[buffer(1)]],                                                                                               \
                     device const SAMPLE *other [[buffer(2)]],                                                                                       \
                     constant uint *p [[buffer(3)]],                                                                                                 \
                     device const float *v [[buffer(4)]],                                                                                            \
                     uint i [[thread_position_in_grid]]) {                                                                                           \
        if (SPECIALIZED_CHANNELS == 1) {                                                                                                             \
            if (SPECIALIZED_OPERATION == OP_WARP || SPECIALIZED_OPERATION == OP_REMAP)                                                               \
                process_sample(src, dst, other, p, v, i);                                                                                            \
            else                                                                                                                                     \
                geometry_pixels(src, dst, p, i);                                                                                                     \
        } else if (SPECIALIZED_CHANNELS == 2) {                                                                                                      \
            process_pixel<TWO, float2, FLOATING, 2>(reinterpret_cast<device const TWO *>(src),                                                       \
                                                    reinterpret_cast<device TWO *>(dst),                                                             \
                                                    reinterpret_cast<device const TWO *>(other),                                                     \
                                                    p,                                                                                               \
                                                    v,                                                                                               \
                                                    i);                                                                                              \
        } else if (SPECIALIZED_CHANNELS == 3) {                                                                                                      \
            process_pixel<THREE, float3, FLOATING, 3>(reinterpret_cast<device const THREE *>(src),                                                   \
                                                      reinterpret_cast<device THREE *>(dst),                                                         \
                                                      reinterpret_cast<device const THREE *>(other),                                                 \
                                                      p,                                                                                             \
                                                      v,                                                                                             \
                                                      i);                                                                                            \
        } else {                                                                                                                                     \
            process_pixel<FOUR, float4, FLOATING, 4>(reinterpret_cast<device const FOUR *>(src),                                                     \
                                                     reinterpret_cast<device FOUR *>(dst),                                                           \
                                                     reinterpret_cast<device const FOUR *>(other),                                                   \
                                                     p,                                                                                              \
                                                     v,                                                                                              \
                                                     i);                                                                                             \
        }                                                                                                                                            \
    }
PIXEL_KERNEL(pixel_u8, uchar, uchar2, packed_uchar3, uchar4, false)
PIXEL_KERNEL(pixel_f32, float, float2, packed_float3, float4, true)

constant uint TRANSPOSE_TILE = 16, TRANSPOSE_STRIDE = TRANSPOSE_TILE + 1;

// Transpose coalesced pixel tiles, falling back for small or partial groups.
template <typename P>
void transpose_pixels(device const P *src, device P *dst, constant uint *p, threadgroup P *tile, uint i, uint member, uint group, uint threads) {
    uint width = p[3], height = p[4], method = p[6];
    if (method == 0 || method == 1 || method == 3 || width < TRANSPOSE_TILE || height < TRANSPOSE_TILE ||
        threads != TRANSPOSE_TILE * TRANSPOSE_TILE) {
        geometry_pixels(src, dst, p, i);
        return;
    }
    uint columns = (width + TRANSPOSE_TILE - 1) >> 4; // TRANSPOSE_TILE == 16
    uint group_y, group_x;
    decompose_2d(group, columns, group_x, group_y);
    uint left = group_x << 4, top = group_y << 4;
    uint x = member & 15u, y = member >> 4;
    uint load_x = left + y, load_y = top + x;
    if (load_x < width && load_y < height) {
        uint sx = (method == 2 || method == 6) ? p[0] - 1 - load_y : load_y;
        uint sy = (method == 4 || method == 6) ? p[1] - 1 - load_x : load_x;
        tile[y * TRANSPOSE_STRIDE + x] = src[sy * p[0] + sx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (left + x < width && top + y < height)
        dst[(top + y) * width + left + x] = tile[x * TRANSPOSE_STRIDE + y];
}

#define TRANSPOSE_KERNEL(NAME, SAMPLE, TWO, THREE, FOUR)                                                                                             \
    kernel void NAME(device const SAMPLE *src [[buffer(0)]],                                                                                         \
                     device SAMPLE *dst [[buffer(1)]],                                                                                               \
                     constant uint *p [[buffer(3)]],                                                                                                 \
                     uint i [[thread_position_in_grid]],                                                                                             \
                     uint member [[thread_index_in_threadgroup]],                                                                                    \
                     uint group [[threadgroup_position_in_grid]],                                                                                    \
                     uint threads [[threads_per_threadgroup]]) {                                                                                     \
        if (SPECIALIZED_CHANNELS == 1) {                                                                                                             \
            threadgroup SAMPLE tile[TRANSPOSE_TILE * TRANSPOSE_STRIDE];                                                                              \
            transpose_pixels(src, dst, p, tile, i, member, group, threads);                                                                          \
        } else if (SPECIALIZED_CHANNELS == 2) {                                                                                                      \
            threadgroup TWO tile[TRANSPOSE_TILE * TRANSPOSE_STRIDE];                                                                                 \
            transpose_pixels(reinterpret_cast<device const TWO *>(src), reinterpret_cast<device TWO *>(dst), p, tile, i, member, group, threads);    \
        } else if (SPECIALIZED_CHANNELS == 3) {                                                                                                      \
            threadgroup THREE tile[TRANSPOSE_TILE * TRANSPOSE_STRIDE];                                                                               \
            transpose_pixels(                                                                                                                        \
                reinterpret_cast<device const THREE *>(src), reinterpret_cast<device THREE *>(dst), p, tile, i, member, group, threads);             \
        } else {                                                                                                                                     \
            threadgroup FOUR tile[TRANSPOSE_TILE * TRANSPOSE_STRIDE];                                                                                \
            transpose_pixels(reinterpret_cast<device const FOUR *>(src), reinterpret_cast<device FOUR *>(dst), p, tile, i, member, group, threads);  \
        }                                                                                                                                            \
    }
TRANSPOSE_KERNEL(transpose_u8, uchar, uchar2, packed_uchar3, uchar4)
TRANSPOSE_KERNEL(transpose_f32, float, float2, packed_float3, float4)