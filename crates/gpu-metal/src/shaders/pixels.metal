// Copy one packed pixel through crop, transpose, or wrapped-offset geometry.
template <typename P> void geometry_pixels(device const P *src, device P *dst, constant uint *p, uint i) {
    uint sw = p[0], sh = p[1], width = p[3], height = p[4];
    if (i >= width * height)
        return;
    uint x = i % width, y = i / width, sx = x, sy = y;
    if (SPECIALIZED_OPERATION == OP_CROP) {
        long cx = long(x) + as_type<int>(p[6]), cy = long(y) + as_type<int>(p[7]);
        dst[i] = cx >= 0 && cy >= 0 && cx < sw && cy < sh ? src[uint(cy) * sw + uint(cx)] : P(0);
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
        sx = x >= p[6] ? x - p[6] : x + sw - p[6];
        sy = y >= p[7] ? y - p[7] : y + sh - p[7];
    }
    dst[i] = src[sy * sw + sx];
}

// Composite or change alpha association for one packed pixel.
template <typename P, typename F, bool Floating, uint Channels>
void alpha_pixels(device const P *src, device P *dst, device const P *other, constant uint *p, uint i) {
    if (i >= p[3] * p[4])
        return;
    F a = F(src[i]), result;
    float maxval = Floating ? 1.0f : 255.0f;
    float raw = float(a[Channels - 1]);
    float da = isnan(raw) ? 0.0f : clamp(raw / maxval, 0.0f, 1.0f);
    if (SPECIALIZED_OPERATION == OP_ALPHA_COMPOSITE) {
        F b = F(other[i]);
        float sa = float(b[Channels - 1]) / maxval;
        sa = isnan(sa) ? 0.0f : clamp(sa, 0.0f, 1.0f);
        float alpha = sa + da * (1 - sa);
        result = alpha == 0 ? F(0) : ((sa == 0 ? F(0) : b * sa) + (da == 0 ? F(0) : a * da) * (1 - sa)) / alpha;
        result[Channels - 1] = alpha * maxval;
    } else {
        result = da == 0 ? ((SPECIALIZED_OPERATION == OP_UNPREMULTIPLY && !Floating) ? a : F(0))
                         : (SPECIALIZED_OPERATION == OP_PREMULTIPLY ? a * da : (Floating ? a / da : a * maxval / raw));
        result[Channels - 1] = da * maxval;
        if (SPECIALIZED_OPERATION == OP_UNPREMULTIPLY && !Floating)
            result = floor(result);
    }
    if (Floating)
        dst[i] = P(result);
    else
        dst[i] = P(clamp(floor(result + 0.5f), 0.0f, 255.0f));
}

// Dispatch a packed pixel to geometry or alpha processing.
template <typename P, typename F, bool Floating, uint Channels>
void process_pixel(device const P *src, device P *dst, device const P *other, constant uint *p, uint i) {
    if (SPECIALIZED_OPERATION == OP_ALPHA_COMPOSITE || SPECIALIZED_OPERATION == OP_PREMULTIPLY || SPECIALIZED_OPERATION == OP_UNPREMULTIPLY)
        alpha_pixels<P, F, Floating, Channels>(src, dst, other, p, i);
    else
        geometry_pixels(src, dst, p, i);
}

#define PIXEL_KERNEL(NAME, SAMPLE, TWO, THREE, FOUR, FLOATING)                                                                                       \
    /* Dispatch geometry or alpha processing for the selected packed pixel type. */                                                                  \
    kernel void NAME(device const SAMPLE *src [[buffer(0)]],                                                                                         \
                     device SAMPLE *dst [[buffer(1)]],                                                                                               \
                     device const SAMPLE *other [[buffer(2)]],                                                                                       \
                     constant uint *p [[buffer(3)]],                                                                                                 \
                     uint i [[thread_position_in_grid]]) {                                                                                           \
        if (SPECIALIZED_CHANNELS == 1)                                                                                                               \
            geometry_pixels(src, dst, p, i);                                                                                                         \
        else if (SPECIALIZED_CHANNELS == 2)                                                                                                          \
            process_pixel<TWO, float2, FLOATING, 2>(                                                                                                 \
                reinterpret_cast<device const TWO *>(src), reinterpret_cast<device TWO *>(dst), reinterpret_cast<device const TWO *>(other), p, i);  \
        else if (SPECIALIZED_CHANNELS == 3)                                                                                                          \
            process_pixel<THREE, float3, FLOATING, 3>(reinterpret_cast<device const THREE *>(src),                                                   \
                                                      reinterpret_cast<device THREE *>(dst),                                                         \
                                                      reinterpret_cast<device const THREE *>(other),                                                 \
                                                      p,                                                                                             \
                                                      i);                                                                                            \
        else                                                                                                                                         \
            process_pixel<FOUR, float4, FLOATING, 4>(reinterpret_cast<device const FOUR *>(src),                                                     \
                                                     reinterpret_cast<device FOUR *>(dst),                                                           \
                                                     reinterpret_cast<device const FOUR *>(other),                                                   \
                                                     p,                                                                                              \
                                                     i);                                                                                             \
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
    uint columns = (width + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE;
    uint left = (group % columns) * TRANSPOSE_TILE, top = (group / columns) * TRANSPOSE_TILE;
    uint x = member % TRANSPOSE_TILE, y = member / TRANSPOSE_TILE;
    uint load_x = left + y, load_y = top + x;
    if (load_x < width && load_y < height) {
        uint sx = method == 2 || method == 6 ? p[0] - 1 - load_y : load_y;
        uint sy = method == 4 || method == 6 ? p[1] - 1 - load_x : load_x;
        tile[y * TRANSPOSE_STRIDE + x] = src[sy * p[0] + sx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (left + x < width && top + y < height)
        dst[(top + y) * width + left + x] = tile[x * TRANSPOSE_STRIDE + y];
}

#define TRANSPOSE_KERNEL(NAME, SAMPLE, TWO, THREE, FOUR)                                                                                             \
    /* Dispatch tiled transpose with storage matching the image channel count. */                                                                    \
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
