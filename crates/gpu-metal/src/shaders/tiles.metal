// Copy one packed pixel between source and destination tile rectangles.
template <typename T> void copy_tile(device const T *src, device T *dst, constant uint *p, uint i) {
    uint width = p[9], height = p[10];
    if (i >= width * height)
        return;
    uint x = i % width, y = i / width;
    dst[(y + p[8]) * p[3] + x + p[7]] = src[(y + p[6]) * p[0] + x + p[5]];
}

#define TILE_KERNEL(NAME, SAMPLE, TWO, THREE, FOUR)                                                                                                  \
    /* Dispatch a tile copy with storage matching the image channel count. */                                                                        \
    kernel void NAME(device const SAMPLE *src [[buffer(0)]],                                                                                         \
                     device SAMPLE *dst [[buffer(1)]],                                                                                               \
                     constant uint *p [[buffer(3)]],                                                                                                 \
                     uint i [[thread_position_in_grid]]) {                                                                                           \
        if (p[2] == 4)                                                                                                                               \
            copy_tile(reinterpret_cast<device const FOUR *>(src), reinterpret_cast<device FOUR *>(dst), p, i);                                       \
        else if (p[2] == 3)                                                                                                                          \
            copy_tile(reinterpret_cast<device const THREE *>(src), reinterpret_cast<device THREE *>(dst), p, i);                                     \
        else if (p[2] == 2)                                                                                                                          \
            copy_tile(reinterpret_cast<device const TWO *>(src), reinterpret_cast<device TWO *>(dst), p, i);                                         \
        else                                                                                                                                         \
            copy_tile(src, dst, p, i);                                                                                                               \
    }
TILE_KERNEL(tile_u8, uchar, uchar2, packed_uchar3, uchar4)
TILE_KERNEL(tile_f32, float, float2, packed_float3, float4)
