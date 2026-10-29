template <typename T> void copy_tile(device const T *src, device T *dst, constant uint *p, uint i) {
    uint channels = p[2], width = p[9], height = p[10];
    if (i >= width * height * channels)
        return;
    uint band = i % channels, x = (i / channels) % width, y = i / (channels * width);
    dst[((y + p[8]) * p[3] + x + p[7]) * channels + band] = src[((y + p[6]) * p[0] + x + p[5]) * channels + band];
}
kernel void tile_u8(device const uchar *src [[buffer(0)]], device uchar *dst [[buffer(1)]], constant uint *p [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
    copy_tile(src, dst, p, i);
}
kernel void tile_f32(device const float *src [[buffer(0)]], device float *dst [[buffer(1)]], constant uint *p [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    copy_tile(src, dst, p, i);
}
