kernel void cast_f32_f16(const device float *src [[buffer(0)]], device half *dst [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    dst[i] = half(src[i]);
}

kernel void cast_f16_f32(const device half *src [[buffer(0)]], device float *dst [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    dst[i] = float(src[i]);
}
