// Shared scalar entry points and operation dispatch.

// Route a scalar sample to its operation-specific implementation.
template <typename T>
void process_sample(device const T *src, device T *dst, device const T *other, constant uint *p, device const float *v, uint i) {
    uint c = p[2], dw = p[3], dh = p[4], op = p[5], k = p[6];
#ifdef BLANKET_METAL_SPECIALIZE
    op = SPECIALIZED_OPERATION;
    c = SPECIALIZED_CHANNELS;
#endif
    if (op == OP_BOX_BLUR && is_same_v<T, uchar> && k >= SLIDING_BLUR_RADIUS) {
        sliding_box_blur<T, uint>(src, dst, p, v, i, c);
        return;
    }
    if (op == OP_STATISTICS) {
        channel_statistics_scalar(src, dst, p, i, c);
        return;
    }
    if (op == OP_CHANNEL) {
        if (i < dw * dh)
            dst[i] = src[i * c + k];
        return;
    }
    if (op == OP_COLOR_LUT) {
        map_color_lut(src, dst, p, v, i);
        return;
    }
    if (i >= dw * dh * c)
        return;
    uint band = i % c;
    switch (op) {
    case OP_QUANTIZE: {
        float value = float(src[i]);
        value = isnan(value) ? 0.0f : clamp(value, 0.0f, 1.0f);
        dst[i] = T(floor(value * v[0] + 0.5f) / v[0]);
        return;
    }
    case OP_MASK: {
        float mask = float(other[i / c]) / v[0];
        mask = isnan(mask) ? 0.0f : clamp(mask, 0.0f, 1.0f);
        if (p[7] == 1)
            mask = 1.0f - mask;
        dst[i] = mask == 0 ? T(0) : T(float(src[i]) * mask);
        return;
    }
    case OP_WARP:
    case OP_REMAP:
        warp_sample(src, dst, other, p, v, i, op, c);
        return;
    case OP_TRANSFER:
    case OP_CLAMP_ALPHA: {
        float value = float(src[i]);
        if (op == OP_TRANSFER && band < 3)
            value = color_transfer(value, k == 0);
        if (op == OP_CLAMP_ALPHA && band == c - 1)
            value = isnan(value) ? 0.0f : clamp(value, 0.0f, 1.0f);
        dst[i] = T(value);
        return;
    }
    case OP_ENHANCE:
        enhance_sample(src, dst, other, p, v, i, c);
        return;
    case OP_REPLACE_CHANNEL:
    case OP_SET_ALPHA: {
        dst[i] = op == OP_REPLACE_CHANNEL ? (band == k ? other[i / c] : src[i]) : (band == c - 1 ? store_sample<T>(v[0]) : src[i]);
        return;
    }
    case OP_BOX_BLUR:
        box_blur_sample(src, dst, p, v, i, c);
        return;
    case OP_RANK:
        rank_sample(src, dst, p, i, c);
        return;
    case OP_CROP:
    case OP_TRANSPOSE:
    case OP_OFFSET:
        geometry_sample(src, dst, p, i, op, c);
        return;
    default:
        break;
    }
    float value;
    if (op == OP_RESIZE)
        value = resize_sample(src, p, v, i, c);
    else if (op == OP_CONVOLVE)
        value = convolve_sample(src, p, v, i, c);
    else
        value = arithmetic_sample(src, other, p, v, i, op, c);
    dst[i] = store_sample<T>(value);
}

// Dispatch one byte sample through the shared scalar operation path.
kernel void image_u8(device const uchar *s [[buffer(0)]], device uchar *d [[buffer(1)]], device const uchar *o [[buffer(2)]],
                     constant uint *p [[buffer(3)]], device const float *v [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    process_sample(s, d, o, p, v, i);
}

// Dispatch one float sample through the shared scalar operation path.
kernel void image_f32(device const float *s [[buffer(0)]], device float *d [[buffer(1)]], device const float *o [[buffer(2)]],
                      constant uint *p [[buffer(3)]], device const float *v [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    process_sample(s, d, o, p, v, i);
}
