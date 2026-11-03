// Helper for fast channel/pixel decomposition when c is often 1, 2, or 4
inline void decompose_index(uint i, uint c, thread uint &pixel_idx, thread uint &band) {
    if ((c & (c - 1)) == 0) {
        uint shift = c == 4 ? 2u : (c == 2 ? 1u : 0u);
        pixel_idx = i >> shift;
        band = i & (c - 1);
    } else {
        pixel_idx = i / c;
        band = i - pixel_idx * c;
    }
}

// Route a scalar sample to its operation-specific implementation.
template <typename T>
void process_sample(device const T *src, device T *dst, device const T *other, constant uint *p, device const float *v, uint i) {
#ifdef BLANKET_METAL_SPECIALIZE
    const uint op = SPECIALIZED_OPERATION;
    const uint c = SPECIALIZED_CHANNELS;
#else
    const uint c = p[2], op = p[5];
#endif
    uint dw = p[3], dh = p[4], k = p[6];

    if (op == OP_BOX_BLUR && is_same_v<T, uchar> && k >= SLIDING_BLUR_RADIUS) {
        sliding_box_blur<T, uint>(src, dst, p, v, i, c);
        return;
    }
    if (op == OP_STATISTICS) {
        channel_statistics_scalar(src, dst, p, i, c);
        return;
    }

    uint num_pixels = dw * dh;
    if (op == OP_CHANNEL) {
        if (i < num_pixels)
            dst[i] = src[i * c + k];
        return;
    }
    if (op == OP_COLOR_LUT) {
        map_color_lut(src, dst, p, v, i);
        return;
    }
    if (i >= num_pixels * c)
        return;

    switch (op) {
    case OP_QUANTIZE: {
        float value = float(src[i]);
        value = isnan(value) ? 0.0f : saturate(value);
        float scale = v[0];
        dst[i] = T(floor(fma(value, scale, 0.5f)) / scale);
        return;
    }
    case OP_MASK: {
        uint pixel_idx = ((c & (c - 1)) == 0) ? (i >> (c == 4 ? 2u : (c == 2 ? 1u : 0u))) : (i / c);
        float mask = float(other[pixel_idx]) / v[0];
        mask = isnan(mask) ? 0.0f : saturate(mask);
        if (p[7] == 1)
            mask = 1.0f - mask;
        dst[i] = (mask == 0.0f) ? T(0) : T(float(src[i]) * mask);
        return;
    }
    case OP_WARP:
    case OP_REMAP:
        warp_sample(src, dst, other, p, v, i, op, c);
        return;
    case OP_TRANSFER: {
        uint band = ((c & (c - 1)) == 0) ? (i & (c - 1)) : (i % c);
        float value = float(src[i]);
        if (band < 3)
            value = color_transfer(value, k == 0);
        dst[i] = T(value);
        return;
    }
    case OP_CLAMP_ALPHA: {
        uint band = ((c & (c - 1)) == 0) ? (i & (c - 1)) : (i % c);
        T raw = src[i];
        if (band == c - 1) {
            float value = float(raw);
            dst[i] = T(isnan(value) ? 0.0f : saturate(value));
        } else {
            dst[i] = raw;
        }
        return;
    }
    case OP_ENHANCE:
        enhance_sample(src, dst, other, p, v, i, c);
        return;
    case OP_REPLACE_CHANNEL: {
        uint pixel_idx, band;
        decompose_index(i, c, pixel_idx, band);
        dst[i] = (band == k) ? other[pixel_idx] : src[i];
        return;
    }
    case OP_SET_ALPHA: {
        uint band = ((c & (c - 1)) == 0) ? (i & (c - 1)) : (i % c);
        dst[i] = (band == c - 1) ? store_sample<T>(v[0]) : src[i];
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
    case OP_RESIZE:
        dst[i] = store_sample<T>(resize_sample(src, p, v, i, c));
        return;
    case OP_CONVOLVE:
        dst[i] = store_sample<T>(convolve_sample(src, p, v, i, c));
        return;
    default:
        dst[i] = store_sample<T>(arithmetic_sample(src, other, p, v, i, op, c));
        return;
    }
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