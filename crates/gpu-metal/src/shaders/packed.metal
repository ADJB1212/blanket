// Native packed-byte entry points.

#ifdef BLANKET_METAL_SPECIALIZE

// Dispatch packed LA/RGB resize, convolution, or fractional box blur.
kernel void color_u8(device const uchar *s [[buffer(0)]], device uchar *d [[buffer(1)]], constant uint *p [[buffer(3)]],
                     device const float *v [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (SPECIALIZED_CHANNELS == 3) {
        if (SPECIALIZED_OPERATION == OP_BOX_BLUR)
            box_pixels<packed_uchar3, uint3>(reinterpret_cast<device const packed_uchar3 *>(s), reinterpret_cast<device packed_uchar3 *>(d), p, v, i);
        else
            filter_pixels<packed_uchar3, int3, float3>(
                reinterpret_cast<device const packed_uchar3 *>(s), reinterpret_cast<device packed_uchar3 *>(d), p, v, i);
    } else {
        if (SPECIALIZED_OPERATION == OP_BOX_BLUR)
            box_pixels<uchar2, uint2>(reinterpret_cast<device const uchar2 *>(s), reinterpret_cast<device uchar2 *>(d), p, v, i);
        else
            filter_pixels<uchar2, int2, float2>(reinterpret_cast<device const uchar2 *>(s), reinterpret_cast<device uchar2 *>(d), p, v, i);
    }
}

// Process four byte samples at a time with a scalar tail.
kernel void packed_u8(device const uchar *s [[buffer(0)]], device uchar *d [[buffer(1)]], device const uchar *o [[buffer(2)]],
                      constant uint *p [[buffer(3)]], device const float *v [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    constexpr uint PACKED_SAMPLES = 4;
    uint first = i * PACKED_SAMPLES, count = p[3] * p[4] * p[2];
    if (first + PACKED_SAMPLES > count) {
        for (uint sample = first; sample < count; sample++)
            process_sample(s, d, o, p, v, sample);
        return;
    }
    if (SPECIALIZED_OPERATION == OP_RANK) {
        uint size = p[6], rank = p[7];
        uchar4 result;
        if (rank == 0 || rank == size * size - 1) {
            result = uchar4(rank == 0 ? 255 : 0);
            for (uint y = 0; y < size; y++)
                for (uint x = 0; x < size; x++) {
                    uchar4 value = rank_samples(s, first, p[0], p[1], int(x) - int(size / 2), int(y) - int(size / 2));
                    result = rank == 0 ? min(result, value) : max(result, value);
                }
        } else if (size == 3) {
            uchar4 samples[9];
#pragma unroll
            for (uint y = 0; y < 3; y++) {
#pragma unroll
                for (uint x = 0; x < 3; x++)
                    samples[y * 3 + x] = rank_samples(s, first, p[0], p[1], int(x) - 1, int(y) - 1);
            }
            if (rank == 4) {
                result = median9(samples);
            } else {
#pragma unroll
                for (uint phase = 0; phase < 9; phase++) {
#pragma unroll
                    for (uint j = phase % 2; j < 8; j += 2) {
                        rank_swap(samples[j], samples[j + 1]);
                    }
                }
                result = samples[rank];
            }
        } else {
            for (uint lane = 0; lane < PACKED_SAMPLES; lane++)
                process_sample(s, d, o, p, v, first + lane);
            return;
        }
        reinterpret_cast<device uchar4 *>(d)[i] = result;
        return;
    }
    if (SPECIALIZED_OPERATION == OP_RESIZE) {
        filter_pixels<uchar4, int4, float4>(reinterpret_cast<device const uchar4 *>(s), reinterpret_cast<device uchar4 *>(d), p, v, i);
        return;
    }
    if (SPECIALIZED_OPERATION == OP_BOX_BLUR) {
        if (SPECIALIZED_CHANNELS != 4) {
            int radius = int(p[6]);
            bool horizontal = p[7] == 0;
            uint4 sum = uint4(0);
            for (int delta = -radius; delta <= radius; delta++)
                sum += uint4(rank_samples(s, first, p[0], p[1], horizontal ? delta : 0, horizontal ? 0 : delta));
            uint4 fixed = sum * uint(v[0]) + BOX_ROUND;
            if (v[1] != 0) {
                uint4 left = uint4(rank_samples(s, first, p[0], p[1], horizontal ? -radius - 1 : 0, horizontal ? 0 : -radius - 1));
                uint4 right = uint4(rank_samples(s, first, p[0], p[1], horizontal ? radius + 1 : 0, horizontal ? 0 : radius + 1));
                fixed += (left + right) * uint(v[1]);
            }
            reinterpret_cast<device uchar4 *>(d)[i] = uchar4(fixed >> BOX_BITS);
            return;
        }
        box_pixels<uchar4, uint4>(reinterpret_cast<device const uchar4 *>(s), reinterpret_cast<device uchar4 *>(d), p, v, i);
        return;
    }
    if (SPECIALIZED_OPERATION == OP_CONVOLVE) {
        filter_pixels<uchar4, int4, float4>(reinterpret_cast<device const uchar4 *>(s), reinterpret_cast<device uchar4 *>(d), p, v, i);
        return;
    }
    uchar4 bytes = reinterpret_cast<device const uchar4 *>(s)[i];
    if (SPECIALIZED_OPERATION == OP_POINT) {
        uchar4 result;
        for (uint lane = 0; lane < PACKED_SAMPLES; lane++)
            result[lane] = uchar(v[((first + lane) % SPECIALIZED_CHANNELS) * 256 + uint(bytes[lane])]);
        reinterpret_cast<device uchar4 *>(d)[i] = result;
        return;
    }
    if (SPECIALIZED_OPERATION == OP_BINARY) {
        uint4 a = uint4(bytes), b = uint4(reinterpret_cast<device const uchar4 *>(o)[i]);
        uint kind = p[6];
        float4 result;
        switch (kind) {
        case 0:
            result = float4(max(a, b) - min(a, b));
            break;
        case 1:
            result = float4(a * b / 255);
            break;
        case 2:
            result = float4(255 - (255 - a) * (255 - b) / 255);
            break;
        case 3:
            result = float4(max(a, b));
            break;
        case 4:
            result = float4(min(a, b));
            break;
        default:
            result = (kind == 5 ? float4(a + b) : float4(int4(a) - int4(b))) / v[0] + v[1];
            for (uint lane = 0; lane < PACKED_SAMPLES; lane++)
                if (!isfinite(result[lane]) || result[lane] >= 2147483648.0f || result[lane] < -2147483648.0f)
                    result[lane] = 0;
            result = floor(result);
        }
        reinterpret_cast<device uchar4 *>(d)[i] = uchar4(clamp(result, 0.0f, 255.0f));
        return;
    }
    if (SPECIALIZED_OPERATION == OP_COPY) {
        reinterpret_cast<device uchar4 *>(d)[i] = bytes;
        return;
    }
    if (SPECIALIZED_OPERATION == OP_INVERT) {
        reinterpret_cast<device uchar4 *>(d)[i] = uchar4(255) - bytes;
        return;
    }
    float4 a = float4(bytes), result;
    if (SPECIALIZED_OPERATION == OP_BLEND) {
        float4 b = float4(reinterpret_cast<device const uchar4 *>(o)[i]);
        result = a + (b - a) * v[0];
    } else {
        result = a * v[0];
        if (SPECIALIZED_CHANNELS == 4)
            result.w = a.w;
    }
    reinterpret_cast<device uchar4 *>(d)[i] = uchar4(clamp(floor(result), 0.0f, 255.0f));
}

#endif
