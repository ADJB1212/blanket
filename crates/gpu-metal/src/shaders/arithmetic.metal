// Blend a sample with grayscale, mean luminance, or a smoothed neighborhood.
template <typename T>
void enhance_sample(device const T *src, device T *dst, device const T *other, constant uint *p, device const float *v, uint i, uint c) {
    uint sw = p[0], sh = p[1], dw = p[3], k = p[6];

    // Fast div/mod replacement: 2 divisions instead of 2 divisions + 2 modulos
    uint pixel_idx = i / c;
    uint band = i - pixel_idx * c;
    uint y = pixel_idx / dw;
    uint x = pixel_idx - y * dw;

    T raw_src = src[i];
    if ((k != 2 && (c == 2 || c == 4) && band == c - 1) ||
        (k == 2 && (x == 0 || y == 0 || x + 1 >= sw || y + 1 >= sh || (c == 4 && band == c - 1)))) {
        dst[i] = raw_src;
        return;
    }

    float original = float(raw_src), base = original;
    if (k == 0) {
        base = luminance(src, pixel_idx * c, c);
    } else if (k == 1) {
        uint count = p[7];
        if (is_same_v<T, uchar>) {
            device const uint *words = reinterpret_cast<device const uint *>(other);
            ulong total = ulong(words[0]) | (ulong(words[1]) << 32);
            base = float((total * 2 + count) / (ulong(count) * 2));
        } else {
            base = float(other[0]) / float(count);
        }
    } else {
        // Precompute row stride and center pointer to eliminate 9x 2D index multiplications
        uint stride = sw * c;
        device const T *center = src + (y * sw + x) * c + band;
        device const T *row0 = center - stride;
        device const T *row1 = center;
        device const T *row2 = center + stride;

        int ic = int(c);
        float sum = fma(original,
                        4.0f,
                        float(row0[-ic]) + float(row0[0]) + float(row0[ic]) + float(row1[-ic]) + float(row1[0]) + float(row1[ic]) + float(row2[-ic]) +
                            float(row2[0]) + float(row2[ic]));

        base = is_same_v<T, uchar> ? float((uint(sum) + 6u) / 13u) : sum * (1.0f / 13.0f);
    }

    float factor = v[0];
    float value = (factor == 0.0f) ? base : fma(factor, original - base, base);
    if (is_same_v<T, uchar>)
        value = floor(value);
    dst[i] = store_sample<T>(value);
}

// Apply pointwise arithmetic, lookup, or alpha math to one sample.
template <typename T>
float arithmetic_sample(device const T *src, device const T *other, constant uint *p, device const float *v, uint i, uint op, uint c) {
    // Bitwise mask when c is power of 2 (1, 2, 4), fallback to modulo
    uint band = ((c & (c - 1)) == 0) ? (i & (c - 1)) : (i % c);
    uint k = p[6];
    float a = read_sample(src, i), out = a;
    constexpr float maxval = is_same_v<T, uchar> ? 255.0f : 1.0f;
    constexpr float inv_maxval = 1.0f / maxval;

    switch (op) {
    case OP_BLEND: {
        out = fma(read_sample(other, i) - a, v[0], a);
        if (is_same_v<T, uchar>)
            out = floor(out);
        break;
    }
    case OP_ALPHA_COMPOSITE: {
        uint alpha_idx = i - band + c - 1;
        float da = read_sample(src, alpha_idx) * inv_maxval;
        float sa = read_sample(other, alpha_idx) * inv_maxval;
        da = isnan(da) ? 0.0f : saturate(da);
        sa = isnan(sa) ? 0.0f : saturate(sa);

        if (is_same_v<T, uchar> && sa == 0.0f)
            return a;
        if (sa == 1.0f)
            return band == c - 1 ? maxval : read_sample(other, i);

        float inv_sa = 1.0f - sa;
        float alpha = fma(da, inv_sa, sa);
        if (band == c - 1) {
            out = alpha * maxval;
        } else {
            out = (alpha == 0.0f) ? 0.0f : ((sa == 0.0f ? 0.0f : read_sample(other, i) * sa) + (da == 0.0f ? 0.0f : a * da) * inv_sa) / alpha;
        }
        break;
    }
    case OP_PREMULTIPLY:
    case OP_UNPREMULTIPLY: {
        float raw = read_sample(src, i - band + c - 1);
        float alpha = isnan(raw) ? 0.0f : saturate(raw * inv_maxval);
        if (band == c - 1) {
            out = alpha * maxval;
        } else if (alpha == 0.0f) {
            out = (op == OP_UNPREMULTIPLY && is_same_v<T, uchar>) ? a : 0.0f;
        } else if (op == OP_PREMULTIPLY) {
            out = a * alpha;
        } else {
            out = is_same_v<T, uchar> ? floor((a * maxval) / raw) : (a / alpha);
        }
        break;
    }
    case OP_BRIGHTNESS: {
        if (!(c == 4 && band == 3)) {
            out = a * v[0];
            if (is_same_v<T, uchar>)
                out = floor(out);
        }
        break;
    }
    case OP_INVERT: {
        out = maxval - a;
        break;
    }
    case OP_POINT: {
        out = v[(band << 8) + uint(a)];
        break;
    }
    case OP_UNSHARP: {
        float difference = a - read_sample(other, i);
        if (abs(difference) > v[0]) {
            if (is_same_v<T, uchar>)
                out = float(int(a) + (int(difference) * int(k)) / 100);
            else
                out = fma(difference, float(int(k)) * 0.01f, a);
        }
        break;
    }
    case OP_BINARY: {
        float b = read_sample(other, i);
        switch (k) {
        case 0:
            out = abs(a - b);
            break;
        case 1:
            out = is_same_v<T, uchar> ? float((uint(a) * uint(b)) / 255u) : (a * b * inv_maxval);
            break;
        case 2:
            out = is_same_v<T, uchar> ? float(255u - ((255u - uint(a)) * (255u - uint(b))) / 255u)
                                      : (maxval - (maxval - a) * (maxval - b) * inv_maxval);
            break;
        case 3:
            out = (isnan(a) || isnan(b)) ? NAN : fmax(a, b);
            break;
        case 4:
            out = (isnan(a) || isnan(b)) ? NAN : fmin(a, b);
            break;
        case 5:
        case 6: {
            float num = (k == 5) ? (a + b) : (a - b);
            out = num / v[0] + v[1];
            if (is_same_v<T, uchar>) {
                if (!isfinite(out) || out >= 2147483648.0f || out < -2147483648.0f)
                    out = 0.0f;
                out = floor(out);
            }
            break;
        }
        }
        break;
    }
    }
    return out;
}