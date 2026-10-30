// Sample arithmetic, lookup, alpha, and enhancement operations.

// Blend a sample with grayscale, mean luminance, or a smoothed neighborhood.
template <typename T>
void enhance_sample(device const T *src, device T *dst, device const T *other, constant uint *p, device const float *v, uint i, uint c) {
    uint sw = p[0], sh = p[1], dw = p[3], k = p[6];
    uint band = i % c, x = (i / c) % dw, y = i / (c * dw);
    float original = float(src[i]), base = original;
    if ((k != 2 && (c == 2 || c == 4) && band == c - 1) ||
        (k == 2 && (x == 0 || y == 0 || x + 1 >= sw || y + 1 >= sh || (c == 4 && band == c - 1)))) {
        dst[i] = src[i];
        return;
    }
    if (k == 0)
        base = luminance(src, i - band, c);
    else if (k == 1) {
        if (is_same_v<T, uchar>) {
            device const uint *words = reinterpret_cast<device const uint *>(other);
            ulong total = ulong(words[0]) | (ulong(words[1]) << 32);
            base = float((total * 2 + p[7]) / (ulong(p[7]) * 2));
        } else
            base = float(other[0]) / float(p[7]);
    } else {
        float sum = original * 4;
        for (uint ky = 0; ky < 3; ky++)
            for (uint kx = 0; kx < 3; kx++) {
                sum += float(src[((y + ky - 1) * sw + x + kx - 1) * c + band]);
            }
        base = is_same_v<T, uchar> ? float((uint(sum) + 6) / 13) : sum / 13;
    }
    float value = v[0] == 0 ? base : fma(v[0], original - base, base);
    if (is_same_v<T, uchar>)
        value = floor(value);
    dst[i] = store_sample<T>(value);
    return;
}

// Apply pointwise arithmetic, lookup, or alpha math to one sample.
template <typename T>
float arithmetic_sample(device const T *src, device const T *other, constant uint *p, device const float *v, uint i, uint op, uint c) {
    uint band = i % c, k = p[6];
    float a = read_sample(src, i), out = a;
    float maxval = is_same_v<T, uchar> ? 255.0f : 1.0f;
    if (op == OP_BLEND) {
        out = a + (read_sample(other, i) - a) * v[0];
        if (is_same_v<T, uchar>)
            out = floor(out);
    } else if (op == OP_ALPHA_COMPOSITE) {
        uint base = i - band;
        float da = read_sample(src, base + c - 1) / maxval, sa = read_sample(other, base + c - 1) / maxval;
        da = isnan(da) ? 0.0f : clamp(da, 0.0f, 1.0f);
        sa = isnan(sa) ? 0.0f : clamp(sa, 0.0f, 1.0f);
        float alpha = sa + da * (1 - sa);
        out = band == c - 1 ? alpha * maxval
                            : (alpha == 0 ? 0 : ((sa == 0 ? 0 : read_sample(other, i) * sa) + (da == 0 ? 0 : a * da) * (1 - sa)) / alpha);
    } else if (op == OP_PREMULTIPLY || op == OP_UNPREMULTIPLY) {
        float raw = read_sample(src, i - band + c - 1);
        float alpha = isnan(raw) ? 0.0f : clamp(raw / maxval, 0.0f, 1.0f);
        out = band == c - 1 ? alpha * maxval
                            : (alpha == 0 ? ((op == OP_UNPREMULTIPLY && is_same_v<T, uchar>) ? a : 0.0f)
                                          : (op == OP_PREMULTIPLY ? a * alpha : (is_same_v<T, uchar> ? a * maxval / raw : a / alpha)));
        if (op == OP_UNPREMULTIPLY && is_same_v<T, uchar>)
            out = floor(out);
    } else if (op == OP_BRIGHTNESS) {
        if (!(c == 4 && band == c - 1)) {
            out = a * v[0];
            if (is_same_v<T, uchar>)
                out = floor(out);
        }
    } else if (op == OP_INVERT) {
        out = maxval - a;
    } else if (op == OP_POINT) {
        out = v[band * 256 + uint(a)];
    } else if (op == OP_UNSHARP) {
        float difference = a - read_sample(other, i);
        if (abs(difference) > v[0]) {
            if (is_same_v<T, uchar>)
                out = float(long(a) + long(difference) * int(k) / 100);
            else
                out = a + difference * float(int(k)) / 100.0f;
        }
    } else if (op == OP_BINARY) {
        float b = read_sample(other, i);
        switch (k) {
        case 0:
            out = abs(a - b);
            break;
        case 1:
            out = a * b / maxval;
            break;
        case 2:
            out = maxval - (maxval - a) * (maxval - b) / maxval;
            break;
        case 3:
            out = (isnan(a) || isnan(b)) ? NAN : max(a, b);
            break;
        case 4:
            out = (isnan(a) || isnan(b)) ? NAN : min(a, b);
            break;
        case 5:
            out = (a + b) / v[0] + v[1];
            break;
        case 6:
            out = (a - b) / v[0] + v[1];
            break;
        }
        if (is_same_v<T, uchar>) {
            if (k == 1)
                out = float(uint(a) * uint(b) / 255);
            else if (k == 2)
                out = float(255 - (255 - uint(a)) * (255 - uint(b)) / 255);
            else if (k >= 5) {
                if (!isfinite(out) || out >= 2147483648.0f || out < -2147483648.0f)
                    out = 0;
                out = floor(out);
            }
        }
    }
    return out;
}
