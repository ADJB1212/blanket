// Shared scalar crop, transpose, wrap, and resampling routines.

// Interpolate four neighboring samples with the warp cubic polynomial.
float warp_cubic(float4 v, float t) {
    float a = -v[0] + v[1] - v[2] + v[3];
    float b = 2 * v[0] - 2 * v[1] + v[2] - v[3];
    float c = -v[0] + v[2];
    return ((a * t + b) * t + c) * t + v[1];
}

// Resample one channel through an affine, perspective, or coordinate map.
template <typename T>
void warp_sample(device const T *src, device T *dst, device const T *other, constant uint *p, device const float *v, uint i, uint op, uint c) {
    uint sw = p[0], sh = p[1], dw = p[3], k = p[6];
    uint band = i % c, x = (i / c) % dw, y = i / (c * dw);
    float px = float(x) + 0.5f, py = float(y) + 0.5f;
    float denominator = v[6] * px + v[7] * py + 1;
    float sx = (v[0] * px + v[1] * py + v[2]) / denominator;
    float sy = (v[3] * px + v[4] * py + v[5]) / denominator;
    if (op == OP_REMAP) {
        device const float *coordinates = reinterpret_cast<device const float *>(other);
        sx = coordinates[(i / c) * 3];
        sy = coordinates[(i / c) * 3 + 1];
    }
    if (!isfinite(sx) || !isfinite(sy) || sx < 0 || sy < 0 || sx >= sw || sy >= sh) {
        dst[i] = store_sample<T>(v[8 + band]);
        return;
    }
    if (k == 0) {
        dst[i] = src[(uint(sy) * sw + uint(sx)) * c + band];
        return;
    }
    sx -= 0.5f;
    sy -= 0.5f;
    int ix = int(floor(sx)), iy = int(floor(sy));
    float fx = sx - floor(sx), fy = sy - floor(sy), result = 0;
    if (k == 2) {
        float4 rows;
        for (int ky = -1; ky <= 2; ky++) {
            float4 taps;
            for (int kx = -1; kx <= 2; kx++) {
                uint index = (uint(clamp(iy + ky, 0, int(sh) - 1)) * sw + uint(clamp(ix + kx, 0, int(sw) - 1))) * c + band;
                taps[kx + 1] = float(src[index]);
            }
            rows[ky + 1] = warp_cubic(taps, fx);
        }
        result = warp_cubic(rows, fy);
        dst[i] = store_sample<T>(is_same_v<T, uchar> ? floor(result) : result);
        return;
    }
    int first = 0, last = 1;
    for (int ky = first; ky <= last; ky++) {
        float wy = ky == 0 ? 1 - fy : fy;
        float row = 0;
        for (int kx = first; kx <= last; kx++) {
            float wx = kx == 0 ? 1 - fx : fx;
            uint index = (uint(clamp(iy + ky, 0, int(sh) - 1)) * sw + uint(clamp(ix + kx, 0, int(sw) - 1))) * c + band;
            row += float(src[index]) * wx;
        }
        result += row * wy;
    }
    dst[i] = store_sample<T>(is_same_v<T, uchar> ? floor(result) : result);
    return;
}

// Crop, transpose, or wrap one channel without changing sample bits.
template <typename T> void geometry_sample(device const T *src, device T *dst, constant uint *p, uint i, uint op, uint c) {
    uint sw = p[0], sh = p[1], dw = p[3], k = p[6];
    uint band = i % c, x = (i / c) % dw, y = i / (c * dw);
    if (op == OP_CROP) {
        long sx = long(x) + int(k), sy = long(y) + int(p[7]);
        dst[i] = (sx >= 0 && sy >= 0 && sx < sw && sy < sh) ? src[(uint(sy) * sw + uint(sx)) * c + band] : T(0);
        return;
    }
    if (op == OP_TRANSPOSE) {
        uint sx = x, sy = y;
        switch (k) {
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
        dst[i] = src[(sy * sw + sx) * c + band];
        return;
    }
    if (op == OP_OFFSET) {
        uint sx = x >= k ? x - k : x + sw - k, sy = y >= p[7] ? y - p[7] : y + sh - p[7];
        dst[i] = src[(sy * sw + sx) * c + band];
        return;
    }
}
