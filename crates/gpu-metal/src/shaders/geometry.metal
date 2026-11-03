// Shared scalar crop, transpose, wrap, and resampling routines.

// Interpolate four neighboring samples with the warp cubic polynomial.
inline float warp_cubic(float4 v, float t) {
    float a = -v.x + v.y - v.z + v.w;
    float b = 2.0f * (v.x - v.y) + v.z - v.w;
    float c = -v.x + v.z;
    return fma(fma(fma(a, t, b), t, c), t, v.y);
}

// Resample one channel through an affine, perspective, or coordinate map.
template <typename T>
void warp_sample(device const T *src, device T *dst, device const T *other, constant uint *p, device const float *v, uint i, uint op, uint c) {
    uint sw = p[0], sh = p[1], dw = p[3], k = p[6];
    uint pixel_idx, band, x, y;
    decompose_channel(i, c, pixel_idx, band);
    decompose_2d(pixel_idx, dw, x, y);

    float sx, sy;
    if (op == OP_REMAP) {
        device const float *coordinates = reinterpret_cast<device const float *>(other);
        sx = coordinates[pixel_idx * 3];
        sy = coordinates[pixel_idx * 3 + 1];
    } else {
        float px = float(x) + 0.5f, py = float(y) + 0.5f;
        float inv_denom = 1.0f / fma(v[6], px, fma(v[7], py, 1.0f));
        sx = fma(v[0], px, fma(v[1], py, v[2])) * inv_denom;
        sy = fma(v[3], px, fma(v[4], py, v[5])) * inv_denom;
    }
    if (!isfinite(sx) || !isfinite(sy) || sx < 0.0f || sy < 0.0f || sx >= float(sw) || sy >= float(sh)) {
        dst[i] = store_sample<T>(v[8 + band]);
        return;
    }
    if (k == 0) {
        dst[i] = src[(uint(sy) * sw + uint(sx)) * c + band];
        return;
    }
    sx -= 0.5f;
    sy -= 0.5f;
    float fl_x = floor(sx), fl_y = floor(sy);
    int ix = int(fl_x), iy = int(fl_y);
    float fx = sx - fl_x, fy = sy - fl_y;
    int max_x = int(sw) - 1, max_y = int(sh) - 1;

    if (k == 2) {
        // Precompute the 4 clamped X offsets once instead of 16 times
        uint x_off[4] = {uint(clamp(ix - 1, 0, max_x)) * c + band,
                         uint(clamp(ix, 0, max_x)) * c + band,
                         uint(clamp(ix + 1, 0, max_x)) * c + band,
                         uint(clamp(ix + 2, 0, max_x)) * c + band};
        float4 rows;
#pragma unroll
        for (int ky = -1; ky <= 2; ky++) {
            device const T *row_ptr = src + uint(clamp(iy + ky, 0, max_y)) * sw * c;
            float4 taps = float4(float(row_ptr[x_off[0]]), float(row_ptr[x_off[1]]), float(row_ptr[x_off[2]]), float(row_ptr[x_off[3]]));
            rows[ky + 1] = warp_cubic(taps, fx);
        }
        float result = warp_cubic(rows, fy);
        dst[i] = store_sample<T>(is_same_v<T, uchar> ? floor(result) : result);
        return;
    }

    uint x0 = uint(clamp(ix, 0, max_x)) * c + band;
    uint x1 = uint(clamp(ix + 1, 0, max_x)) * c + band;
    uint y0 = uint(clamp(iy, 0, max_y)) * sw * c;
    uint y1 = uint(clamp(iy + 1, 0, max_y)) * sw * c;

    float wx0 = 1.0f - fx, wy0 = 1.0f - fy;
    float r0 = float(src[y0 + x0]) * wx0 + float(src[y0 + x1]) * fx;
    float r1 = float(src[y1 + x0]) * wx0 + float(src[y1 + x1]) * fx;
    float result = r0 * wy0 + r1 * fy;
    dst[i] = store_sample<T>(is_same_v<T, uchar> ? floor(result) : result);
}

// Crop, transpose, or wrap one channel without changing sample bits.
template <typename T> void geometry_sample(device const T *src, device T *dst, constant uint *p, uint i, uint op, uint c) {
    uint sw = p[0], sh = p[1], dw = p[3], k = p[6];
    uint band, x, y;
    decompose_3d(i, c, dw, band, x, y);
    if (op == OP_CROP) {
        int sx = int(x) + as_type<int>(k), sy = int(y) + as_type<int>(p[7]);
        dst[i] = (sx >= 0 && sy >= 0 && uint(sx) < sw && uint(sy) < sh) ? src[(uint(sy) * sw + uint(sx)) * c + band] : T(0);
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
        uint p7 = p[7];
        uint sx = x >= k ? x - k : x + sw - k;
        uint sy = y >= p7 ? y - p7 : y + sh - p7;
        dst[i] = src[(sy * sw + sx) * c + band];
    }
}