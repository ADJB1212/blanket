// Shared Metal types, operation IDs, precision constants, and sample access.

#include <metal_stdlib>
using namespace metal;

#ifdef BLANKET_METAL_SPECIALIZE
// Specialization removes unrelated operations and their register requirements.
constant uint SPECIALIZED_OPERATION [[function_constant(0)]];
constant uint SPECIALIZED_CHANNELS [[function_constant(1)]];
#endif

// Match CPU luminance conversion and fractional box-filter precision.
constant uint LUMA_RED = 19595, LUMA_GREEN = 38470, LUMA_BLUE = 7471, LUMA_BITS = 16;
constant uint BOX_BITS = 24, BOX_SCALE = 1u << BOX_BITS, BOX_ROUND = BOX_SCALE / 2;
// Rank validation limits each neighborhood to 7x7 samples.
constant uint MAX_RANK_SAMPLES = 49;
// Keep reduction blocks bounded and byte luminance sums exact across all passes.
constant uint REDUCTION_BLOCK_SIZE = 256;
// CPU 3D LUTs use signed samples with 6 fractional bits and 15-bit interpolation.
constant uint LUT_SAMPLE_BITS = 6, LUT_POSITION_BITS = 18, LUT_BLEND_BITS = 15;
constant int LUT_BLEND_SCALE = 1 << LUT_BLEND_BITS;
constant uint LUT_HEADER_LEN = 6;
// Statistics store extrema, four raw power sums, and the exact pixel count.
constant uint STATISTICS_FIELDS = 7, MOMENT_ORDERS = 4, HISTOGRAM_BINS = 256;
// Resize coefficients use the CPU kernel's 22-bit fixed-point scale.
constant uint RESIZE_BITS = 22, RESIZE_SCALE = 1u << RESIZE_BITS, RESIZE_ROUND = RESIZE_SCALE / 2;
constant uint BLUR_OUTPUTS = 16, SLIDING_BLUR_RADIUS = 32;

// Operation values match blanket_gpu::protocol::operation.
enum Operation : uint {
    OP_RESIZE = 0,
    OP_CONVOLVE = 1,
    OP_BLEND = 2,
    OP_ALPHA_COMPOSITE = 3,
    OP_PREMULTIPLY = 4,
    OP_UNPREMULTIPLY = 5,
    OP_COPY = 6,
    OP_CROP = 7,
    OP_TRANSPOSE = 8,
    OP_OFFSET = 9,
    OP_BRIGHTNESS = 10,
    OP_INVERT = 11,
    OP_POINT = 12,
    OP_BINARY = 13,
    OP_CONVERT = 14,
    OP_CHANNEL = 15,
    OP_REPLACE_CHANNEL = 16,
    OP_SET_ALPHA = 17,
    OP_BOX_BLUR = 18,
    OP_RANK = 19,
    OP_UNSHARP = 20,
    OP_LUMINANCE_REDUCE = 21,
    OP_ENHANCE = 22,
    OP_TRANSFER = 23,
    OP_CLAMP_ALPHA = 24,
    OP_COLOR_LUT = 25,
    OP_WARP = 26,
    OP_REMAP = 27,
    OP_MASK = 28,
    OP_STATISTICS = 29,
    OP_QUANTIZE = 30,
};

// Decompose 1D index into (x, y) using 1 division instead of 1 division + 1 modulo.
inline void decompose_2d(uint i, uint width, thread uint &x, thread uint &y) {
    y = i / width;
    x = i - y * width;
}

// Decompose 1D sample index into (band, pixel_idx) using bitops when c is a power of 2.
inline void decompose_channel(uint i, uint c, thread uint &pixel_idx, thread uint &band) {
    if ((c & (c - 1)) == 0) {
        uint shift = (c == 4) ? 2u : ((c == 2) ? 1u : 0u);
        pixel_idx = i >> shift;
        band = i & (c - 1);
    } else {
        pixel_idx = i / c;
        band = i - pixel_idx * c;
    }
}

// Decompose 1D sample index into (band, x, y).
inline void decompose_3d(uint i, uint c, uint width, thread uint &band, thread uint &x, thread uint &y) {
    uint pixel_idx;
    decompose_channel(i, c, pixel_idx, band);
    decompose_2d(pixel_idx, width, x, y);
}

// Read a byte or float sample in the arithmetic working type.
template <typename T> inline float read_sample(device const T *p, uint i) {
    return float(p[i]);
}

// Store a float sample without changing its working range.
template <typename T> inline T store_sample(float v) {
    return T(v);
}

// Round and saturate a computed sample to an unsigned byte.
template <> inline uchar store_sample<uchar>(float v) {
    return uchar(saturate(floor(v + 0.5f) * (1.0f / 255.0f)) * 255.0f);
}