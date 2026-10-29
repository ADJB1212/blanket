//! Numeric layout shared by GPU dispatch validation and Metal shaders.

// One dispatch carries source shape, destination shape, and operation-specific controls.
pub const PARAMETER_COUNT: usize = 8;
pub mod parameter {
    // Source and destination raster dimensions use packed, interleaved channels.
    pub const SOURCE_WIDTH: usize = 0;
    pub const SOURCE_HEIGHT: usize = 1;
    pub const CHANNELS: usize = 2;
    pub const OUTPUT_WIDTH: usize = 3;
    pub const OUTPUT_HEIGHT: usize = 4;
    // The operation selects how the final two control words are interpreted.
    pub const OPERATION: usize = 5;
    pub const KERNEL_OR_X: usize = 6;
    pub const AXIS_OR_Y: usize = 7;
}

pub mod operation {
    // IDs match the operation branches in kernels.metal and ImageGpu._run.
    pub const RESIZE: u32 = 0;
    pub const CONVOLVE: u32 = 1;
    pub const BLEND: u32 = 2;
    pub const ALPHA_COMPOSITE: u32 = 3;
    pub const PREMULTIPLY: u32 = 4;
    pub const UNPREMULTIPLY: u32 = 5;
    pub const COPY: u32 = 6;
    pub const CROP: u32 = 7;
    pub const TRANSPOSE: u32 = 8;
    pub const OFFSET: u32 = 9;
    pub const BRIGHTNESS: u32 = 10;
    pub const INVERT: u32 = 11;
    pub const POINT: u32 = 12;
    pub const BINARY: u32 = 13;
    pub const CONVERT: u32 = 14;
    pub const CHANNEL: u32 = 15;
    pub const REPLACE_CHANNEL: u32 = 16;
    pub const SET_ALPHA: u32 = 17;
    pub const BOX_BLUR: u32 = 18;
    pub const RANK: u32 = 19;
    pub const UNSHARP: u32 = 20;
    pub const LUMINANCE_REDUCE: u32 = 21;
    pub const ENHANCE: u32 = 22;
    pub const TRANSFER: u32 = 23;
    pub const CLAMP_ALPHA: u32 = 24;
    pub const COLOR_LUT: u32 = 25;
    pub const WARP: u32 = 26;
    pub const REMAP: u32 = 27;
    pub const MASK: u32 = 28;
    pub const STATISTICS: u32 = 29;
    pub const QUANTIZE: u32 = 30;
}

pub mod enhancement {
    // Enhancement selectors share the ENHANCE dispatch branch.
    pub const COLOR: usize = 0;
    pub const CONTRAST: usize = 1;
    pub const SHARPNESS: usize = 2;
}

pub mod statistics {
    // Extrema, four raw power sums, and the exact pixel count.
    pub const FIELDS: u32 = 7;
    pub const HISTOGRAM_BINS: u32 = 256;
}

// Perspective numerator rows and denominator coefficients precede warp fill samples.
pub const WARP_COEFFICIENTS: usize = 8;

pub mod reduction {
    // Each worker combines a bounded block; uint8 sums retain all 64 bits.
    pub const BLOCK_SIZE: u32 = 256;
    pub const BYTE_INPUT: usize = 0;
    pub const FLOAT_INPUT: usize = 1;
    pub const BYTE_PARTIALS: usize = 2;
    pub const FLOAT_PARTIALS: usize = 3;
}

pub mod lut {
    // Three dimensions and three fixed-point coordinate scales precede the table.
    pub const HEADER_LEN: usize = 6;
    pub const MIN_SIZE: usize = 2;
    pub const MAX_SIZE: usize = 65;
    // Match CPU table quantization and trilinear interpolation precision.
    pub const SAMPLE_BITS: u32 = 6;
    pub const SAMPLE_SCALE: f32 = ((u8::MAX as u32) << SAMPLE_BITS) as f32;
    pub const POSITION_BITS: u32 = 18;
}

pub mod format {
    // Codes follow ImageGpu's supported modes; float grayscale uses raw F samples.
    pub const L: u32 = 0;
    pub const LA: u32 = 1;
    pub const RGB: u32 = 2;
    pub const RGBA: u32 = 3;
    pub const F: u32 = 4;
    pub const RGB_F: u32 = 5;
    pub const RGBA_F: u32 = 6;

    pub fn layout(code: u32) -> Option<(u32, bool)> {
        Some(match code {
            L => (1, false),
            LA => (2, false),
            RGB => (3, false),
            RGBA => (4, false),
            F => (1, true),
            RGB_F => (3, true),
            RGBA_F => (4, true),
            _ => return None,
        })
    }
}

// Match CPU fractional box filtering's fixed-point weights and Gaussian pass count.
pub const BOX_WEIGHT_BITS: u32 = 24;
pub const BOX_WEIGHT_SCALE: u32 = 1 << BOX_WEIGHT_BITS;
pub const GAUSSIAN_PASSES: u32 = 3;
// Bound per-thread median storage and neighborhood work to a 7x7 window.
pub const MAX_RANK_SIZE: usize = 7;

pub mod resampling {
    // Public Pillow-compatible Image.Resampling identifiers.
    pub const NEAREST: u8 = 0;
    pub const LANCZOS: u8 = 1;
    pub const BILINEAR: u8 = 2;
    pub const BICUBIC: u8 = 3;
    pub const BOX: u8 = 4;
    pub const HAMMING: u8 = 5;
}

pub mod weights {
    // Each padded row stores the source start, tap count, then coefficients.
    pub const START: usize = 0;
    pub const COUNT: usize = 1;
    pub const HEADER_LEN: usize = 2;
    // CPU resampling uses 22 fractional bits in its signed coefficients.
    pub const SCALE: f64 = (1_u32 << 22) as f64;
}

pub mod transpose {
    // Image.Transpose methods that swap the raster dimensions.
    pub const ROTATE_90: usize = 2;
    pub const ROTATE_270: usize = 4;
    pub const TRANSPOSE: usize = 5;
    pub const TRANSVERSE: usize = 6;
}

// Final ImageChops operation ID supported by the binary shader branch.
pub const BINARY_SUBTRACT: usize = 6;
// Binary controls store scale followed by offset.
pub const BINARY_CONTROL_LEN: usize = 2;
// Convolution controls store a reserved scalar and offset before the coefficients.
pub const CONVOLUTION_HEADER_LEN: usize = 2;
// The shaders support square 3x3 and 5x5 convolution kernels.
pub const SMALL_KERNEL_SIZE: usize = 3;
pub const LARGE_KERNEL_SIZE: usize = 5;
// Resize axes select horizontal or vertical coefficient rows.
pub const HORIZONTAL: u32 = 0;
pub const VERTICAL: u32 = 1;
// Alpha occupies the final channel in LA and RGBA storage.
pub const GRAY_ALPHA_CHANNELS: u32 = 2;
pub const COLOR_ALPHA_CHANNELS: u32 = 4;
// Uint8 lookup tables contain one entry for every possible sample.
pub const LUT_ENTRIES: usize = u8::MAX as usize + 1;
