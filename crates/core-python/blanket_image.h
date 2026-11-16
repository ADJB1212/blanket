#ifndef BLANKET_IMAGE_H
#define BLANKET_IMAGE_H

#include <stddef.h>
#include <stdint.h>

/* PyCapsule_GetPointer(capsule, "blanket.image.v1") returns this descriptor.
 * All pointers are read-only and valid while the caller retains the capsule.
 * Never free pointers or mutate samples. Copy data[0:len] for independent use.
 * Samples use Blanket's native storage: little-endian except I;16B (and
 * I;16N on big-endian hosts), unpacked 0/255 for mode 1, and biased LAB chroma.
 * P images have mode L plus an RGB/RGBA palette. PA retains two channels.
 * No Rust layout or allocator assumptions are required by consumers.
 */
typedef struct BlanketImageV1 {
    uint32_t abi_version;
    uint32_t width;
    uint32_t height;
    uint32_t bit_depth;
    uint32_t channels;
    const uint8_t *data;
    size_t len;
    size_t row_stride;
    const char *mode;
    const uint8_t *palette;
    size_t palette_len;
    uint32_t palette_channels;
} BlanketImageV1;

#endif
