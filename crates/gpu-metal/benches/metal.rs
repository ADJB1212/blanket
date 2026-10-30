#[cfg(target_os = "macos")]
mod benchmarks {
    use std::hint::black_box;

    use blanket_gpu::protocol::{operation, reduction, resampling, statistics};
    use blanket_gpu::{Backend, Descriptor, Dispatch, blur_parameters, resize_weights};
    use blanket_gpu_metal::{Context, Image};
    use criterion::{Criterion, Throughput};

    // Small and medium rasters expose dispatch costs; the full matrix adds HD.
    const SIZES: [(u32, u32); 3] = [(64, 64), (512, 512), (1920, 1080)];
    // RGBA covers straight-alpha color in both shipped sample formats.
    const CHANNELS: u32 = 4;

    fn fixture(descriptor: Descriptor) -> Vec<u8> {
        let samples = descriptor.width as usize * descriptor.height as usize * descriptor.channels as usize;
        if descriptor.float {
            (0..samples).flat_map(|i| ((i % 251) as f32 / 250.0).to_ne_bytes()).collect()
        } else {
            (0..samples).map(|i| (i % 251) as u8).collect()
        }
    }

    fn request(source: Descriptor, output: Descriptor, op: u32, k: u32, axis: u32, values: Vec<f32>) -> Dispatch {
        Dispatch { output, parameters: [source.width, source.height, source.channels, output.width, output.height, op, k, axis], values }
    }

    fn dispatch(context: &Context, source: &Image, other: &Image, request: &Dispatch) -> Image {
        // Requests below use matching immutable fixtures and bounded kernel controls.
        unsafe { context.dispatch(black_box(source), black_box(other), black_box(request)) }.expect("Metal benchmark dispatch failed")
    }

    fn complete(context: &Context, image: Image) -> Image {
        context.synchronize(&image).expect("Metal benchmark execution failed");
        black_box(image)
    }

    pub(super) fn metal(c: &mut Criterion) {
        if !Context::is_available() {
            eprintln!("skipping Metal benchmarks: Metal 4 requires macOS 26 or later and a supported GPU");
            return;
        }
        let context = Context::new().expect("Metal benchmarks require an available Metal device");
        {
            let mut initialization = c.benchmark_group("gpu/metal/context");
            initialization.bench_function("create_and_compile", |b| {
                b.iter(|| black_box(Context::new().expect("Metal context creation failed")));
            });
            initialization.finish();
        }

        let sizes = if blanket_bench_support::full_matrix() { &SIZES[..] } else { &SIZES[..2] };
        for &(width, height) in sizes {
            {
                let mut scans = c.benchmark_group(format!("gpu/metal/scan/{width}x{height}"));
                scans.throughput(Throughput::Elements(u64::from(width) * u64::from(height)));
                for channels in [1, 2, 3, 4] {
                    let descriptor = Descriptor { width, height, channels, float: false };
                    let source = context.upload(descriptor, &fixture(descriptor)).unwrap();
                    let (radius, controls) = blur_parameters(32.5, false).unwrap();
                    for (axis, name) in [(0, "horizontal"), (1, "vertical")] {
                        let request = request(descriptor, descriptor, operation::BOX_BLUR, radius, axis, controls.to_vec());
                        complete(&context, dispatch(&context, &source, &source, &request));
                        scans.bench_function(format!("c{channels}/{name}"), |b| {
                            b.iter(|| complete(&context, dispatch(&context, &source, &source, &request)))
                        });
                    }
                }
                scans.finish();
            }
            {
                let mut histograms = c.benchmark_group(format!("gpu/metal/histogram/{width}x{height}"));
                histograms.throughput(Throughput::Elements(u64::from(width) * u64::from(height)));
                let descriptor = Descriptor { width, height, channels: CHANNELS, float: false };
                for (name, bytes) in [
                    ("constant", vec![71; descriptor.byte_len().unwrap()]),
                    ("striped", (0..descriptor.byte_len().unwrap()).map(|i| ((i / 64) % 4 * 85) as u8).collect()),
                    ("patterned", fixture(descriptor)),
                ] {
                    let source = context.upload(descriptor, &bytes).unwrap();
                    let request = request(
                        descriptor,
                        Descriptor { width: statistics::HISTOGRAM_BINS, height: CHANNELS, channels: 1, float: true },
                        operation::STATISTICS,
                        1,
                        0,
                        vec![],
                    );
                    complete(&context, dispatch(&context, &source, &source, &request));
                    histograms.bench_function(name, |b| b.iter(|| complete(&context, dispatch(&context, &source, &source, &request))));
                }
                histograms.finish();
            }
            for floating in [false, true] {
                let descriptor = Descriptor { width, height, channels: CHANNELS, float: floating };
                let bytes = fixture(descriptor);
                let source = context.upload(descriptor, &bytes).unwrap();
                let other = context.upload(descriptor, &bytes).unwrap();
                let format = if floating { "rgba_f32" } else { "rgba_u8" };
                {
                    let mut transfers = c.benchmark_group(format!("gpu/metal/transfers/{format}/{width}x{height}"));
                    transfers.throughput(Throughput::Bytes(bytes.len() as u64));
                    transfers.bench_function("upload", |b| b.iter(|| context.upload(descriptor, black_box(&bytes)).unwrap()));
                    transfers.bench_function("download", |b| b.iter(|| context.download(black_box(&source)).unwrap()));
                    transfers.bench_function("roundtrip", |b| {
                        b.iter(|| {
                            let image = context.upload(descriptor, black_box(&bytes)).unwrap();
                            context.download(&image).unwrap()
                        })
                    });
                    transfers.finish();
                }

                let brightness = request(descriptor, descriptor, operation::BRIGHTNESS, 0, 0, vec![1.2]);
                let invert = request(descriptor, descriptor, operation::INVERT, 0, 0, vec![]);
                let (radius, controls) = blur_parameters(2.5, false).unwrap();
                let small = Descriptor { width: width / 2, ..descriptor };
                let resized = Descriptor { height: height / 2, ..small };
                let (stride, weights) = resize_weights(width, small.width, resampling::LANCZOS).unwrap();
                let horizontal = request(descriptor, small, operation::RESIZE, stride, 0, weights);
                let (stride, weights) = resize_weights(height, resized.height, resampling::LANCZOS).unwrap();
                let vertical = request(small, resized, operation::RESIZE, stride, 1, weights);
                let resized_brightness = request(resized, resized, operation::BRIGHTNESS, 0, 0, vec![1.2]);
                let mut cases = vec![
                    ("copy", request(descriptor, descriptor, operation::COPY, 0, 0, vec![])),
                    ("brightness", request(descriptor, descriptor, operation::BRIGHTNESS, 0, 0, vec![1.2])),
                    ("blend", request(descriptor, descriptor, operation::BLEND, 0, 0, vec![0.4])),
                    ("multiply", request(descriptor, descriptor, operation::BINARY, 1, 0, vec![])),
                    ("offset", request(descriptor, descriptor, operation::OFFSET, 7, 5, vec![])),
                    ("crop", request(descriptor, resized, operation::CROP, width / 4, height / 4, vec![])),
                    ("rotate_90", request(descriptor, Descriptor { width: height, height: width, ..descriptor }, operation::TRANSPOSE, 2, 0, vec![])),
                    ("alpha_composite", request(descriptor, descriptor, operation::ALPHA_COMPOSITE, 0, 0, vec![])),
                    ("premultiply", request(descriptor, descriptor, operation::PREMULTIPLY, 0, 0, vec![])),
                    ("unpremultiply", request(descriptor, descriptor, operation::UNPREMULTIPLY, 0, 0, vec![])),
                    ("convolve_3x3", request(descriptor, descriptor, operation::CONVOLVE, 3, 0, [vec![0.0, 0.0], vec![1.0 / 9.0; 9]].concat())),
                    ("box_blur_horizontal", request(descriptor, descriptor, operation::BOX_BLUR, radius, 0, controls.to_vec())),
                    ("median_3x3", request(descriptor, descriptor, operation::RANK, 3, 4, vec![])),
                    (
                        "affine_bilinear",
                        request(descriptor, descriptor, operation::WARP, 1, 0, vec![1.0, 0.05, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                    ),
                    (
                        "statistics",
                        request(
                            descriptor,
                            Descriptor { width: statistics::FIELDS, height: CHANNELS, channels: 1, float: true },
                            operation::STATISTICS,
                            0,
                            0,
                            vec![],
                        ),
                    ),
                    (
                        "luminance_first_pass",
                        request(
                            descriptor,
                            Descriptor { width: (width * height).div_ceil(reduction::BLOCK_SIZE), height: 1, channels: CHANNELS, float: true },
                            operation::LUMINANCE_REDUCE,
                            u32::from(floating),
                            0,
                            vec![],
                        ),
                    ),
                ];
                for radius in [8.5, 32.5, 128.5] {
                    let (kernel, controls) = blur_parameters(radius, false).unwrap();
                    cases.push((
                        match kernel {
                            8 => "box_blur_r8",
                            32 => "box_blur_r32",
                            _ => "box_blur_r128",
                        },
                        request(descriptor, descriptor, operation::BOX_BLUR, kernel, 0, controls.to_vec()),
                    ));
                    cases.push((
                        match kernel {
                            8 => "box_blur_vertical_r8",
                            32 => "box_blur_vertical_r32",
                            _ => "box_blur_vertical_r128",
                        },
                        request(descriptor, descriptor, operation::BOX_BLUR, kernel, 1, controls.to_vec()),
                    ));
                }
                if floating {
                    cases.push(("quantize_u16", request(descriptor, descriptor, operation::QUANTIZE, 0, 0, vec![f32::from(u16::MAX)])));
                } else {
                    cases.push((
                        "histogram",
                        request(
                            descriptor,
                            Descriptor { width: statistics::HISTOGRAM_BINS, height: CHANNELS, channels: 1, float: true },
                            operation::STATISTICS,
                            1,
                            0,
                            vec![],
                        ),
                    ));
                }
                {
                    let mut resident = c.benchmark_group(format!("gpu/metal/resident/{format}/{width}x{height}"));
                    resident.throughput(Throughput::Elements(u64::from(width) * u64::from(height)));
                    for (name, request) in cases {
                        // Warm execution also surfaces asynchronous shader failures before timing.
                        complete(&context, dispatch(&context, &source, &other, &request));
                        resident.bench_function(name, |b| b.iter(|| complete(&context, dispatch(&context, &source, &other, &request))));
                    }
                    let mask_descriptor = Descriptor { channels: 1, ..descriptor };
                    let mask = context.upload(mask_descriptor, &fixture(mask_descriptor)).unwrap();
                    resident.bench_function("masked_paste", |b| {
                        b.iter(|| complete(&context, context.paste(&source, &other, Some(&mask), [3, 5]).unwrap()))
                    });
                    resident.bench_function("resize_lanczos_2pass", |b| {
                        b.iter(|| {
                            let first = dispatch(&context, &source, &source, &horizontal);
                            complete(&context, dispatch(&context, &first, &first, &vertical))
                        })
                    });
                    resident.bench_function("brightness_invert_chain", |b| {
                        b.iter(|| {
                            let first = dispatch(&context, &source, &source, &brightness);
                            complete(&context, dispatch(&context, &first, &first, &invert))
                        })
                    });
                    resident.bench_function("queued_brightness_chain", |b| {
                        b.iter(|| {
                            let mut image = dispatch(&context, &source, &source, &brightness);
                            for _ in 0..15 {
                                image = dispatch(&context, &image, &image, &brightness);
                            }
                            complete(&context, image)
                        })
                    });
                    resident.finish();
                }

                let mut pipeline = c.benchmark_group(format!("gpu/metal/end_to_end/{format}/{width}x{height}"));
                pipeline.throughput(Throughput::Elements(u64::from(width) * u64::from(height)));
                pipeline.bench_function("upload_resize_brightness_download", |b| {
                    b.iter(|| {
                        let uploaded = context.upload(descriptor, black_box(&bytes)).unwrap();
                        let first = dispatch(&context, &uploaded, &uploaded, &horizontal);
                        let second = dispatch(&context, &first, &first, &vertical);
                        let result = dispatch(&context, &second, &second, &resized_brightness);
                        context.download(&result).unwrap()
                    })
                });
                pipeline.finish();
            }
        }
    }
}

#[cfg(target_os = "macos")]
criterion::criterion_group! { name = benches; config = blanket_bench_support::criterion(); targets = benchmarks::metal }
#[cfg(target_os = "macos")]
criterion::criterion_main!(benches);

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("Native Metal benchmarks require macOS and a Metal device");
}
