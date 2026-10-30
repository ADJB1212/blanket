use super::*;

fn scalar_pipeline(context: &Context, op: u32, channels: u32, floating: bool) -> Pipeline {
    let constants = MTLFunctionConstantValues::new();
    unsafe {
        constants.setConstantValue_type_atIndex(std::ptr::NonNull::from(&op).cast(), MTLDataType::UInt, OPERATION_CONSTANT);
        constants.setConstantValue_type_atIndex(std::ptr::NonNull::from(&channels).cast(), MTLDataType::UInt, CHANNELS_CONSTANT);
    }
    let name = NSString::from_str(if floating { "image_f32" } else { "image_u8" });
    let function = context.library.newFunctionWithName_constantValues_error(&name, &constants).unwrap();
    context.device.newComputePipelineStateWithFunction_error(&function).unwrap()
}

fn guarded_dispatch(
    context: &Context, pipeline: &ProtocolObject<dyn MTLComputePipelineState>, request: &Dispatch, input: &[u8], other: &[u8], threads: usize,
    group_threads: usize,
) -> Vec<u8> {
    let upload = |data: &[u8]| {
        let mut bytes = vec![0xa5; data.len() + 32];
        bytes[16..16 + data.len()].copy_from_slice(data);
        context.upload(Descriptor { width: bytes.len() as u32, height: 1, channels: 1, float: false }, &bytes).unwrap()
    };
    let source = upload(input);
    let other = upload(other);
    let len = request.output.byte_len().unwrap();
    let output = upload(&vec![0xa5; len]);
    let command = context.queue.command_buffer().unwrap();
    let encoder = command.compute_encoder().unwrap();
    let values = if request.values.is_empty() { &[0.0][..] } else { &request.values };
    encoder.set_pipeline(pipeline);
    unsafe {
        encoder.set_buffer(Some(&source.buffer), 16, SOURCE_SLOT);
        encoder.set_buffer(Some(&output.buffer), 16, OUTPUT_SLOT);
        encoder.set_buffer(Some(&other.buffer), 16, OTHER_SLOT);
        encoder.set_bytes(std::ptr::NonNull::from(&request.parameters).cast(), size_of_val(&request.parameters), PARAMETERS_SLOT);
        encoder.set_bytes(std::ptr::NonNull::from(values).cast(), size_of_val(values), VALUES_SLOT);
    }
    encoder.dispatch_threads(
        MTLSize { width: threads, height: 1, depth: 1 },
        MTLSize { width: group_threads.min(pipeline.maxTotalThreadsPerThreadgroup()), height: 1, depth: 1 },
    );
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed, "{:?}", command.error());
    let bytes = context.download(&output).unwrap();
    assert!(bytes[..16].iter().chain(&bytes[16 + len..]).all(|&value| value == 0xa5));
    assert_eq!(&context.download(&source).unwrap()[16..16 + input.len()], input);
    bytes[16..16 + len].to_vec()
}

fn assert_float_samples(actual: &[u8], expected: &[u8], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &b)) in actual.as_chunks::<4>().0.iter().zip(expected.as_chunks::<4>().0).enumerate() {
        let (a, b) = (f32::from_ne_bytes(a), f32::from_ne_bytes(b));
        assert!(a == b || (a.is_nan() && b.is_nan()) || (a - b).abs() <= tolerance * b.abs().max(1.0), "sample {index}: {a} != {b}");
    }
}

fn samples(count: usize, floating: bool, seed: usize) -> Vec<u8> {
    if floating {
        let values = [0.0, -0.0, -1.25, 0.25, 1.0, f32::INFINITY, f32::NEG_INFINITY, f32::from_bits(0x7fc12345), 1e30, 1e-30, 2.0];
        (0..count).flat_map(|i| values[(i * 7 + seed) % values.len()].to_ne_bytes()).collect()
    } else {
        (0..count).map(|i| (i * 97 + i / 7 + seed) as u8).collect()
    }
}

#[test]
fn pixel_geometry_and_alpha_match_scalar_shaders_and_preserve_guards() {
    let Some(context) = test_context() else { return };
    for floating in [false, true] {
        for channels in [1, 2, 3, 4] {
            let mut cases = vec![(operation::CROP, (-2_i32) as u32, 1), (operation::OFFSET, 2, 1)];
            cases.extend((0..7).map(|method| (operation::TRANSPOSE, method, 0)));
            if matches!(channels, 2 | 4) {
                cases.extend([operation::ALPHA_COMPOSITE, operation::PREMULTIPLY, operation::UNPREMULTIPLY].map(|op| (op, 0, 0)));
            }
            for (op, kind, axis) in cases {
                let scalar = scalar_pipeline(&context, op, channels, floating);
                let packed = context.operation_pipeline(op, floating, channels).unwrap();
                for (width, height) in [(1, 1), (1, 33), (33, 1), (15, 17), (16, 16), (17, 15), (17, 17), (31, 33), (32, 32), (33, 31), (65, 63)] {
                    let (dw, dh) = if op == operation::TRANSPOSE && matches!(kind, 2 | 4 | 5 | 6) { (height, width) } else { (width, height) };
                    let descriptor = Descriptor { width: dw, height: dh, channels, float: floating };
                    let count = width as usize * height as usize * channels as usize;
                    let input = samples(count, floating, 0);
                    let other = samples(count, floating, 3);
                    let offset = if op == operation::OFFSET { kind % width } else { kind };
                    let axis = if op == operation::OFFSET { axis % height } else { axis };
                    let request = Dispatch { output: descriptor, parameters: [width, height, channels, dw, dh, op, offset, axis], values: vec![] };
                    let expected = guarded_dispatch(&context, &scalar, &request, &input, &other, count, THREADGROUP_THREADS);
                    let threads = dispatch_samples(&request.parameters, floating, count, packed.threadExecutionWidth());
                    let actual = guarded_dispatch(&context, packed, &request, &input, &other, threads, THREADGROUP_THREADS);
                    if op == operation::TRANSPOSE {
                        let fallback = guarded_dispatch(&context, packed, &request, &input, &other, threads, 64);
                        assert_eq!(fallback, expected);
                    }
                    if floating && matches!(op, operation::ALPHA_COMPOSITE | operation::PREMULTIPLY | operation::UNPREMULTIPLY) {
                        assert_float_samples(&actual, &expected, 1e-6);
                    } else {
                        assert_eq!(actual, expected, "floating={floating}, channels={channels}, width={width}, op={op}, kind={kind}");
                    }
                }
            }
        }
    }
}

#[test]
fn packed_filters_match_scalar_shaders_and_preserve_guards() {
    let Some(context) = test_context() else { return };
    for floating in [false, true] {
        for channels in [1, 2, 3, 4] {
            for op in [operation::RESIZE, operation::CONVOLVE] {
                let scalar = scalar_pipeline(&context, op, channels, floating);
                let packed = context.operation_pipeline(op, floating, channels).unwrap();
                for (width, height) in [(1_u32, 1_u32), (1, 33), (33, 1), (7, 5), (31, 17), (32, 16), (33, 15)] {
                    for nonfinite in [false, true] {
                        let count = width as usize * height as usize * channels as usize;
                        let input = if floating && !nonfinite {
                            (0..count).flat_map(|i| (((i * 97 % 251) as f32 - 127.0) / 37.0).to_ne_bytes()).collect()
                        } else {
                            samples(count, floating, 0)
                        };
                        for axis in [0, 1] {
                            let (dw, dh, kernel, values) = if op == operation::RESIZE {
                                let (dw, dh) = if axis == 0 { (width * 2 + 1, height) } else { (width, height.div_ceil(2)) };
                                let (kernel, values) = blanket_gpu::resize_weights(
                                    if axis == 0 { width } else { height },
                                    if axis == 0 { dw } else { dh },
                                    blanket_gpu::protocol::resampling::LANCZOS,
                                )
                                .unwrap();
                                (dw, dh, kernel, values)
                            } else {
                                let size = if axis == 0 { 3 } else { 5 };
                                let values = [vec![0.0, -0.375], (0..size * size).map(|i| (i as f32 - 7.0) / 13.0).collect()].concat();
                                (width, height, size, values)
                            };
                            let output = Descriptor { width: dw, height: dh, channels, float: floating };
                            let request = Dispatch { output, parameters: [width, height, channels, dw, dh, op, kernel, axis], values };
                            let count = dw as usize * dh as usize * channels as usize;
                            let expected = guarded_dispatch(&context, &scalar, &request, &input, &input, count, THREADGROUP_THREADS);
                            let threads = dispatch_samples(&request.parameters, floating, count, packed.threadExecutionWidth());
                            let actual = guarded_dispatch(&context, packed, &request, &input, &input, threads, THREADGROUP_THREADS);
                            if floating {
                                assert_float_samples(&actual, &expected, 1e-6);
                            } else {
                                assert_eq!(actual, expected, "channels={channels}, size={width}x{height}, op={op}, axis={axis}");
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn packed_warps_match_scalar_shaders_and_preserve_guards() {
    let Some(context) = test_context() else { return };
    for floating in [false, true] {
        for channels in [1, 2, 3, 4] {
            for op in [operation::WARP, operation::REMAP] {
                let scalar = scalar_pipeline(&context, op, channels, floating);
                let packed = context.operation_pipeline(op, floating, channels).unwrap();
                for (width, height) in [(1, 1), (1, 33), (33, 1), (7, 5), (31, 17), (32, 16), (33, 15)] {
                    let dw = width + 2;
                    let dh = height + 1;
                    let count = width as usize * height as usize * channels as usize;
                    let coordinates: Vec<u8> = (0..dw * dh)
                        .flat_map(|i| {
                            let x = i % dw;
                            let y = i / dw;
                            let sx = match x % 7 {
                                0 => -0.25,
                                1 => 0.0,
                                2 => 0.5,
                                3 => width as f32 - 0.25,
                                4 => width as f32,
                                5 => f32::NAN,
                                _ => f32::INFINITY,
                            };
                            [sx, y as f32 - 0.125, 0.0].into_iter().flat_map(f32::to_ne_bytes)
                        })
                        .collect();
                    for nonfinite in [false, true] {
                        let input = if floating && !nonfinite {
                            (0..count).flat_map(|i| (((i * 97 % 251) as f32 - 127.0) / 37.0).to_ne_bytes()).collect()
                        } else {
                            samples(count, floating, 0)
                        };
                        for kernel in [0, 1, 2] {
                            for perspective in [false, true] {
                                let values =
                                    vec![1.0, 0.05, -0.75, -0.05, 1.0, 0.125, if perspective { -0.25 } else { 0.0 }, 0.0, -1.25, 17.5, 259.0, 0.25];
                                let output = Descriptor { width: dw, height: dh, channels, float: floating };
                                let request = Dispatch { output, parameters: [width, height, channels, dw, dh, op, kernel, 0], values };
                                let count = dw as usize * dh as usize * channels as usize;
                                let expected = guarded_dispatch(&context, &scalar, &request, &input, &coordinates, count, THREADGROUP_THREADS);
                                let threads = dispatch_samples(&request.parameters, floating, count, packed.threadExecutionWidth());
                                let actual = guarded_dispatch(&context, packed, &request, &input, &coordinates, threads, THREADGROUP_THREADS);
                                if floating {
                                    assert_float_samples(&actual, &expected, 1e-5);
                                } else {
                                    for (index, (a, b)) in actual.iter().zip(&expected).enumerate() {
                                        assert_eq!(
                                            a, b,
                                            "sample={index}, channels={channels}, size={width}x{height}, op={op}, kernel={kernel}, perspective={perspective}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn packed_float_convolution_plans_match_direct_dispatch() {
    let Some(context) = test_context() else { return };
    for channels in [1, 2, 3, 4] {
        let descriptor = Descriptor { width: 37, height: 29, channels, float: true };
        let input = samples(37 * 29 * channels as usize, true, 0);
        let source = context.upload(descriptor, &input).unwrap();
        for kernel in [3, 5] {
            let values = [vec![0.0, -0.375], vec![1.0 / (kernel * kernel) as f32; (kernel * kernel) as usize]].concat();
            let request =
                Dispatch { output: descriptor, parameters: [37, 29, channels, 37, 29, operation::CONVOLVE, kernel, 0], values: values.clone() };
            let direct = unsafe { context.dispatch(&source, &source, &request) }.unwrap();
            let expected = context.download(&direct).unwrap();
            for limit in [4096, 1_000_000] {
                let step = blanket_gpu::plan::Step { operation: operation::CONVOLVE, kernel, axis: 0, values: values.clone() };
                let geometry = blanket_gpu::plan::Plan::new(descriptor, vec![step], limit).unwrap();
                assert_eq!(geometry.tile() == [37, 29], limit == 1_000_000);
                let plan = context.prepare_plan(geometry).unwrap();
                let outputs = context.execute_plan(&plan, &[&source]).unwrap();
                assert_float_samples(&context.download(&outputs[0]).unwrap(), &expected, 1e-6);
            }
            assert_eq!(context.download(&source).unwrap(), input);
        }
    }
}

#[test]
fn packed_tile_copies_preserve_pixels_outside_odd_rectangles() {
    let Some(context) = test_context() else { return };
    for floating in [false, true] {
        for channels in [1, 2, 3, 4] {
            let src = Descriptor { width: 37, height: 7, channels, float: floating };
            let dst = Descriptor { width: 41, height: 9, ..src };
            let input = samples(37 * 7 * channels as usize, floating, 0);
            let source = context.upload(src, &input).unwrap();
            let pixel_bytes = channels as usize * if floating { 4 } else { 1 };
            for width in [1, 3, 31, 32, 33] {
                let mut expected = vec![0xa5; dst.byte_len().unwrap()];
                let output = context.upload(dst, &expected).unwrap();
                let command = context.queue.command_buffer().unwrap();
                let encoder = command.compute_encoder().unwrap();
                context.encode_tile(&encoder, &source.buffer, &output.buffer, [37, 7, channels, 41, 9, 3, 2, 5, 4, width, 3], floating);
                encoder.end_encoding();
                command.commit();
                command.wait_until_completed();
                assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                for y in 0..3 {
                    let from = ((y + 2) * 37 + 3) * pixel_bytes;
                    let to = ((y + 4) * 41 + 5) * pixel_bytes;
                    expected[to..to + width as usize * pixel_bytes].copy_from_slice(&input[from..from + width as usize * pixel_bytes]);
                }
                assert_eq!(context.download(&output).unwrap(), expected);
            }
        }
    }
}
