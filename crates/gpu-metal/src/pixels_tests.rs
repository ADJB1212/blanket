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

fn assert_float_samples(actual: &[u8], expected: &[u8]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &b)) in actual.as_chunks::<4>().0.iter().zip(expected.as_chunks::<4>().0).enumerate() {
        let (a, b) = (f32::from_ne_bytes(a), f32::from_ne_bytes(b));
        assert!(a == b || (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-6 * b.abs().max(1.0), "sample {index}: {a} != {b}");
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
                        assert_float_samples(&actual, &expected);
                    } else {
                        assert_eq!(actual, expected, "floating={floating}, channels={channels}, width={width}, op={op}, kind={kind}");
                    }
                }
            }
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
