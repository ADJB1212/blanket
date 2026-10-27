use std::fs;
use std::hint::black_box;

use blanket_bench_support::{fixture, sizes};
use blanket_codecs::codecs::{ImageFormat, SaveOptions, encode};
use blanket_codecs::open_bytes;
use blanket_core::PixelMode;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use pyo3::Python;
use pyo3::types::{PyBytes, PyBytesMethods};

fn file_io(c: &mut Criterion) {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("input");
    let output = directory.path().join("output");
    Python::attach(|py| {
        for (width, height) in sizes().iter().copied() {
            for &(content, noise) in blanket_bench_support::contents() {
                let image = fixture(width, height, PixelMode::Rgb, 8, 12345, noise);
                for (name, format) in [("png", ImageFormat::Png), ("jpeg", ImageFormat::Jpeg), ("jxl", ImageFormat::Jxl), ("webp", ImageFormat::Webp)]
                {
                    let options = SaveOptions { quality: 85, effort: 3, ..SaveOptions::default() };
                    let encoded = encode(&image, format, options).unwrap();
                    fs::write(&input, &encoded).unwrap();
                    let mut group = c.benchmark_group(format!("io/{name}/{content}/{width}x{height}"));
                    group.throughput(Throughput::Bytes(encoded.len() as u64));
                    group.bench_function("read-bytes", |b| {
                        b.iter(|| fs::read(black_box(&input)).unwrap());
                    });
                    group.bench_function("write-bytes", |b| {
                        b.iter(|| fs::write(black_box(&output), black_box(&encoded)).unwrap());
                    });
                    group.throughput(Throughput::Elements(u64::from(width) * u64::from(height)));
                    group.bench_function("open-memory", |b| {
                        b.iter(|| open_bytes(py, black_box(&encoded), None).unwrap());
                    });
                    group.bench_function("read-and-open", |b| {
                        b.iter(|| {
                            let bytes = fs::read(black_box(&input)).unwrap();
                            open_bytes(py, &bytes, None).unwrap()
                        });
                    });
                    group.bench_function("encode-and-write", |b| {
                        b.iter(|| {
                            let bytes = encode(black_box(&image), format, black_box(options)).unwrap();
                            fs::write(black_box(&output), bytes).unwrap();
                        });
                    });
                    group.finish();
                }
            }
        }
    });
}

fn metadata_io(c: &mut Criterion) {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("metadata");
    let exif = b"Exif\0\0II*\0\x08\0\0\0\0\0\0\0\0\0";
    let xmp = b"<x:xmpmeta xmlns:x=\"adobe:ns:meta/\">benchmark</x:xmpmeta>";
    Python::attach(|py| {
        for &(width, height) in sizes() {
            let image = fixture(width, height, PixelMode::Rgb, 8, 12345, false);
            for (name, format) in [("png-level-0", ImageFormat::Png), ("jpeg", ImageFormat::Jpeg), ("jxl", ImageFormat::Jxl)] {
                let options = SaveOptions { quality: 85, effort: 3, compress_level: 0, ..SaveOptions::default() };
                let mut group = c.benchmark_group(format!("io-metadata/{name}/{width}x{height}"));
                group.throughput(Throughput::Elements(u64::from(width) * u64::from(height)));
                group.bench_function("native-write", |b| {
                    b.iter(|| {
                        py.detach(|| {
                            let bytes = encode(black_box(&image), format, options).unwrap();
                            let bytes = blanket_codecs::metadata::encode_metadata(bytes, format.as_str(), exif, xmp, &[], &[]).unwrap();
                            fs::write(black_box(&output), bytes).unwrap();
                        });
                    })
                });
                group.bench_function("python-bytes-reference", |b| {
                    b.iter(|| {
                        let encoded = {
                            let bytes = py.detach(|| encode(black_box(&image), format, options)).unwrap();
                            PyBytes::new(py, &bytes)
                        };
                        let encoded =
                            blanket_codecs::metadata::_write_metadata(py, encoded.as_bytes(), format.as_str(), exif, xmp, &[], Vec::new()).unwrap();
                        fs::write(black_box(&output), encoded.as_bytes()).unwrap();
                    })
                });
                group.finish();
            }
        }
    });
}

criterion_group! { name = benches; config = blanket_bench_support::criterion(); targets = file_io, metadata_io }
criterion_main!(benches);
