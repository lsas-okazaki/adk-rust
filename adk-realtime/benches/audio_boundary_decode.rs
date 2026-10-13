//! Decode boundary benchmark: owned PCM16 decode vs borrowed (zero-copy) decode.
//!
//! The payload is one realtime frame: 20 ms of 24 kHz mono PCM16 =
//! 480 samples = 960 bytes. That is the unit a provider transport actually
//! hands to the codec, so per-iteration numbers map directly onto per-frame cost.
//!
//! This binary measures **latency only**, with Criterion. Criterion batches
//! iterations and amortizes the timer, so the measurement is not floored by
//! `Instant::now()` overhead (~20-25 ns), which would swamp a sub-100 ns
//! operation.
//!
//! Allocation counting deliberately lives in
//! `tests/audio_allocation_tests.rs`, not here. It needs an instrumented
//! `#[global_allocator]`, which taxes every allocation with an atomic counter
//! update — and only the owned arm allocates, so keeping it in this binary
//! would penalize that arm alone and inflate the reported speedup.
//!
//! Both latency arms consume the decoded samples identically (a checksum fold
//! behind `black_box`). Without that, the borrowed arm has no observable effect
//! and LLVM is free to delete it, which would report a fictional speedup.

use std::borrow::Cow;
use std::hint::black_box;
use std::time::Duration;

use adk_realtime::audio::{AudioChunk, AudioFormat};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

/// 20 ms at 24 kHz mono.
const SAMPLES_PER_FRAME: usize = 480;
/// PCM16 is two bytes per sample.
const FRAME_BYTES: usize = SAMPLES_PER_FRAME * size_of::<i16>();

/// Build one deterministic frame. The values span positive and negative
/// amplitudes so the checksum fold cannot be constant-folded away.
fn frame() -> AudioChunk {
    let samples: Vec<i16> =
        (0..SAMPLES_PER_FRAME).map(|i| ((i as i32 * 2617) % 65_536 - 32_768) as i16).collect();
    let chunk = AudioChunk::from_i16_samples(&samples, AudioFormat::pcm16_24khz());
    assert_eq!(chunk.data.len(), FRAME_BYTES);
    chunk
}

/// The previous `AudioChunk::to_i16_samples` body, kept local to this bench.
///
/// Holding a copy here keeps the comparison honest after the production source
/// changed: the "before" arm is the code that actually shipped, not a
/// reconstruction that drifts with the crate.
fn decode_owned(data: &[u8]) -> Vec<i16> {
    let mut samples = Vec::with_capacity(data.len() / size_of::<i16>());
    // Deliberately not a constant size: this "before" snapshot must keep the
    // shipped `chunks_exact` idiom verbatim, and newer clippy only suggests
    // `as_chunks` for constant sizes.
    let lane_bytes = size_of::<i16>();
    for chunk in data.chunks_exact(lane_bytes) {
        samples.push(i16::from_le_bytes([chunk[0], chunk[1]]));
    }
    samples
}

/// The current production path.
fn decode_borrowed(chunk: &AudioChunk) -> Cow<'_, [i16]> {
    chunk.to_i16_samples().expect("benchmark frame is valid PCM16")
}

/// Identical consumption for both arms: fold every sample into a checksum.
///
/// `i64` accumulation cannot overflow for a frame of `i16` values, so the fold
/// is total and both arms do exactly the same amount of downstream work.
fn consume(samples: &[i16]) -> i64 {
    samples.iter().map(|sample| *sample as i64).sum()
}

/// Assert the two arms agree, once, outside every measured region.
fn assert_arms_agree(chunk: &AudioChunk) {
    let owned = decode_owned(&chunk.data);
    let borrowed = decode_borrowed(chunk);
    assert_eq!(
        owned.as_slice(),
        borrowed.as_ref(),
        "owned and borrowed decode disagree; the comparison would be meaningless"
    );
    assert_eq!(consume(&owned), consume(borrowed.as_ref()));
}

fn audio_boundary_decode(criterion: &mut Criterion) {
    let chunk = frame();
    assert_arms_agree(&chunk);

    let mut group = criterion.benchmark_group("audio_boundary_decode");
    group.throughput(Throughput::Bytes(FRAME_BYTES as u64));

    group.bench_function("owned_decode", |bencher| {
        bencher.iter(|| {
            let samples = decode_owned(black_box(chunk.data.as_slice()));
            black_box(consume(samples.as_ref()))
        });
    });

    group.bench_function("borrowed_decode", |bencher| {
        bencher.iter(|| {
            let samples = decode_borrowed(black_box(&chunk));
            black_box(consume(samples.as_ref()))
        });
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .sample_size(100);
    targets = audio_boundary_decode
}

criterion_main!(benches);
