//! Allocation accounting for the PCM16 decode boundary: owned decode vs
//! borrowed (zero-copy) decode.
//!
//! This lives in its own integration test rather than in
//! `benches/audio_boundary_decode.rs` on purpose. Counting allocations requires
//! installing an instrumented `#[global_allocator]`, and every allocation in
//! that binary then pays for an atomic counter update. Only the owned arm
//! allocates, so sharing a binary with the Criterion latency arms would tax the
//! owned arm alone and inflate the reported speedup. A separate test binary
//! keeps the instrumented allocator away from the timings.
//!
//! The payload is one realtime frame: 20 ms of 24 kHz mono PCM16 =
//! 480 samples = 960 bytes. That is the unit a provider transport actually
//! hands to the codec, so per-iteration numbers map directly onto per-frame
//! cost.
//!
//! Both arms consume the decoded samples identically (a checksum fold behind
//! `black_box`). Without that, the borrowed arm has no observable effect and
//! LLVM is free to delete it, which would report a fictional result.

use std::alloc::System;
use std::borrow::Cow;
use std::hint::black_box;

use adk_realtime::audio::{AudioChunk, AudioFormat};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, Stats, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

/// 20 ms at 24 kHz mono.
const SAMPLES_PER_FRAME: usize = 480;
/// PCM16 is two bytes per sample.
const FRAME_BYTES: usize = SAMPLES_PER_FRAME * size_of::<i16>();

/// Measurement loop lengths. Long enough to make a per-iteration figure
/// meaningful, short enough to stay instant.
const WARMUP_ITERATIONS: usize = 1_000;
const MEASURED_ITERATIONS: usize = 10_000;

/// Build one deterministic frame. The values span positive and negative
/// amplitudes so the checksum fold cannot be constant-folded away.
fn frame() -> AudioChunk {
    let samples: Vec<i16> =
        (0..SAMPLES_PER_FRAME).map(|i| ((i as i32 * 2617) % 65_536 - 32_768) as i16).collect();
    let chunk = AudioChunk::from_i16_samples(&samples, AudioFormat::pcm16_24khz());
    assert_eq!(chunk.data.len(), FRAME_BYTES);
    chunk
}

/// The previous `AudioChunk::to_i16_samples` body, kept local to this test.
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
    chunk.to_i16_samples().expect("test frame is valid PCM16")
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

/// Count allocations for one arm. No timing here at all.
///
/// The `Region` is opened only after warm-up, so allocator growth and any
/// harness setup are excluded. `operation` returns the checksum, which is
/// black-boxed, so each arm still performs its full work.
fn measure_allocations<F: FnMut() -> i64>(mut operation: F) -> Stats {
    for _ in 0..WARMUP_ITERATIONS {
        black_box(operation());
    }

    let region = Region::new(GLOBAL);
    for _ in 0..MEASURED_ITERATIONS {
        black_box(operation());
    }
    region.change()
}

fn print_table(owned: &Stats, borrowed: &Stats) {
    println!("audio_boundary_decode allocations");
    println!("os={},arch={}", std::env::consts::OS, std::env::consts::ARCH);
    println!("frame_bytes={FRAME_BYTES},samples_per_frame={SAMPLES_PER_FRAME}");
    println!("warmup_iterations={WARMUP_ITERATIONS},measured_iterations={MEASURED_ITERATIONS}");
    println!("arm,allocations_per_iter,reallocations_per_iter,allocated_bytes_per_iter");
    for (arm, stats) in [("owned_decode", owned), ("borrowed_decode", borrowed)] {
        println!(
            "{},{},{},{}",
            arm,
            stats.allocations / MEASURED_ITERATIONS,
            stats.reallocations / MEASURED_ITERATIONS,
            stats.bytes_allocated / MEASURED_ITERATIONS,
        );
    }
}

/// The owned decode allocates exactly one buffer of `FRAME_BYTES` per frame;
/// the borrowed decode allocates nothing at all.
///
/// Assertions are on exact per-iteration integer totals, not averages, so the
/// invariant is deterministic rather than a threshold that can drift.
#[test]
fn pcm16_decode_allocation_invariants() {
    let chunk = frame();
    assert_arms_agree(&chunk);

    let owned = measure_allocations(|| {
        let samples = decode_owned(black_box(chunk.data.as_slice()));
        consume(samples.as_ref())
    });
    let borrowed = measure_allocations(|| {
        let samples = decode_borrowed(black_box(&chunk));
        consume(samples.as_ref())
    });

    print_table(&owned, &borrowed);

    assert_eq!(
        owned.allocations, MEASURED_ITERATIONS,
        "owned decode must allocate exactly one buffer per frame"
    );
    assert_eq!(
        owned.reallocations, 0,
        "owned decode pre-sizes its buffer, so it never reallocates"
    );
    assert_eq!(
        owned.bytes_allocated,
        MEASURED_ITERATIONS * FRAME_BYTES,
        "owned decode must allocate exactly {FRAME_BYTES} bytes per frame"
    );

    assert_eq!(borrowed.allocations, 0, "borrowed decode must not allocate");
    assert_eq!(borrowed.reallocations, 0, "borrowed decode must not reallocate");
    assert_eq!(borrowed.bytes_allocated, 0, "borrowed decode must not allocate any bytes");
}
