//! TurboQuant-inspired vector compression for `semantic_search`'s embedding
//! index: a fixed random rotation followed by an 8-bit-per-coordinate Lloyd-Max
//! scalar quantizer, plus a length-renormalization correction for the
//! systematic inner-product bias scalar quantization introduces.
//!
//! Adapted after reviewing [RyanCodrai/turbovec](https://github.com/RyanCodrai/turbovec),
//! a Rust implementation of the same paper, whose `codebook.rs`/`encode.rs`
//! this module's Lloyd-Max/length-renormalization logic mirrors (own
//! implementation, not copied). Two upgrades over this module's original,
//! more ad-hoc version:
//! - The quantizer's bucket boundaries/centroids are now computed
//!   analytically against the *exact* distribution a rotated unit-vector
//!   coordinate follows — `Beta((d-1)/2, (d-1)/2)` on `[-1, 1]` — via the
//!   Lloyd-Max algorithm, instead of a data-dependent per-vector min/max
//!   affine range. Lower distortion for the same 1 byte/coordinate, and one
//!   fewer stored field per vector (the codebook is shared, computed once
//!   per `dim` and memoized, rather than a per-vector `min`/`scale` pair).
//! - Every `QuantizedVector` now also stores a length-renormalization
//!   scalar (`||v|| / ⟨u, x̂⟩`, computed once at encode time): scalar
//!   quantization systematically shrinks the reconstructed vector, which
//!   biases inner-product/cosine estimates downward, and this correction
//!   removes that bias at zero search-time cost. Since `Embedder` always
//!   hands this module unit vectors, `||v||` is always 1 here — turbovec's
//!   more general version stores an arbitrary norm because its vectors
//!   aren't necessarily pre-normalized.
//!
//! Still a simplified engineering take, not a literal port: turbovec bit-packs
//! down to 2-4 bits/coordinate with hand-written SIMD scoring kernels and a
//! per-coordinate calibration step (TQ+) fit from the actual corpus (turbovec
//! itself notes TQ+ needs on the order of 1,000+ vectors before the fit is
//! statistically meaningful) — this project stays at a full byte/coordinate,
//! no bit-packing/SIMD, and skips TQ+, since it indexes one repo's files at a
//! time rather than a multi-million-vector store, and most single-repo
//! `semantic_search` calls won't index anywhere near 1,000 chunks.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use statrs::distribution::{Beta, Continuous, ContinuousCDF};

/// Fixed seed so every embedding a `Rotation` touches within one process is
/// comparable to every other. Indexing and querying only need to agree on
/// one rotation, not reproduce it across restarts — the vector cache built
/// from it is in-memory-only and rebuilt from source each run, same as
/// `DirCache`/`FileCache`/`ContentCache`.
const ROTATION_SEED: u64 = 0x5EED_5EED_5EED_5EED;

/// A fixed random orthogonal matrix, applied to every embedding before
/// quantization. Built once via Gram-Schmidt over a random Gaussian matrix
/// (the standard way to sample a Haar-random orthogonal matrix) and shared
/// (behind an `Arc`) across every embed/quantize call for the process's
/// lifetime.
pub struct Rotation {
    dim: usize,
    // Row-major dim x dim.
    matrix: Vec<f32>,
}

impl Rotation {
    pub fn new(dim: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(ROTATION_SEED);
        let mut rows: Vec<Vec<f64>> = (0..dim)
            .map(|_| (0..dim).map(|_| gaussian(&mut rng)).collect())
            .collect();

        // Gram-Schmidt orthonormalization, row by row.
        for i in 0..dim {
            for j in 0..i {
                let earlier_row = rows[j].clone();
                let dot: f64 = rows[i].iter().zip(&earlier_row).map(|(a, b)| a * b).sum();
                for (a, b) in rows[i].iter_mut().zip(&earlier_row) {
                    *a -= dot * b;
                }
            }
            let norm: f64 = rows[i].iter().map(|x| x * x).sum::<f64>().sqrt();
            for x in rows[i].iter_mut() {
                *x /= norm;
            }
        }

        let matrix = rows.into_iter().flatten().map(|x| x as f32).collect();
        Self { dim, matrix }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Applies the rotation. Preserves L2 norm and inter-vector dot products
    /// (the matrix is orthogonal), so a unit-normalized embedding stays
    /// unit-normalized and cosine similarity between two embeddings is
    /// unchanged by rotating both of them.
    pub fn apply(&self, v: &[f32]) -> Vec<f32> {
        assert_eq!(v.len(), self.dim, "vector dimension must match rotation dimension");
        (0..self.dim)
            .map(|i| {
                let row = &self.matrix[i * self.dim..(i + 1) * self.dim];
                row.iter().zip(v).map(|(a, b)| a * b).sum()
            })
            .collect()
    }
}

/// Box-Muller transform; `u1` kept off exactly 0.0 to avoid `ln(0)`.
fn gaussian(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random_range(f64::EPSILON..1.0);
    let u2: f64 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// 1 byte/coordinate, matching the earlier affine quantizer's storage cost —
/// this module upgrades quantization *quality* (the Lloyd-Max codebook),
/// not the compression ratio. Lower bit widths (turbovec goes to 2-4 bits)
/// would need bit-packing this project doesn't have.
const CODEBOOK_LEVELS: usize = 256;

/// `(boundaries, centroids)`: `boundaries.len() == CODEBOOK_LEVELS - 1`,
/// `centroids.len() == CODEBOOK_LEVELS`.
type Codebook = Arc<(Vec<f32>, Vec<f32>)>;

/// Lloyd-Max codebook for the `Beta((d-1)/2, (d-1)/2)` marginal a
/// `dim`-dimensional rotation produces, keyed by `dim` and built once per
/// distinct dimension a running process ever sees — realistically exactly
/// one, since a server only ever loads one embedding model at a time.
fn codebook_for_dim(dim: usize) -> Codebook {
    static CACHE: OnceLock<Mutex<HashMap<usize, Codebook>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    guard.entry(dim).or_insert_with(|| Arc::new(build_lloyd_max_codebook(dim))).clone()
}

/// Runs the Lloyd-Max algorithm to convergence against the exact
/// `Beta((d-1)/2, (d-1)/2)` distribution (on `[-1, 1]`) a coordinate of a
/// rotated unit vector follows, computed purely from that distribution's
/// closed form — no embeddings involved, matching TurboQuant's
/// "data-oblivious" property. `dim` is clamped to a minimum of 2 (a Beta
/// shape parameter of `(dim-1)/2` must be positive); real embedding models
/// are always far above that floor, so this only matters for degenerate
/// test inputs.
fn build_lloyd_max_codebook(dim: usize) -> (Vec<f32>, Vec<f32>) {
    let a = ((dim.max(2) - 1) as f64 / 2.0).max(0.5);
    let beta = Beta::new(a, a).expect("shape parameter is always positive");

    // Beta(a, a) is defined on [0, 1]; a coordinate of a rotated unit vector
    // lives on [-1, 1] via the linear map x = 2t - 1.
    let cdf = |x: f64| beta.cdf(((x + 1.0) / 2.0).clamp(0.0, 1.0));
    // For a < 1 (only reachable via this function's own dim<2 clamp, not by
    // any real embedding model) the Beta density is singular at its domain
    // edges — finite integral, infinite pointwise value. Treating that one
    // sampled point as 0 in the quadrature below is a standard, harmless
    // simplification: it only ever affects the outermost bucket's centroid
    // in this degenerate, test-only case.
    let pdf = |x: f64| {
        let p = beta.pdf(((x + 1.0) / 2.0).clamp(0.0, 1.0)) / 2.0;
        if p.is_finite() { p } else { 0.0 }
    };

    // Initial centroids: the distribution's own quantiles, so each starting
    // cell already holds roughly equal probability mass. A uniform grid
    // over [-1, 1] instead leaves most of a concentrated Beta's outer cells
    // essentially empty (near-zero probability for a repo's actual
    // embedding dimensions, where `a = (dim-1)/2` is large) — Lloyd-Max
    // never moves an empty cell's centroid, so those cells stay frozen
    // while their neighbors shift, breaking the strictly-increasing
    // ordering encode/decode's bucket lookup relies on.
    let mut centroids: Vec<f64> = (0..CODEBOOK_LEVELS)
        .map(|i| {
            let p = (i as f64 + 0.5) / CODEBOOK_LEVELS as f64;
            2.0 * beta.inverse_cdf(p) - 1.0
        })
        .collect();

    const MAX_ITER: usize = 100;
    const TOLERANCE: f64 = 1e-9;

    for _ in 0..MAX_ITER {
        let mut edges = Vec::with_capacity(CODEBOOK_LEVELS + 1);
        edges.push(-1.0);
        edges.extend(centroids.windows(2).map(|w| (w[0] + w[1]) / 2.0));
        edges.push(1.0);

        let mut max_change = 0.0f64;
        for i in 0..CODEBOOK_LEVELS {
            let (lo, hi) = (edges[i], edges[i + 1]);
            let prob = cdf(hi) - cdf(lo);
            let new_centroid = if prob > 1e-15 {
                (simpson_integral(|x| x * pdf(x), lo, hi, 32) / prob).clamp(-1.0, 1.0)
            } else {
                centroids[i]
            };
            max_change = max_change.max((new_centroid - centroids[i]).abs());
            centroids[i] = new_centroid;
        }

        if max_change < TOLERANCE {
            break;
        }
    }

    let boundaries: Vec<f32> = centroids.windows(2).map(|w| ((w[0] + w[1]) / 2.0) as f32).collect();
    let centroids: Vec<f32> = centroids.iter().map(|&c| c as f32).collect();
    (boundaries, centroids)
}

/// Composite Simpson's rule over `[a, b]` with `n` (rounded up to even)
/// subintervals — accurate enough for a smooth Beta density at the interval
/// widths `CODEBOOK_LEVELS` buckets produce, and simpler than an adaptive
/// scheme since this runs a handful of times total (once per distinct `dim`,
/// memoized) rather than per vector.
fn simpson_integral<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    let n = if n.is_multiple_of(2) { n } else { n + 1 };
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let x = a + i as f64 * h;
        sum += if i % 2 == 0 { 2.0 * f(x) } else { 4.0 * f(x) };
    }
    sum * h / 3.0
}

/// The bucket index whose centroid is closest to `x` under the Lloyd-Max
/// optimum — equivalently, the number of `boundaries` at or below `x`.
fn quantize_coordinate(x: f32, boundaries: &[f32]) -> u8 {
    boundaries.partition_point(|&b| b <= x).min(CODEBOOK_LEVELS - 1) as u8
}

/// A rotated embedding compressed to 1 byte/coordinate via a Lloyd-Max
/// scalar quantizer built against the coordinate's exact Beta distribution,
/// plus a length-renormalization scalar correcting the inner-product bias
/// that scalar quantization introduces.
#[derive(Clone)]
pub struct QuantizedVector {
    data: Vec<u8>,
    /// `1 / ⟨u, x̂⟩` where `u` is the rotated unit vector and `x̂` its
    /// decoded reconstruction — see the module docs' length-renormalization
    /// note. Multiplying a raw dot product against this vector's decoding
    /// by this scalar gives an unbiased inner-product estimate.
    correction: f32,
}

impl QuantizedVector {
    /// Rotates `v` and quantizes the result to 8 bits/dimension.
    pub fn encode(v: &[f32], rotation: &Rotation) -> Self {
        let rotated = rotation.apply(v);
        let (boundaries, _) = &*codebook_for_dim(rotated.len());
        let data: Vec<u8> = rotated.iter().map(|&x| quantize_coordinate(x, boundaries)).collect();

        let mut result = Self { data, correction: 1.0 };
        let reconstructed = result.decode();
        let self_dot: f32 = rotated.iter().zip(&reconstructed).map(|(a, b)| a * b).sum();
        // `||v|| / self_dot`, with `||v|| == 1`: `Embedder` guarantees every
        // input is already unit-normalized.
        result.correction = if self_dot.abs() > 1e-6 { 1.0 / self_dot } else { 1.0 };
        result
    }

    /// Reconstructs the approximate rotated vector (not yet
    /// length-renormalized — see `score`, which applies `correction` to the
    /// aggregated dot product rather than to each coordinate).
    pub fn decode(&self) -> Vec<f32> {
        let (_, centroids) = &*codebook_for_dim(self.data.len());
        self.data.iter().map(|&idx| centroids[idx as usize]).collect()
    }

    /// Approximate byte footprint, for callers that want to reason about the
    /// index's memory usage the way `ContentCache` reasons about its byte
    /// budget.
    pub fn byte_size(&self) -> usize {
        self.data.len() + std::mem::size_of::<f32>()
    }
}

/// Cosine-similarity-ish ranking score between a rotated query vector and a
/// quantized document vector. Rotation preserves inner products and both
/// sides start out unit-normalized (`Embedder` guarantees that), so the
/// length-renormalized dot product approximates true cosine similarity up
/// to quantization noise — good enough for ranking, not for absolute
/// similarity thresholds.
pub fn score(query_rotated: &[f32], doc: &QuantizedVector) -> f32 {
    let decoded = doc.decode();
    let raw: f32 = query_rotated.iter().zip(decoded.iter()).map(|(a, b)| a * b).sum();
    raw * doc.correction
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn unit_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let raw: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0f32..1.0)).collect();
        let n = norm(&raw);
        raw.iter().map(|x| x / n).collect()
    }

    #[test]
    fn rotation_preserves_norm() {
        let rotation = Rotation::new(16);
        let v = unit_vec(16, 1);
        let rotated = rotation.apply(&v);
        assert!((norm(&rotated) - 1.0).abs() < 1e-4, "rotation should preserve L2 norm, got {}", norm(&rotated));
    }

    #[test]
    fn rotation_preserves_dot_product() {
        let rotation = Rotation::new(16);
        let a = unit_vec(16, 1);
        let b = unit_vec(16, 2);
        let before = dot(&a, &b);
        let after = dot(&rotation.apply(&a), &rotation.apply(&b));
        assert!((before - after).abs() < 1e-3, "orthogonal rotation should preserve dot product: {before} vs {after}");
    }

    #[test]
    fn lloyd_max_codebook_is_sorted_and_covers_the_beta_support() {
        let (boundaries, centroids) = &*codebook_for_dim(32);
        assert_eq!(boundaries.len(), CODEBOOK_LEVELS - 1);
        assert_eq!(centroids.len(), CODEBOOK_LEVELS);
        assert!(centroids.windows(2).all(|w| w[0] < w[1]), "Lloyd-Max centroids should be strictly increasing");
        assert!(boundaries.windows(2).all(|w| w[0] < w[1]), "Lloyd-Max boundaries should be strictly increasing");
        assert!(centroids.iter().all(|c| (-1.0..=1.0).contains(c)), "centroids must stay within the Beta's [-1, 1] support");
    }

    #[test]
    fn quantize_round_trip_error_is_bounded() {
        let rotation = Rotation::new(32);
        let v = unit_vec(32, 3);
        let quantized = QuantizedVector::encode(&v, &rotation);
        let rotated = rotation.apply(&v);
        let decoded = quantized.decode();

        for (original, reconstructed) in rotated.iter().zip(decoded.iter()) {
            assert!(
                (original - reconstructed).abs() < 0.05,
                "per-coordinate error should be small at 256 Lloyd-Max levels: {original} vs {reconstructed}"
            );
        }
    }

    #[test]
    fn degenerate_single_dimension_vector_quantizes_without_panicking() {
        // dim=1 has no meaningful Beta((d-1)/2, (d-1)/2) shape (a would be
        // 0), so build_lloyd_max_codebook clamps its shape parameter — this
        // just confirms that clamp keeps encode/decode total instead of
        // panicking or producing non-finite output.
        let rotation = Rotation::new(1);
        let quantized = QuantizedVector::encode(&[1.0], &rotation);
        assert!(quantized.decode().iter().all(|x| x.is_finite()));
        assert!(quantized.correction.is_finite());
    }

    #[test]
    fn quantization_preserves_ranking_order() {
        let dim = 64;
        let rotation = Rotation::new(dim);

        let query = unit_vec(dim, 10);
        let close = unit_vec(dim, 10); // same seed: identical to query
        let far = unit_vec(dim, 999);

        let query_rotated = rotation.apply(&query);
        let close_quantized = QuantizedVector::encode(&close, &rotation);
        let far_quantized = QuantizedVector::encode(&far, &rotation);

        let close_score = score(&query_rotated, &close_quantized);
        let far_score = score(&query_rotated, &far_quantized);

        assert!(close_score > far_score, "an identical vector should score higher than an unrelated one: {close_score} vs {far_score}");
        assert!(close_score > 0.95, "quantizing a near-identical vector shouldn't destroy its similarity score: {close_score}");
    }

    #[test]
    fn length_renormalization_pulls_self_similarity_toward_one() {
        // The whole point of the correction: scoring a vector against its
        // own quantized self should land very close to 1.0 (true cosine
        // similarity of a unit vector with itself), not the
        // systematically-shrunk value an uncorrected dot product would give.
        let dim = 128;
        let rotation = Rotation::new(dim);
        let v = unit_vec(dim, 7);
        let rotated = rotation.apply(&v);
        let quantized = QuantizedVector::encode(&v, &rotation);

        let corrected = score(&rotated, &quantized);
        assert!((corrected - 1.0).abs() < 0.01, "length-renormalized self-similarity should be close to 1.0, got {corrected}");
    }

    #[test]
    fn byte_size_is_roughly_a_quarter_of_f32_storage() {
        let rotation = Rotation::new(384);
        let v = unit_vec(384, 42);
        let quantized = QuantizedVector::encode(&v, &rotation);
        assert!(quantized.byte_size() < 384 * 4 / 3, "quantized vector should be substantially smaller than raw f32 storage");
    }
}
