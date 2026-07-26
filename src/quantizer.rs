//! TurboQuant-inspired vector compression for `semantic_search`'s embedding
//! index: a fixed random rotation concentrates each embedding's coordinate
//! magnitudes before a plain per-vector affine int8 scalar quantizer shrinks
//! it ~4x versus storing raw `f32`s.
//!
//! This is a simplified engineering take on Google Research's TurboQuant
//! paper, not a literal reproduction: TurboQuant itself uses a structured
//! O(d log d) fast rotation and a Beta-distribution-optimized non-uniform
//! scalar quantizer (plus a QJL residual step for unbiased inner-product
//! estimation). Here a dense O(d²) random-orthogonal rotation and a standard
//! uniform affine quantizer stand in for those — accurate enough at this
//! project's index sizes (single-repo code search, not a billion-vector
//! index) and far simpler to implement correctly.

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

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

/// A rotated embedding compressed to 1 byte/dimension via a per-vector
/// affine (min/max) scalar quantizer.
#[derive(Clone)]
pub struct QuantizedVector {
    data: Vec<u8>,
    min: f32,
    // (max - min) / 255; 0.0 when every rotated coordinate is identical.
    scale: f32,
}

impl QuantizedVector {
    /// Rotates `v` and quantizes the result to 8 bits/dimension.
    pub fn encode(v: &[f32], rotation: &Rotation) -> Self {
        let rotated = rotation.apply(v);
        let min = rotated.iter().copied().fold(f32::INFINITY, f32::min);
        let max = rotated.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = if max > min { (max - min) / 255.0 } else { 0.0 };

        let data = rotated
            .iter()
            .map(|&x| {
                if scale > 0.0 {
                    (((x - min) / scale).round().clamp(0.0, 255.0)) as u8
                } else {
                    0
                }
            })
            .collect();

        Self { data, min, scale }
    }

    /// Reconstructs the approximate rotated vector.
    pub fn decode(&self) -> Vec<f32> {
        self.data.iter().map(|&b| self.min + self.scale * b as f32).collect()
    }

    /// Approximate byte footprint, for callers that want to reason about the
    /// index's memory usage the way `ContentCache` reasons about its byte
    /// budget.
    pub fn byte_size(&self) -> usize {
        self.data.len() + std::mem::size_of::<f32>() * 2
    }
}

/// Cosine-similarity-ish ranking score between a rotated query vector and a
/// quantized document vector. Rotation preserves inner products and both
/// sides start out unit-normalized (`Embedder` guarantees that), so this dot
/// product approximates true cosine similarity up to quantization noise —
/// good enough for ranking, not for absolute similarity thresholds.
pub fn score(query_rotated: &[f32], doc: &QuantizedVector) -> f32 {
    let decoded = doc.decode();
    query_rotated.iter().zip(decoded.iter()).map(|(a, b)| a * b).sum()
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
    fn quantize_round_trip_error_is_bounded() {
        let rotation = Rotation::new(32);
        let v = unit_vec(32, 3);
        let quantized = QuantizedVector::encode(&v, &rotation);
        let rotated = rotation.apply(&v);
        let decoded = quantized.decode();

        for (original, reconstructed) in rotated.iter().zip(decoded.iter()) {
            assert!(
                (original - reconstructed).abs() <= quantized.scale.max(1e-6),
                "per-coordinate error should stay within one quantization step: {original} vs {reconstructed}"
            );
        }
    }

    #[test]
    fn constant_vector_quantizes_without_division_by_zero() {
        // A rotation of a *general* constant vector isn't itself constant
        // (rotating spreads it across coordinates), so the one dimension
        // guaranteed to always hit min == max after rotation is a
        // single-element vector — there's only one coordinate, so it's
        // trivially its own min and max.
        let rotation = Rotation::new(1);
        let quantized = QuantizedVector::encode(&[1.0], &rotation);
        assert_eq!(quantized.scale, 0.0);
        assert!(quantized.decode().iter().all(|x| x.is_finite()));
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
    fn byte_size_is_roughly_a_quarter_of_f32_storage() {
        let rotation = Rotation::new(384);
        let v = unit_vec(384, 42);
        let quantized = QuantizedVector::encode(&v, &rotation);
        assert!(quantized.byte_size() < 384 * 4 / 3, "quantized vector should be substantially smaller than raw f32 storage");
    }
}
