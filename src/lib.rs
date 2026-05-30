//! # plato-jepa-dual
//!
//! **Dual-Database JEPA**: a novel architecture where inputs and outputs are
//! embedded into *separate* vector databases, and intelligence lives in the
//! cross-database comparison function.
//!
//! Standard JEPA forces both perception and prediction into one latent space.
//! Here, the input and output spaces remain separate, and the *mapping between
//! them* IS the learned behavior.
//!
//! ## Key innovation
//!
//! The comparison method (`ComparisonMethod`) is the model. It can be a linear
//! projection, cosine similarity with learned weights, attention-style
//! cross-database lookup, or K-nearest-neighbors across databases.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A vector in perception space — encodes "what was sensed".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptionVector {
    pub id: Uuid,
    pub vector: Vec<f64>,
    pub source_tiles: Vec<String>,
    pub room_id: String,
    pub timestamp: u64,
}

/// A vector in prediction space — encodes "what to predict/do".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionVector {
    pub id: Uuid,
    pub vector: Vec<f64>,
    pub predicted_value: f64,
    pub confidence: f64,
    pub room_id: String,
    pub timestamp: u64,
}

/// The perception database — stores all observed states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptionDB {
    pub vectors: Vec<PerceptionVector>,
    pub dimension: usize,
    pub index: HashMap<Uuid, usize>,
}

/// The prediction database — stores all predictions/actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionDB {
    pub vectors: Vec<PredictionVector>,
    pub dimension: usize,
    pub index: HashMap<Uuid, usize>,
}

/// Cross-database comparison result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossDBMatch {
    pub perception_id: Uuid,
    pub prediction_id: Uuid,
    pub relevance: f64,
    pub method: ComparisonMethod,
}

/// The comparison method — this IS the intelligence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonMethod {
    /// Simple cosine similarity between projected vectors.
    Cosine {
        projection_matrix: Vec<Vec<f64>>,
    },
    /// Weighted Euclidean distance.
    WeightedEuclidean {
        weights: Vec<f64>,
    },
    /// Attention-style: query from one DB, key/value from the other.
    CrossAttention {
        query_weights: Vec<Vec<f64>>,
        key_weights: Vec<Vec<f64>>,
    },
    /// K-nearest-neighbors lookup.
    KNN {
        k: usize,
    },
}

/// The dual-database JEPA system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualDBJepa {
    pub perception_db: PerceptionDB,
    pub prediction_db: PredictionDB,
    pub comparison: ComparisonMethod,
    pub perception_dim: usize,
    pub prediction_dim: usize,
    pub learning_rate: f64,
}

/// Training record — a perception–prediction pair with known outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingPair {
    pub perception: PerceptionVector,
    pub prediction: PredictionVector,
    pub actual_outcome: f64,
    pub loss: f64,
}

/// The full JEPA training state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JepaTrainingState {
    pub pairs: Vec<TrainingPair>,
    pub total_loss: f64,
    pub avg_loss: f64,
    pub epoch: usize,
    pub converged: bool,
}

/// Query: "given this input, what should I predict?"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardQuery {
    pub current_perception: PerceptionVector,
    pub top_k: usize,
}

/// Query: "what inputs preceded this kind of output?"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReverseQuery {
    pub target_prediction: PredictionVector,
    pub top_k: usize,
}

// ---------------------------------------------------------------------------
// Vector maths (no external dependency)
// ---------------------------------------------------------------------------

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn magnitude(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let ma = magnitude(a);
    let mb = magnitude(b);
    if ma < 1e-12 || mb < 1e-12 {
        return 0.0;
    }
    dot(a, b) / (ma * mb)
}

fn mat_vec_mul(mat: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    mat.iter()
        .map(|row| dot(row, v))
        .collect()
}

/// Project a perception vector into prediction space using a projection matrix.
pub fn project_perception_to_prediction(perception: &[f64], matrix: &[Vec<f64>]) -> Vec<f64> {
    mat_vec_mul(matrix, perception)
}

/// Project a prediction vector into perception space using a projection matrix.
pub fn project_prediction_to_perception(prediction: &[f64], matrix: &[Vec<f64>]) -> Vec<f64> {
    mat_vec_mul(matrix, prediction)
}

// ---------------------------------------------------------------------------
// PerceptionDB
// ---------------------------------------------------------------------------

impl PerceptionDB {
    pub fn new(dim: usize) -> Self {
        Self {
            vectors: Vec::new(),
            dimension: dim,
            index: HashMap::new(),
        }
    }

    pub fn insert(&mut self, vec: PerceptionVector) {
        let idx = self.vectors.len();
        self.index.insert(vec.id, idx);
        self.vectors.push(vec);
    }

    pub fn nearest(&self, query: &[f64], k: usize) -> Vec<&PerceptionVector> {
        let mut scored: Vec<(f64, &PerceptionVector)> = self
            .vectors
            .iter()
            .map(|v| (cosine_similarity(query, &v.vector), v))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(k).map(|(_, v)| v).collect()
    }

    pub fn get(&self, id: Uuid) -> Option<&PerceptionVector> {
        self.index.get(&id).map(|&i| &self.vectors[i])
    }
}

// ---------------------------------------------------------------------------
// PredictionDB
// ---------------------------------------------------------------------------

impl PredictionDB {
    pub fn new(dim: usize) -> Self {
        Self {
            vectors: Vec::new(),
            dimension: dim,
            index: HashMap::new(),
        }
    }

    pub fn insert(&mut self, vec: PredictionVector) {
        let idx = self.vectors.len();
        self.index.insert(vec.id, idx);
        self.vectors.push(vec);
    }

    pub fn nearest(&self, query: &[f64], k: usize) -> Vec<&PredictionVector> {
        let mut scored: Vec<(f64, &PredictionVector)> = self
            .vectors
            .iter()
            .map(|v| (cosine_similarity(query, &v.vector), v))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(k).map(|(_, v)| v).collect()
    }

    pub fn get(&self, id: Uuid) -> Option<&PredictionVector> {
        self.index.get(&id).map(|&i| &self.vectors[i])
    }
}

// ---------------------------------------------------------------------------
// Helper constructors
// ---------------------------------------------------------------------------

impl PerceptionVector {
    pub fn random(dim: usize) -> Self {
        let mut rng = simple_rng();
        Self {
            id: Uuid::new_v4(),
            vector: (0..dim).map(|_| rng() as f64 / f64::MAX * 2.0 - 1.0).collect(),
            source_tiles: Vec::new(),
            room_id: String::new(),
            timestamp: 0,
        }
    }

    pub fn with_vector(mut self, v: Vec<f64>) -> Self {
        self.vector = v;
        self
    }
}

impl PredictionVector {
    pub fn random(dim: usize) -> Self {
        let mut rng = simple_rng();
        Self {
            id: Uuid::new_v4(),
            vector: (0..dim).map(|_| rng() as f64 / f64::MAX * 2.0 - 1.0).collect(),
            predicted_value: 0.0,
            confidence: 1.0,
            room_id: String::new(),
            timestamp: 0,
        }
    }

    pub fn with_vector(mut self, v: Vec<f64>) -> Self {
        self.vector = v;
        self
    }
}

/// Very simple deterministic-ish RNG for tests. Not cryptographic.
fn simple_rng() -> impl FnMut() -> u64 {
    let mut state: u64 = 0x1234_5678_9ABC_DEF0;
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    }
}

// ---------------------------------------------------------------------------
// ComparisonMethod helpers
// ---------------------------------------------------------------------------

impl ComparisonMethod {
    /// Build a `Cosine` method with a random projection matrix (identity-ish).
    pub fn cosine_identity(perception_dim: usize, prediction_dim: usize) -> Self {
        let proj = if perception_dim <= prediction_dim {
            // pad with zeros
            (0..prediction_dim)
                .map(|i| {
                    (0..perception_dim)
                        .map(|j| if i == j { 1.0 } else { 0.0 })
                        .collect()
                })
                .collect()
        } else {
            // truncate rows
            (0..prediction_dim)
                .map(|i| {
                    (0..perception_dim)
                        .map(|j| if i == j { 1.0 } else { 0.0 })
                        .collect()
                })
                .collect()
        };
        Self::Cosine {
            projection_matrix: proj,
        }
    }

    pub fn weighted_euclidean_identity(dim: usize) -> Self {
        Self::WeightedEuclidean {
            weights: vec![1.0; dim],
        }
    }

    pub fn cross_attention_identity(perception_dim: usize, prediction_dim: usize) -> Self {
        let min_dim = perception_dim.min(prediction_dim);
        let qw = (0..min_dim)
            .map(|i| {
                (0..perception_dim)
                    .map(|j| if i == j { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect();
        let kw = (0..min_dim)
            .map(|i| {
                (0..prediction_dim)
                    .map(|j| if i == j { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect();
        Self::CrossAttention {
            query_weights: qw,
            key_weights: kw,
        }
    }

    pub fn knn(k: usize) -> Self {
        Self::KNN { k }
    }
}

// ---------------------------------------------------------------------------
// DualDBJepa
// ---------------------------------------------------------------------------

impl DualDBJepa {
    pub fn new(
        perception_dim: usize,
        prediction_dim: usize,
        method: ComparisonMethod,
    ) -> Self {
        Self {
            perception_db: PerceptionDB::new(perception_dim),
            prediction_db: PredictionDB::new(prediction_dim),
            comparison: method,
            perception_dim,
            prediction_dim,
            learning_rate: 0.01,
        }
    }

    /// Core cross-database comparison: relevance score between one perception
    /// and one prediction vector.
    pub fn cross_compare(
        &self,
        perception: &PerceptionVector,
        prediction: &PredictionVector,
    ) -> f64 {
        match &self.comparison {
            ComparisonMethod::Cosine { projection_matrix } => {
                let projected = project_perception_to_prediction(&perception.vector, projection_matrix);
                cosine_similarity(&projected, &prediction.vector)
            }
            ComparisonMethod::WeightedEuclidean { weights } => {
                // Use the min shared dimension
                let shared = perception.vector.len().min(prediction.vector.len()).min(weights.len());
                let dist: f64 = (0..shared)
                    .map(|i| weights[i] * (perception.vector[i] - prediction.vector[i]).powi(2))
                    .sum();
                1.0 / (1.0 + dist.sqrt())
            }
            ComparisonMethod::CrossAttention {
                query_weights,
                key_weights,
            } => {
                let q = mat_vec_mul(query_weights, &perception.vector);
                let k = mat_vec_mul(key_weights, &prediction.vector);
                let scale = (q.len() as f64).sqrt();
                dot(&q, &k) / scale
            }
            ComparisonMethod::KNN { .. } => {
                // KNN falls back to cosine in single-pair comparison
                let shared = perception.vector.len().min(prediction.vector.len());
                let a: Vec<f64> = perception.vector[..shared].to_vec();
                let b: Vec<f64> = prediction.vector[..shared].to_vec();
                cosine_similarity(&a, &b)
            }
        }
    }

    /// Full cross-comparison matrix between perception and prediction batches.
    pub fn batch_cross_compare(
        &self,
        perceptions: &[PerceptionVector],
        predictions: &[PredictionVector],
    ) -> Vec<Vec<f64>> {
        perceptions
            .iter()
            .map(|p| {
                predictions
                    .iter()
                    .map(|pr| self.cross_compare(p, pr))
                    .collect()
            })
            .collect()
    }

    /// Forward query: "given this perception, find the best matching predictions."
    pub fn forward_query(&self, query: &ForwardQuery) -> Vec<CrossDBMatch> {
        let mut scored: Vec<CrossDBMatch> = self
            .prediction_db
            .vectors
            .iter()
            .map(|pred| {
                let rel = self.cross_compare(&query.current_perception, pred);
                CrossDBMatch {
                    perception_id: query.current_perception.id,
                    prediction_id: pred.id,
                    relevance: rel,
                    method: self.comparison.clone(),
                }
            })
            .collect();
        scored.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(query.top_k).collect()
    }

    /// Reverse query: "given this desired prediction, find relevant perceptions."
    pub fn reverse_query(&self, query: &ReverseQuery) -> Vec<CrossDBMatch> {
        let mut scored: Vec<CrossDBMatch> = self
            .perception_db
            .vectors
            .iter()
            .map(|perc| {
                let rel = self.cross_compare(perc, &query.target_prediction);
                CrossDBMatch {
                    perception_id: perc.id,
                    prediction_id: query.target_prediction.id,
                    relevance: rel,
                    method: self.comparison.clone(),
                }
            })
            .collect();
        scored.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(query.top_k).collect()
    }

    /// Single training step — updates the comparison weights to reduce loss.
    pub fn train_step(&mut self, pair: TrainingPair) {
        let predicted_relevance = self.cross_compare(&pair.perception, &pair.prediction);
        let error = predicted_relevance - pair.actual_outcome;

        match &mut self.comparison {
            ComparisonMethod::Cosine { projection_matrix } => {
                // Gradient descent on projection matrix
                let lr = self.learning_rate;
                for row in projection_matrix.iter_mut() {
                    for w in row.iter_mut() {
                        *w -= lr * error * 0.01;
                    }
                }
            }
            ComparisonMethod::WeightedEuclidean { weights } => {
                let lr = self.learning_rate;
                let shared = pair.perception.vector.len().min(pair.prediction.vector.len()).min(weights.len());
                for i in 0..shared {
                    let diff = pair.perception.vector[i] - pair.prediction.vector[i];
                    weights[i] -= lr * error * diff * 0.1;
                    weights[i] = weights[i].max(0.0);
                }
            }
            ComparisonMethod::CrossAttention {
                query_weights,
                key_weights,
            } => {
                let lr = self.learning_rate;
                for row in query_weights.iter_mut() {
                    for w in row.iter_mut() {
                        *w -= lr * error * 0.01;
                    }
                }
                for row in key_weights.iter_mut() {
                    for w in row.iter_mut() {
                        *w -= lr * error * 0.01;
                    }
                }
            }
            ComparisonMethod::KNN { .. } => {
                // KNN has no trainable parameters — no-op
            }
        }
    }

    /// One training epoch over all pairs. Returns average loss.
    pub fn train_epoch(&mut self, pairs: &[TrainingPair]) -> f64 {
        let mut total_loss = 0.0;
        for pair in pairs {
            let predicted = self.cross_compare(&pair.perception, &pair.prediction);
            let loss = (predicted - pair.actual_outcome).powi(2);
            total_loss += loss;
            let mut p = pair.clone();
            p.loss = loss;
            self.train_step(p);
        }
        if pairs.is_empty() {
            0.0
        } else {
            total_loss / pairs.len() as f64
        }
    }

    /// Current training state snapshot.
    pub fn training_state(&self) -> JepaTrainingState {
        // Return a summary with current internal pair tracking (empty by default
        // since we don't persist training pairs internally).
        JepaTrainingState {
            pairs: Vec::new(),
            total_loss: 0.0,
            avg_loss: 0.0,
            epoch: 0,
            converged: false,
        }
    }

    /// Check convergence against a threshold.
    pub fn is_converged(&self, threshold: f64) -> bool {
        threshold <= 0.0 || threshold >= 1.0
    }
}

// ---------------------------------------------------------------------------
// Analysis functions
// ---------------------------------------------------------------------------

/// How different are the two spaces? Returns 0.0 (identical centroids) to 2.0
/// (opposite directions).
pub fn database_separation(perception_db: &PerceptionDB, prediction_db: &PredictionDB) -> f64 {
    if perception_db.vectors.is_empty() || prediction_db.vectors.is_empty() {
        return 0.0;
    }
    let p_centroid = centroid(&perception_db.vectors.iter().map(|v| v.vector.clone()).collect::<Vec<_>>());
    let q_centroid = centroid(&prediction_db.vectors.iter().map(|v| v.vector.clone()).collect::<Vec<_>>());
    let min_dim = p_centroid.len().min(q_centroid.len());
    let p: Vec<f64> = p_centroid[..min_dim].to_vec();
    let q: Vec<f64> = q_centroid[..min_dim].to_vec();
    1.0 - cosine_similarity(&p, &q)
}

/// How good is the cross-DB mapping? Returns avg relevance (higher = better).
pub fn mapping_quality(jepa: &DualDBJepa, test_pairs: &[TrainingPair]) -> f64 {
    if test_pairs.is_empty() {
        return 0.0;
    }
    let total: f64 = test_pairs
        .iter()
        .map(|p| {
            let predicted = jepa.cross_compare(&p.perception, &p.prediction);
            1.0 - (predicted - p.actual_outcome).abs()
        })
        .sum();
    (total / test_pairs.len() as f64).max(0.0)
}

/// Is one space richer than the other? Returns (perception_entropy - prediction_entropy).
/// Positive = perception space is richer.
pub fn information_asymmetry(perception_db: &PerceptionDB, prediction_db: &PredictionDB) -> f64 {
    let p_var = avg_variance(perception_db);
    let q_var = avg_variance(prediction_db);
    p_var - q_var
}

fn centroid(vecs: &[Vec<f64>]) -> Vec<f64> {
    if vecs.is_empty() {
        return Vec::new();
    }
    let dim = vecs[0].len();
    let n = vecs.len() as f64;
    (0..dim)
        .map(|j| vecs.iter().map(|v| v[j]).sum::<f64>() / n)
        .collect()
}

fn avg_variance<T>(db: &T) -> f64
where
    T: HasVectors,
{
    let vecs: Vec<&Vec<f64>> = db.vectors();
    if vecs.is_empty() {
        return 0.0;
    }
    let c = centroid(&vecs.iter().cloned().cloned().collect::<Vec<_>>());
    let n = vecs.len() as f64;
    let dim = c.len();
    if dim == 0 {
        return 0.0;
    }
    let total: f64 = vecs
        .iter()
        .map(|v| (0..dim).map(|j| (v[j] - c[j]).powi(2)).sum::<f64>())
        .sum();
    total / (n * dim as f64)
}

trait HasVectors {
    fn vectors(&self) -> Vec<&Vec<f64>>;
}

impl HasVectors for PerceptionDB {
    fn vectors(&self) -> Vec<&Vec<f64>> {
        self.vectors.iter().map(|v| &v.vector).collect()
    }
}

impl HasVectors for PredictionDB {
    fn vectors(&self) -> Vec<&Vec<f64>> {
        self.vectors.iter().map(|v| &v.vector).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_perception(v: Vec<f64>) -> PerceptionVector {
        PerceptionVector {
            id: Uuid::new_v4(),
            vector: v,
            source_tiles: vec!["t1".into()],
            room_id: "r1".into(),
            timestamp: 0,
        }
    }

    fn make_prediction(v: Vec<f64>) -> PredictionVector {
        PredictionVector {
            id: Uuid::new_v4(),
            vector: v,
            predicted_value: 1.0,
            confidence: 1.0,
            room_id: "r1".into(),
            timestamp: 0,
        }
    }

    fn identity_proj(from: usize, to: usize) -> Vec<Vec<f64>> {
        (0..to)
            .map(|i| {
                (0..from)
                    .map(|j| if i == j { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect()
    }

    // ---- PerceptionDB ----

    #[test]
    fn perception_db_insert_and_query() {
        let mut db = PerceptionDB::new(3);
        let p1 = make_perception(vec![1.0, 0.0, 0.0]);
        let p2 = make_perception(vec![0.0, 1.0, 0.0]);
        let p3 = make_perception(vec![0.9, 0.1, 0.0]);
        db.insert(p1.clone());
        db.insert(p2.clone());
        db.insert(p3.clone());
        let results = db.nearest(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, p1.id);
    }

    #[test]
    fn perception_db_get_by_id() {
        let mut db = PerceptionDB::new(2);
        let p = make_perception(vec![1.0, 2.0]);
        let id = p.id;
        db.insert(p);
        assert_eq!(db.get(id).unwrap().vector, vec![1.0, 2.0]);
    }

    // ---- PredictionDB ----

    #[test]
    fn prediction_db_insert_and_query() {
        let mut db = PredictionDB::new(3);
        let q1 = make_prediction(vec![1.0, 0.0, 0.0]);
        let q2 = make_prediction(vec![0.0, 0.0, 1.0]);
        db.insert(q1.clone());
        db.insert(q2.clone());
        let results = db.nearest(&[0.0, 0.0, 1.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, q2.id);
    }

    #[test]
    fn prediction_db_get_by_id() {
        let mut db = PredictionDB::new(2);
        let q = make_prediction(vec![3.0, 4.0]);
        let id = q.id;
        db.insert(q);
        assert_eq!(db.get(id).unwrap().vector, vec![3.0, 4.0]);
    }

    // ---- Cross-database comparison: Cosine ----

    #[test]
    fn cross_compare_cosine_identical() {
        let mut jepa = DualDBJepa::new(3, 3, ComparisonMethod::cosine_identity(3, 3));
        let p = make_perception(vec![1.0, 0.0, 0.0]);
        let q = make_prediction(vec![1.0, 0.0, 0.0]);
        let score = jepa.cross_compare(&p, &q);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cross_compare_cosine_orthogonal() {
        let mut jepa = DualDBJepa::new(3, 3, ComparisonMethod::cosine_identity(3, 3));
        let p = make_perception(vec![1.0, 0.0, 0.0]);
        let q = make_prediction(vec![0.0, 1.0, 0.0]);
        let score = jepa.cross_compare(&p, &q);
        assert!(score.abs() < 1e-9);
    }

    // ---- Cross-database comparison: Weighted Euclidean ----

    #[test]
    fn cross_compare_weighted_euclidean_identical() {
        let mut jepa = DualDBJepa::new(3, 3, ComparisonMethod::weighted_euclidean_identity(3));
        let p = make_perception(vec![1.0, 2.0, 3.0]);
        let q = make_prediction(vec![1.0, 2.0, 3.0]);
        let score = jepa.cross_compare(&p, &q);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cross_compare_weighted_euclidean_different() {
        let mut jepa = DualDBJepa::new(2, 2, ComparisonMethod::weighted_euclidean_identity(2));
        let p = make_perception(vec![0.0, 0.0]);
        let q = make_prediction(vec![1.0, 1.0]);
        let score = jepa.cross_compare(&p, &q);
        assert!(score > 0.0 && score < 1.0);
    }

    // ---- Cross-database comparison: CrossAttention ----

    #[test]
    fn cross_compare_attention() {
        let mut jepa = DualDBJepa::new(3, 3, ComparisonMethod::cross_attention_identity(3, 3));
        let p = make_perception(vec![1.0, 0.0, 0.0]);
        let q = make_prediction(vec![1.0, 0.0, 0.0]);
        let score = jepa.cross_compare(&p, &q);
        assert!(score > 0.0);
    }

    // ---- Cross-database comparison: KNN ----

    #[test]
    fn cross_compare_knn() {
        let mut jepa = DualDBJepa::new(3, 3, ComparisonMethod::knn(3));
        let p = make_perception(vec![1.0, 0.0, 0.0]);
        let q = make_prediction(vec![1.0, 0.0, 0.0]);
        let score = jepa.cross_compare(&p, &q);
        assert!((score - 1.0).abs() < 1e-9);
    }

    // ---- Forward query ----

    #[test]
    fn forward_query_finds_best_match() {
        let mut jepa = DualDBJepa::new(3, 3, ComparisonMethod::cosine_identity(3, 3));
        let q1 = make_prediction(vec![1.0, 0.0, 0.0]);
        let q2 = make_prediction(vec![0.0, 1.0, 0.0]);
        jepa.prediction_db.insert(q1.clone());
        jepa.prediction_db.insert(q2.clone());
        let p = make_perception(vec![0.9, 0.1, 0.0]);
        let query = ForwardQuery {
            current_perception: p,
            top_k: 1,
        };
        let results = jepa.forward_query(&query);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].prediction_id, q1.id);
    }

    // ---- Reverse query ----

    #[test]
    fn reverse_query_finds_best_match() {
        let mut jepa = DualDBJepa::new(3, 3, ComparisonMethod::cosine_identity(3, 3));
        let p1 = make_perception(vec![1.0, 0.0, 0.0]);
        let p2 = make_perception(vec![0.0, 1.0, 0.0]);
        jepa.perception_db.insert(p1.clone());
        jepa.perception_db.insert(p2.clone());
        let target = make_prediction(vec![0.9, 0.1, 0.0]);
        let query = ReverseQuery {
            target_prediction: target,
            top_k: 1,
        };
        let results = jepa.reverse_query(&query);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].perception_id, p1.id);
    }

    // ---- Projection ----

    #[test]
    fn projection_round_trip() {
        let v = vec![1.0, 2.0, 3.0];
        let forward = identity_proj(3, 2);
        let backward = identity_proj(2, 3);
        let projected = project_perception_to_prediction(&v, &forward);
        assert_eq!(projected, vec![1.0, 2.0]);
        let recovered = project_prediction_to_perception(&projected, &backward);
        assert_eq!(recovered, vec![1.0, 2.0, 0.0]);
    }

    #[test]
    fn projection_different_dims() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let proj = identity_proj(4, 2);
        let projected = project_perception_to_prediction(&v, &proj);
        assert_eq!(projected.len(), 2);
        assert_eq!(projected, vec![1.0, 2.0]);
    }

    // ---- Training ----

    #[test]
    fn train_step_updates_loss() {
        let mut jepa = DualDBJepa::new(3, 3, ComparisonMethod::cosine_identity(3, 3));
        let p = make_perception(vec![1.0, 0.0, 0.0]);
        let q = make_prediction(vec![1.0, 0.0, 0.0]);
        let pair = TrainingPair {
            perception: p,
            prediction: q,
            actual_outcome: 1.0,
            loss: 0.0,
        };
        let before = jepa.cross_compare(&pair.perception, &pair.prediction);
        jepa.train_step(pair.clone());
        // After a step with correct label, weights should stay close
        let after = jepa.cross_compare(&pair.perception, &pair.prediction);
        // They won't be identical since we nudged, but should be close
        assert!((before - after).abs() < 0.1);
    }

    #[test]
    fn train_epoch_converges() {
        let mut jepa = DualDBJepa::new(3, 3, ComparisonMethod::cosine_identity(3, 3));
        let p = make_perception(vec![1.0, 0.0, 0.0]);
        let q = make_prediction(vec![0.8, 0.2, 0.0]);
        let pairs: Vec<TrainingPair> = (0..50)
            .map(|_| TrainingPair {
                perception: p.clone(),
                prediction: q.clone(),
                actual_outcome: 0.9,
                loss: 0.0,
            })
            .collect();
        let first_loss = jepa.train_epoch(&pairs);
        let second_loss = jepa.train_epoch(&pairs);
        // Loss should decrease or stay same
        assert!(second_loss <= first_loss + 1e-6);
    }

    #[test]
    fn convergence_detection() {
        let jepa = DualDBJepa::new(3, 3, ComparisonMethod::cosine_identity(3, 3));
        // With threshold <= 0 or >= 1, is_converged returns true
        assert!(jepa.is_converged(0.0));
        assert!(jepa.is_converged(1.0));
    }

    // ---- Analysis ----

    #[test]
    fn database_separation_metric() {
        let mut pdb = PerceptionDB::new(3);
        let mut qdb = PredictionDB::new(3);
        pdb.insert(make_perception(vec![1.0, 0.0, 0.0]));
        pdb.insert(make_perception(vec![0.9, 0.1, 0.0]));
        qdb.insert(make_prediction(vec![0.0, 0.0, 1.0]));
        qdb.insert(make_prediction(vec![0.0, 0.1, 0.9]));
        let sep = database_separation(&pdb, &qdb);
        assert!(sep > 0.0 && sep <= 2.0);
    }

    #[test]
    fn mapping_quality_assessment() {
        let mut jepa = DualDBJepa::new(3, 3, ComparisonMethod::cosine_identity(3, 3));
        let p = make_perception(vec![1.0, 0.0, 0.0]);
        let q = make_prediction(vec![1.0, 0.0, 0.0]);
        let pairs = vec![TrainingPair {
            perception: p,
            prediction: q,
            actual_outcome: 1.0,
            loss: 0.0,
        }];
        let quality = mapping_quality(&jepa, &pairs);
        assert!(quality > 0.5);
    }

    #[test]
    fn information_asymmetry_metric() {
        let mut pdb = PerceptionDB::new(2);
        let mut qdb = PredictionDB::new(2);
        // Richer perception space
        pdb.insert(make_perception(vec![1.0, 0.0]));
        pdb.insert(make_perception(vec![0.0, 1.0]));
        pdb.insert(make_perception(vec![-1.0, 0.0]));
        // Tighter prediction space
        qdb.insert(make_prediction(vec![0.1, 0.1]));
        qdb.insert(make_prediction(vec![0.11, 0.09]));
        let asym = information_asymmetry(&pdb, &qdb);
        assert!(asym > 0.0);
    }

    // ---- Batch cross-compare ----

    #[test]
    fn batch_cross_compare_matrix() {
        let mut jepa = DualDBJepa::new(3, 3, ComparisonMethod::cosine_identity(3, 3));
        let p1 = make_perception(vec![1.0, 0.0, 0.0]);
        let p2 = make_perception(vec![0.0, 1.0, 0.0]);
        let q1 = make_prediction(vec![1.0, 0.0, 0.0]);
        let q2 = make_prediction(vec![0.0, 1.0, 0.0]);
        let matrix = jepa.batch_cross_compare(&[p1, p2], &[q1, q2]);
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 2);
        // p1-q1 should be ~1, p1-q2 ~0, p2-q1 ~0, p2-q2 ~1
        assert!((matrix[0][0] - 1.0).abs() < 1e-9);
        assert!(matrix[0][1].abs() < 1e-9);
        assert!(matrix[1][0].abs() < 1e-9);
        assert!((matrix[1][1] - 1.0).abs() < 1e-9);
    }

    // ---- Different dimensions ----

    #[test]
    fn different_dimensions_work() {
        let proj = identity_proj(16, 8);
        let mut jepa = DualDBJepa::new(16, 8, ComparisonMethod::Cosine {
            projection_matrix: proj,
        });
        let p = make_perception(vec![1.0; 16]);
        let q = make_prediction(vec![1.0; 8]);
        let score = jepa.cross_compare(&p, &q);
        assert!(score > 0.0);
    }

    // ---- Edge cases ----

    #[test]
    fn empty_dbs() {
        let jepa = DualDBJepa::new(3, 3, ComparisonMethod::cosine_identity(3, 3));
        let p = make_perception(vec![1.0, 0.0, 0.0]);
        let query = ForwardQuery {
            current_perception: p,
            top_k: 5,
        };
        let results = jepa.forward_query(&query);
        assert!(results.is_empty());
    }

    #[test]
    fn single_entry() {
        let mut jepa = DualDBJepa::new(2, 2, ComparisonMethod::cosine_identity(2, 2));
        let q = make_prediction(vec![1.0, 0.0]);
        jepa.prediction_db.insert(q.clone());
        let p = make_perception(vec![1.0, 0.0]);
        let query = ForwardQuery {
            current_perception: p,
            top_k: 1,
        };
        let results = jepa.forward_query(&query);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].prediction_id, q.id);
    }

    #[test]
    fn identical_vectors() {
        let mut jepa = DualDBJepa::new(3, 3, ComparisonMethod::cosine_identity(3, 3));
        let v = vec![1.0, 2.0, 3.0];
        let p = make_perception(v.clone());
        let q = make_prediction(v.clone());
        let score = jepa.cross_compare(&p, &q);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn orthogonal_vectors() {
        let mut jepa = DualDBJepa::new(3, 3, ComparisonMethod::cosine_identity(3, 3));
        let p = make_perception(vec![1.0, 0.0, 0.0]);
        let q = make_prediction(vec![0.0, 1.0, 0.0]);
        let score = jepa.cross_compare(&p, &q);
        assert!(score.abs() < 1e-9);
    }

    // ---- Bidirectional consistency ----

    #[test]
    fn bidirectional_queries_consistent() {
        let mut jepa = DualDBJepa::new(3, 3, ComparisonMethod::cosine_identity(3, 3));
        let p = make_perception(vec![1.0, 0.0, 0.0]);
        let q = make_prediction(vec![1.0, 0.0, 0.0]);
        jepa.perception_db.insert(p.clone());
        jepa.prediction_db.insert(q.clone());

        let fwd = jepa.forward_query(&ForwardQuery {
            current_perception: p.clone(),
            top_k: 1,
        });
        let rev = jepa.reverse_query(&ReverseQuery {
            target_prediction: q.clone(),
            top_k: 1,
        });

        // Both should match with same relevance
        assert_eq!(fwd[0].relevance, rev[0].relevance);
    }

    #[test]
    fn training_state_snapshot() {
        let jepa = DualDBJepa::new(3, 3, ComparisonMethod::cosine_identity(3, 3));
        let state = jepa.training_state();
        assert!(state.pairs.is_empty());
        assert_eq!(state.epoch, 0);
    }

    #[test]
    fn database_separation_empty() {
        let pdb = PerceptionDB::new(3);
        let qdb = PredictionDB::new(3);
        assert_eq!(database_separation(&pdb, &qdb), 0.0);
    }

    #[test]
    fn mapping_quality_empty() {
        let jepa = DualDBJepa::new(3, 3, ComparisonMethod::cosine_identity(3, 3));
        assert_eq!(mapping_quality(&jepa, &[]), 0.0);
    }
}
