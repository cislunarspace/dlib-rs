//! Clustering ported from dlib's `dlib/clustering.h` aggregate header:
//! `dlib/clustering/chinese_whispers.h`, `dlib/clustering/spectral_cluster.h`,
//! `dlib/clustering/bottom_up_cluster.h` and the k-means helpers of
//! `dlib/svm/kkmeans.h` (`pick_initial_centers`, `find_clusters_using_kmeans`,
//! `nearest_center`).
//!
//! Samples are column vectors (`Matrix<f64>` with `nc == 1`).

use std::cmp::Ordering;

use dlib_rs_core::matrix::{dot, svd::svd3, Matrix};
use dlib_rs_core::rand::Rand;

/// Similarity kernel used by the clustering routines; the `k` name mirrors
/// dlib's `kernel_type::operator()` (see e.g. `dlib/svm/kernel_abstract.h`).
///
/// Implemented for any closure `Fn(&Matrix<f64>, &Matrix<f64>) -> f64`, so the
/// kernels from [`crate::svm`] can be passed via a closure adapter
/// (`|a, b| radial_basis_kernel(a, b)`) without coupling this module to their
/// concrete types.
pub trait ClusterKernel {
    /// Kernel value between two samples.
    fn k(&self, a: &Matrix<f64>, b: &Matrix<f64>) -> f64;
}

impl<F> ClusterKernel for F
where
    F: Fn(&Matrix<f64>, &Matrix<f64>) -> f64,
{
    fn k(&self, a: &Matrix<f64>, b: &Matrix<f64>) -> f64 {
        self(a, b)
    }
}

// ----------------------------------------------------------------------------------------

/// Port of `dlib::sample_pair` from `dlib/graph_utils/sample_pair.h` (the
/// fields needed by the clustering routines).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplePair {
    index1: u64,
    index2: u64,
    distance: f64,
}

impl SamplePair {
    /// `sample_pair(idx1, idx2, distance)`.
    pub fn new(index1: u64, index2: u64, distance: f64) -> Self {
        SamplePair {
            index1,
            index2,
            distance,
        }
    }

    /// `sample_pair::index1()`.
    pub fn index1(&self) -> u64 {
        self.index1
    }

    /// `sample_pair::index2()`.
    pub fn index2(&self) -> u64 {
        self.index2
    }

    /// `sample_pair::distance()`.
    pub fn distance(&self) -> f64 {
        self.distance
    }
}

/// Ordered edge: same shape as `dlib::ordered_sample_pair`, always stored with
/// `index1 <= index2` semantics once converted from [`SamplePair`].
#[derive(Clone, Copy, Debug)]
struct OrderedSamplePair {
    index1: u64,
    index2: u64,
    distance: f64,
}

/// Port of `convert_unordered_to_ordered` from
/// `dlib/graph_utils/edge_list_graphs.h`.
fn convert_unordered_to_ordered(edges: &[SamplePair]) -> Vec<OrderedSamplePair> {
    let mut out = Vec::with_capacity(edges.len() * 2);
    for e in edges {
        out.push(OrderedSamplePair {
            index1: e.index1(),
            index2: e.index2(),
            distance: e.distance(),
        });
        if e.index1() != e.index2() {
            out.push(OrderedSamplePair {
                index1: e.index2(),
                index2: e.index1(),
                distance: e.distance(),
            });
        }
    }
    out
}

/// Port of `find_neighbor_ranges` from
/// `dlib/graph_utils/edge_list_graphs.h`: `neighbors[i]` is the
/// `[start, end)` range of `edges` holding node `i`'s edges.
fn find_neighbor_ranges(edges: &[OrderedSamplePair]) -> Vec<(usize, usize)> {
    let num_nodes = edges
        .iter()
        .map(|e| e.index2.max(e.index1))
        .max()
        .unwrap_or(0) as usize
        + 1;
    let mut neighbors = vec![(0usize, 0usize); num_nodes];
    let mut cur_node = 0usize;
    let mut start_idx = 0usize;
    for (i, e) in edges.iter().enumerate() {
        if e.index1 as usize != cur_node {
            neighbors[cur_node] = (start_idx, i);
            start_idx = i;
            cur_node = e.index1 as usize;
        }
    }
    if !neighbors.is_empty() {
        neighbors[cur_node] = (start_idx, edges.len());
    }
    neighbors
}

/// Port of the edge-based `chinese_whispers(edges, labels, num_iterations, rnd)`
/// from `dlib/clustering/chinese_whispers.h` (the `std::vector<sample_pair>`
/// overload, which converts to ordered edges and sorts by
/// `order_by_index` first). Returns `(num_clusters, labels)`.
pub fn chinese_whispers_edges(
    edges: &[SamplePair],
    num_iterations: u32,
    rnd: &mut Rand,
) -> (usize, Vec<u64>) {
    let mut oedges = convert_unordered_to_ordered(edges);
    // std::sort with order_by_index (lexicographic on index1, index2).
    oedges.sort_by_key(|e| (e.index1, e.index2));

    if oedges.is_empty() {
        return (0, Vec::new());
    }

    let neighbors = find_neighbor_ranges(&oedges);

    // Initialize the labels, each node gets a different label.
    let mut labels: Vec<u64> = (0..neighbors.len() as u64).collect();

    // The C++ counts labels in a std::map<unsigned long, double> and scans it
    // in ascending key order, breaking score ties in favor of the smallest
    // label.  Replicated with a BTreeMap.
    let mut labels_to_counts = std::collections::BTreeMap::new();

    for _iter in 0..neighbors.len() * num_iterations as usize {
        // Pick a random node.
        let idx = (rnd.get_random_64bit_number() % neighbors.len() as u64) as usize;

        // Count how many times each label happens amongst our neighbors.
        labels_to_counts.clear();
        for i in neighbors[idx].0..neighbors[idx].1 {
            *labels_to_counts
                .entry(labels[oedges[i].index2 as usize])
                .or_insert(0.0) += oedges[i].distance;
        }

        // find the most common label
        let mut best_score = f64::NEG_INFINITY;
        let mut best_label = labels[idx];
        for (label, score) in &labels_to_counts {
            if *score > best_score {
                best_score = *score;
                best_label = *label;
            }
        }

        labels[idx] = best_label;
    }

    // Remap the labels into a contiguous range.
    let mut label_remap: std::collections::BTreeMap<u64, u64> = std::collections::BTreeMap::new();
    for l in &labels {
        let next_id = label_remap.len() as u64;
        label_remap.entry(*l).or_insert(next_id);
    }
    for l in labels.iter_mut() {
        *l = label_remap[l];
    }

    (label_remap.len(), labels)
}

/// Kernel-based convenience overload of `chinese_whispers`: connects every
/// pair of samples with an edge whose weight (dlib's `distance()` field is a
/// similarity weight here) is the kernel value, then runs
/// [`chinese_whispers_edges`] with `num_iterations = 100` and a
/// default-constructed `dlib::rand` (exactly how the default overload of
/// `dlib::chinese_whispers` seeds itself).
///
/// Ported from `dlib/clustering/chinese_whispers.h`.
pub fn chinese_whispers(
    samples: &[Matrix<f64>],
    kernel: &impl ClusterKernel,
    num_iterations: u32,
) -> Vec<u64> {
    let mut edges = Vec::with_capacity(samples.len() * (samples.len() - 1) / 2);
    for i in 0..samples.len() {
        for j in (i + 1)..samples.len() {
            edges.push(SamplePair::new(
                i as u64,
                j as u64,
                kernel.k(&samples[i], &samples[j]),
            ));
        }
    }
    let mut rnd = Rand::new();
    chinese_whispers_edges(&edges, num_iterations, &mut rnd).1
}

// ----------------------------------------------------------------------------------------

/// Port of `pick_initial_centers` from `dlib/svm/kkmeans.h` (the kernel
/// overload with `percentile = 0.01` default). A non-randomized kmeans++
/// seeding: the first center is `samples[0]`, subsequent centers are the
/// sample at the given percentile of distance-to-nearest-center.
pub fn pick_initial_centers(
    num_centers: usize,
    samples: &[Matrix<f64>],
    kernel: &impl ClusterKernel,
    percentile: f64,
) -> Vec<Matrix<f64>> {
    assert!(
        num_centers > 1 && (0.0..1.0).contains(&percentile) && samples.len() > 1,
        "pick_initial_centers: invalid arguments"
    );

    #[derive(Clone, Copy)]
    struct ScoreData {
        idx: usize,
        dist: f64,
    }

    let mut scores = vec![
        ScoreData {
            idx: 0,
            dist: f64::INFINITY,
        };
        samples.len()
    ];
    let mut centers = Vec::new();

    // pick the first sample as one of the centers
    centers.push(samples[0].clone());

    let best_idx =
        (samples.len() as f64 - samples.len() as f64 * percentile - 1.0).max(0.0) as usize;

    // pick the next center
    for i in 0..num_centers - 1 {
        // Store the distance from each sample to its closest center in scores.
        let k_cc = kernel.k(&centers[i], &centers[i]);
        for (s, sample) in samples.iter().enumerate() {
            // compute the distance between this sample and the current center
            let dist = k_cc + kernel.k(sample, sample) - 2.0 * kernel.k(sample, &centers[i]);

            if dist < scores[s].dist {
                scores[s].dist = dist;
                scores[s].idx = s;
            }
        }

        // now find the winning center and add it to centers.  It is the one
        // that is far away from all the other centers.
        let mut scores_sorted = scores.clone();
        scores_sorted.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        centers.push(samples[scores_sorted[best_idx].idx].clone());
    }

    centers
}

/// Port of `find_clusters_using_kmeans` from `dlib/svm/kkmeans.h` (centers
/// overload): Lloyd's algorithm refining `centers` in place, `max_iter = 1000`
/// in dlib. Returns after `centers` stops changing or `max_iter` iterations.
pub fn find_clusters_using_kmeans_centers(
    samples: &[Matrix<f64>],
    centers: &mut [Matrix<f64>],
    max_iter: usize,
) {
    assert!(
        !samples.is_empty() && !centers.is_empty(),
        "find_clusters_using_kmeans: invalid arguments"
    );

    let zero = Matrix::zeros(centers[0].nr(), centers[0].nc());

    // tells which center a sample belongs to
    let mut assignments = vec![samples.len(); samples.len()];

    let mut iter = 0usize;
    let mut centers_changed = true;
    while centers_changed && iter < max_iter {
        iter += 1;
        centers_changed = false;
        let mut center_element_count = vec![0u64; centers.len()];

        // loop over each sample and see which center it is closest to
        for (i, sample) in samples.iter().enumerate() {
            // find the best center for sample[i]
            let mut best_dist = f64::MAX;
            let mut best_center = 0usize;
            for (j, center) in centers.iter().enumerate() {
                let diff = center.clone() - sample.clone();
                let dist = dot(&diff, &diff).sqrt();
                if dist < best_dist {
                    best_dist = dist;
                    best_center = j;
                }
            }

            if assignments[i] != best_center {
                centers_changed = true;
                assignments[i] = best_center;
            }

            center_element_count[best_center] += 1;
        }

        // now update all the centers
        for c in centers.iter_mut() {
            *c = zero.clone();
        }
        for (i, sample) in samples.iter().enumerate() {
            centers[assignments[i]] += sample.clone();
        }
        for (i, c) in centers.iter_mut().enumerate() {
            if center_element_count[i] != 0 {
                *c = c.clone() / center_element_count[i] as f64;
            }
        }
    }
}

/// Port of `nearest_center` from `dlib/svm/kkmeans.h`.
pub fn nearest_center(centers: &[Matrix<f64>], sample: &Matrix<f64>) -> u64 {
    assert!(
        !centers.is_empty(),
        "nearest_center: centers must not be empty"
    );

    let mut best_dist = {
        let diff = centers[0].clone() - sample.clone();
        dot(&diff, &diff)
    };
    let mut best_idx = 0usize;
    for (i, center) in centers.iter().enumerate().skip(1) {
        let diff = center.clone() - sample.clone();
        let dist = dot(&diff, &diff);
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }
    best_idx as u64
}

/// End-to-end k-means clustering returning labels: deterministic kmeans++
/// initial centers ([`pick_initial_centers`] with a linear kernel, exactly as
/// dlib's `pick_initial_centers` overload without a kernel does), Lloyd
/// refinement and final assignment via [`nearest_center`]. No randomness is
/// involved, so results are reproducible. Composed from `dlib/svm/kkmeans.h`.
pub fn find_clusters_using_kmeans(
    samples: &[Matrix<f64>],
    num_clusters: usize,
    max_iter: usize,
) -> Vec<u64> {
    assert!(num_clusters > 0 && !samples.is_empty());
    let linear = |a: &Matrix<f64>, b: &Matrix<f64>| dot(a, b);
    let mut centers = pick_initial_centers(num_clusters, samples, &linear, 0.01);
    find_clusters_using_kmeans_centers(samples, &mut centers, max_iter);
    samples
        .iter()
        .map(|s| nearest_center(&centers, s))
        .collect()
}

// ----------------------------------------------------------------------------------------

/// Port of `buc_impl::merge_sets` from `dlib/clustering/bottom_up_cluster.h`.
fn merge_dists(dists: &mut Matrix<f64>, dest: usize, src: usize) {
    for r in 0..dists.nr() {
        let d = dists[(r, dest)].max(dists[(r, src)]);
        dists[(dest, r)] = d;
        dists[(r, dest)] = d;
    }
}

/// Minimal port of `dlib::disjoint_subsets`
/// (`dlib/disjoint_subsets/disjoint_subsets.h`): union-find with union by rank
/// and path compression; `merge_sets` returns the new root.
#[derive(Clone, Debug, Default)]
struct DisjointSubsets {
    parent: Vec<usize>,
    rank: Vec<u32>,
}

impl DisjointSubsets {
    fn set_size(&mut self, n: usize) {
        self.parent = (0..n).collect();
        self.rank = vec![0; n];
    }

    fn find_set(&mut self, item: usize) -> usize {
        // find root of item
        let mut x = item;
        while self.parent[x] != x {
            x = self.parent[x];
        }
        // compress the path
        let mut cur = item;
        while self.parent[cur] != cur {
            let next = self.parent[cur];
            self.parent[cur] = x;
            cur = next;
        }
        x
    }

    fn merge_sets(&mut self, a: usize, b: usize) -> usize {
        if self.rank[a] > self.rank[b] {
            self.parent[b] = a;
            a
        } else {
            self.parent[a] = b;
            if self.rank[a] == self.rank[b] {
                self.rank[b] += 1;
            }
            b
        }
    }
}

/// Priority-queue element for `bottom_up_cluster`; ordered so the *smallest*
/// distance pops first, matching the C++ `priority_queue` with
/// `compare_dist` (`a.distance() > b.distance()`).
#[derive(Clone, Copy)]
struct QueuePair {
    index1: usize,
    index2: usize,
    distance: f64,
}

impl PartialEq for QueuePair {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for QueuePair {}
impl PartialOrd for QueuePair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for QueuePair {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; reverse so the smallest distance is greatest.
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Port of `bottom_up_cluster` from `dlib/clustering/bottom_up_cluster.h`.
///
/// `dists` is a symmetric distance matrix; merging continues until only
/// `min_num_clusters` remain or the best merge distance exceeds `max_dist`.
/// Returns `(num_clusters, labels)`.
pub fn bottom_up_cluster(
    dists: &Matrix<f64>,
    min_num_clusters: usize,
    max_dist: f64,
) -> (usize, Vec<u64>) {
    assert!(
        dists.nr() == dists.nc() && min_num_clusters > 0,
        "bottom_up_cluster: invalid arguments"
    );

    let mut dists = dists.clone();
    let n = dists.nr();
    let mut labels = vec![0u64; n];
    let mut sets = DisjointSubsets::default();
    sets.set_size(n);
    if labels.is_empty() {
        return (0, labels);
    }

    // push all the edges in the graph into a priority queue so the best edges
    // to merge come first.
    let mut que = std::collections::BinaryHeap::with_capacity(n * (n - 1) / 2);
    for r in 0..n {
        for c in (r + 1)..n {
            que.push(QueuePair {
                index1: r,
                index2: c,
                distance: dists[(r, c)],
            });
        }
    }

    // Now start merging nodes.
    for _iter in min_num_clusters..sets.parent.len() {
        // find the next best thing to merge.
        let mut top = que.pop().expect("bottom_up_cluster: queue exhausted");
        let mut best_dist = top.distance;
        let mut a = sets.find_set(top.index1);
        let mut b = sets.find_set(top.index2);
        // we have been merging and modifying the distances, so make sure this
        // distance is still valid and these guys haven't been merged already.
        while a == b || best_dist < dists[(a, b)] {
            // Haven't merged it yet, so put it back in with updated distance
            // for reconsideration later.
            if a != b {
                que.push(QueuePair {
                    index1: a,
                    index2: b,
                    distance: dists[(a, b)],
                });
            }

            top = que.pop().expect("bottom_up_cluster: queue exhausted");
            best_dist = top.distance;
            a = sets.find_set(top.index1);
            b = sets.find_set(top.index2);
        }

        // now merge these sets if the best distance is small enough
        if best_dist > max_dist {
            break;
        }
        let news = sets.merge_sets(a, b);
        let olds = if news == a { b } else { a };
        merge_dists(&mut dists, news, olds);
    }

    // figure out which cluster each element is in.  Also make sure the labels
    // are contiguous.
    let mut relabel: std::collections::BTreeMap<usize, u64> = std::collections::BTreeMap::new();
    for (r, l) in labels.iter_mut().enumerate() {
        let s = sets.find_set(r);
        let next = relabel.len() as u64;
        relabel.entry(s).or_insert(next);
        *l = relabel[&s];
    }

    (relabel.len(), labels)
}

// ----------------------------------------------------------------------------------------

/// Port of `rsort_columns(v, w)` from `dlib/matrix/matrix_utilities.h`:
/// permutes the columns of `m` and entries of `v` so the `v` values are in
/// descending order (stable with respect to the original column order, which
/// matches dlib's pair sort on non-tied values).
fn rsort_columns(m: &mut Matrix<f64>, v: &mut Matrix<f64>) {
    assert!(v.nc() == 1 && v.nr() == m.nc());
    let nc = m.nc();
    let mut order: Vec<usize> = (0..nc).collect();
    order.sort_by(|&a, &b| v[(b, 0)].partial_cmp(&v[(a, 0)]).unwrap_or(Ordering::Equal));
    let old_m = m.clone();
    let old_v = v.clone();
    for (i, &src) in order.iter().enumerate() {
        for r in 0..m.nr() {
            m[(r, i)] = old_m[(r, src)];
        }
        v[(i, 0)] = old_v[(src, 0)];
    }
}

/// Port of `spectral_cluster` from `dlib/clustering/spectral_cluster.h`.
///
/// Builds the kernel similarity matrix, normalizes it as
/// `diagm(1/sqrt(D)) * K * diagm(1/sqrt(D))`, takes the `num_clusters`
/// eigenvectors with the largest singular values, normalizes each row, and
/// runs k-means on the resulting spectral vectors.
pub fn spectral_cluster(
    kernel: &impl ClusterKernel,
    samples: &[Matrix<f64>],
    num_clusters: usize,
) -> Vec<u64> {
    assert!(
        num_clusters > 0,
        "spectral_cluster: num_clusters can't be 0."
    );

    if num_clusters == 1 {
        // nothing to do, just assign everything to the 0 cluster.
        return vec![0; samples.len()];
    }

    // compute the similarity matrix.
    let n = samples.len();
    let mut k = Matrix::zeros(n, n);
    for r in 0..n {
        for c in (r + 1)..n {
            let val = kernel.k(&samples[r], &samples[c]);
            k[(r, c)] = val;
            k[(c, r)] = val;
        }
    }
    for r in 0..n {
        k[(r, r)] = 0.0;
    }

    // D = sqrt(reciprocal(row_sums(K))); K = diagm(D)*K*diagm(D)
    let mut d = Matrix::zeros(n, 1);
    for r in 0..n {
        let mut s = 0.0;
        for c in 0..n {
            s += k[(r, c)];
        }
        d[(r, 0)] = (1.0 / s).sqrt();
    }
    for r in 0..n {
        for c in 0..n {
            let val = d[(r, 0)] * k[(r, c)] * d[(c, 0)];
            k[(r, c)] = val;
        }
    }

    // Use the normal SVD routine (the C++ switches to svd_fast only when
    // K.nr() >= 1000).
    let mut u = Matrix::new();
    let mut w = Matrix::new();
    let mut v = Matrix::new();
    svd3(&k, &mut u, &mut w, &mut v);

    // Pick out the eigenvectors associated with the largest eigenvalues.
    rsort_columns(&mut v, &mut w);
    let v = v.subm(0, 0, v.nr(), num_clusters);

    // Now build the normalized spectral vectors, one for each input vector.
    let mut spec_samps = Vec::with_capacity(v.nr());
    for r in 0..v.nr() {
        let mut s = v.rowm(r).transpose();
        let len = dot(&s, &s).sqrt();
        if len != 0.0 {
            for i in 0..s.nr() {
                s[(i, 0)] /= len;
            }
        }
        spec_samps.push(s);
    }

    // Finally do the K-means clustering
    let linear = |a: &Matrix<f64>, b: &Matrix<f64>| dot(a, b);
    let mut centers = pick_initial_centers(num_clusters, &spec_samps, &linear, 0.01);
    find_clusters_using_kmeans_centers(&spec_samps, &mut centers, 1000);

    // And then compute the cluster assignments based on the output of K-means.
    spec_samps
        .iter()
        .map(|s| nearest_center(&centers, s))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use dlib_rs_core::matrix::dot;

    fn col(x: f64, y: f64) -> Matrix<f64> {
        Matrix::from_row_vec(2, 1, &[x, y])
    }

    fn three_blobs() -> Vec<Matrix<f64>> {
        let mut s = Vec::new();
        for &(bx, by) in &[(0.0, 0.0), (20.0, 0.0), (0.0, 20.0)] {
            for i in 0..5 {
                let d = i as f64;
                s.push(col(bx + 0.1 * d, by + 0.1 * d));
            }
        }
        s
    }

    fn two_blobs() -> Vec<Matrix<f64>> {
        let mut s = Vec::new();
        for &(bx, by) in &[(0.0, 0.0), (30.0, 30.0)] {
            for i in 0..4 {
                let d = i as f64;
                s.push(col(bx + 0.2 * d, by + 0.2 * d));
            }
        }
        s
    }

    fn rbf(gamma: f64) -> impl ClusterKernel {
        move |a: &Matrix<f64>, b: &Matrix<f64>| {
            let diff = a.clone() - b.clone();
            (-gamma * dot(&diff, &diff)).exp()
        }
    }

    fn assert_grouping(labels: &[u64], groups: &[&[usize]]) {
        let ids: Vec<u64> = groups.iter().map(|g| labels[g[0]]).collect();
        // distinct cluster ids across groups
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                assert_ne!(ids[i], ids[j], "labels {:?} groups {:?}", labels, groups);
            }
        }
        for (g, &id) in groups.iter().zip(&ids) {
            for &idx in *g {
                assert_eq!(labels[idx], id, "labels {:?} groups {:?}", labels, groups);
            }
        }
    }

    #[test]
    fn test_chinese_whispers_three_blobs() {
        let samples = three_blobs();
        let labels1 = chinese_whispers(&samples, &rbf(1.0), 100);
        let labels2 = chinese_whispers(&samples, &rbf(1.0), 100);
        assert_eq!(
            labels1, labels2,
            "must be deterministic (fixed default seed)"
        );
        assert_grouping(
            &labels1,
            &[&[0, 1, 2, 3, 4], &[5, 6, 7, 8, 9], &[10, 11, 12, 13, 14]],
        );
        assert_eq!(*labels1.iter().max().unwrap(), 2);
    }

    #[test]
    fn test_kmeans_two_blobs() {
        let samples = two_blobs();
        let labels1 = find_clusters_using_kmeans(&samples, 2, 1000);
        let labels2 = find_clusters_using_kmeans(&samples, 2, 1000);
        assert_eq!(labels1, labels2);
        // Pinned expected result from construction: pick_initial_centers picks
        // sample 0 (blob A) then the sample farthest away (blob B), so blob A
        // gets center 0 and blob B center 1.
        assert_grouping(&labels1, &[&[0, 1, 2, 3], &[4, 5, 6, 7]]);
    }

    #[test]
    fn test_spectral_two_blobs() {
        let samples = two_blobs();
        let labels = spectral_cluster(&rbf(0.1), &samples, 2);
        assert_grouping(&labels, &[&[0, 1, 2, 3], &[4, 5, 6, 7]]);
    }

    #[test]
    fn test_bottom_up_cluster() {
        let samples = two_blobs();
        let n = samples.len();
        let mut dists = Matrix::zeros(n, n);
        for r in 0..n {
            for c in 0..n {
                let diff = samples[r].clone() - samples[c].clone();
                dists[(r, c)] = dot(&diff, &diff).sqrt();
            }
        }
        let (num, labels) = bottom_up_cluster(&dists, 2, f64::INFINITY);
        assert_eq!(num, 2);
        assert_grouping(&labels, &[&[0, 1, 2, 3], &[4, 5, 6, 7]]);
        // max_dist cut: with a threshold between the blobs the same two
        // clusters appear; with a tiny threshold nothing merges.
        let (num2, _) = bottom_up_cluster(&dists, 1, 0.05);
        assert_eq!(num2, 8);
    }

    #[test]
    fn test_chinese_whispers_empty_edges() {
        let mut rnd = Rand::new();
        assert_eq!(
            chinese_whispers_edges(&[], 100, &mut rnd),
            (0, Vec::<u64>::new())
        );
    }
}
