//! Statistics ported from dlib's `dlib/statistics/statistics.h`
//! (`running_stats`, `running_scalar_covariance`, the free functions
//! `mean_sign_agreement`, `correlation`, `covariance`, `r_squared`,
//! `mean_squared_error`) and `dlib/statistics/random_subset_selector.h`
//! (`random_subset_selector`).

use dlib_rs_core::rand::Rand;

/// Port of `dlib::running_stats<double>` from `dlib/statistics/statistics.h`.
///
/// Accumulates raw power sums `sum`, `sum^2`, `sum^3`, `sum^4` and derives
/// mean/variance/skewness/excess-kurtosis from them using dlib's exact
/// formulas (including the small-sample bias corrections).
#[derive(Clone, Debug)]
pub struct RunningStats {
    sum: f64,
    sum_sqr: f64,
    sum_cub: f64,
    sum_four: f64,
    n: f64,
    min_value: f64,
    max_value: f64,
}

impl Default for RunningStats {
    fn default() -> Self {
        Self::new()
    }
}

impl RunningStats {
    /// `running_stats::running_stats()` — same as `clear()`.
    pub fn new() -> Self {
        Self::clear()
    }

    /// `running_stats::clear()`.
    pub fn clear() -> Self {
        RunningStats {
            sum: 0.0,
            sum_sqr: 0.0,
            sum_cub: 0.0,
            sum_four: 0.0,
            n: 0.0,
            min_value: f64::INFINITY,
            max_value: f64::NEG_INFINITY,
        }
    }

    /// `running_stats::add(val)`.
    pub fn add(&mut self, val: f64) {
        self.sum += val;
        self.sum_sqr += val * val;
        self.sum_cub += val * val * val;
        self.sum_four += val * val * val * val;

        if val < self.min_value {
            self.min_value = val;
        }
        if val > self.max_value {
            self.max_value = val;
        }

        self.n += 1.0;
    }

    /// `running_stats::current_n()`.
    pub fn current_n(&self) -> f64 {
        self.n
    }

    /// `running_stats::sum()`.
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// `running_stats::mean()`.
    pub fn mean(&self) -> f64 {
        if self.n != 0.0 {
            self.sum / self.n
        } else {
            0.0
        }
    }

    /// `running_stats::max()` (requires at least one `add`).
    pub fn max(&self) -> f64 {
        assert!(self.n > 0.0, "running_stats::max: no samples yet");
        self.max_value
    }

    /// `running_stats::min()` (requires at least one `add`).
    pub fn min(&self) -> f64 {
        assert!(self.n > 0.0, "running_stats::min: no samples yet");
        self.min_value
    }

    /// `running_stats::variance()` (n-1 denominator, clamped at zero).
    pub fn variance(&self) -> f64 {
        assert!(self.n > 1.0, "running_stats::variance: need n > 1");
        let mut temp = 1.0 / (self.n - 1.0);
        temp *= self.sum_sqr - self.sum * self.sum / self.n;
        // make sure the variance is never negative.  This might
        // happen due to numerical errors.
        if temp >= 0.0 {
            temp
        } else {
            0.0
        }
    }

    /// `running_stats::stddev()`.
    pub fn stddev(&self) -> f64 {
        assert!(self.n > 1.0, "running_stats::stddev: need n > 1");
        self.variance().sqrt()
    }

    /// `running_stats::skewness()`.
    pub fn skewness(&self) -> f64 {
        assert!(self.n > 2.0, "running_stats::skewness: need n > 2");
        let temp = 1.0 / self.n;
        let temp1 = (self.n * (self.n - 1.0)).sqrt() / (self.n - 2.0);
        temp1
            * temp
            * (self.sum_cub - 3.0 * self.sum_sqr * self.sum * temp
                + 2.0 * self.sum * self.sum * self.sum * temp * temp)
            / ((temp * (self.sum_sqr - self.sum * self.sum * temp))
                .powi(3)
                .sqrt())
    }

    /// `running_stats::ex_kurtosis()`.
    pub fn ex_kurtosis(&self) -> f64 {
        assert!(self.n > 3.0, "running_stats::ex_kurtosis: need n > 3");
        let temp = 1.0 / self.n;
        let m4 = temp
            * (self.sum_four - 4.0 * self.sum_cub * self.sum * temp
                + 6.0 * self.sum_sqr * self.sum * self.sum * temp * temp
                - 3.0 * self.sum * self.sum * self.sum * self.sum * temp * temp * temp);
        let m2 = temp * (self.sum_sqr - self.sum * self.sum * temp);
        (self.n - 1.0) * ((self.n + 1.0) * m4 / (m2 * m2) - 3.0 * (self.n - 1.0))
            / ((self.n - 2.0) * (self.n - 3.0))
    }

    /// `running_stats::scale(val)`: `(val - mean())/stddev()`.
    pub fn scale(&self, val: f64) -> f64 {
        assert!(self.n > 1.0, "running_stats::scale: need n > 1");
        (val - self.mean()) / self.variance().sqrt()
    }

    /// `running_stats::operator+`: merge two accumulators.
    pub fn merge(&self, rhs: &RunningStats) -> RunningStats {
        let mut temp = self.clone();
        temp.sum += rhs.sum;
        temp.sum_sqr += rhs.sum_sqr;
        temp.sum_cub += rhs.sum_cub;
        temp.sum_four += rhs.sum_four;
        temp.n += rhs.n;
        temp.min_value = rhs.min_value.min(self.min_value);
        temp.max_value = rhs.max_value.max(self.max_value);
        temp
    }
}

impl std::ops::Add for RunningStats {
    type Output = RunningStats;
    fn add(self, rhs: RunningStats) -> RunningStats {
        self.merge(&rhs)
    }
}

// ----------------------------------------------------------------------------------------

/// Port of `dlib::running_scalar_covariance<double>` from
/// `dlib/statistics/statistics.h`.
#[derive(Clone, Debug)]
pub struct RunningScalarCovariance {
    sum_xy: f64,
    sum_x: f64,
    sum_y: f64,
    sum_xx: f64,
    sum_yy: f64,
    n: f64,
}

impl Default for RunningScalarCovariance {
    fn default() -> Self {
        Self::new()
    }
}

impl RunningScalarCovariance {
    /// `running_scalar_covariance::running_scalar_covariance()`.
    pub fn new() -> Self {
        Self::clear()
    }

    /// `running_scalar_covariance::clear()`.
    pub fn clear() -> Self {
        RunningScalarCovariance {
            sum_xy: 0.0,
            sum_x: 0.0,
            sum_y: 0.0,
            sum_xx: 0.0,
            sum_yy: 0.0,
            n: 0.0,
        }
    }

    /// `running_scalar_covariance::add(x, y)`.
    pub fn add(&mut self, x: f64, y: f64) {
        self.sum_xy += x * y;
        self.sum_xx += x * x;
        self.sum_yy += y * y;
        self.sum_x += x;
        self.sum_y += y;
        self.n += 1.0;
    }

    /// `running_scalar_covariance::current_n()`.
    pub fn current_n(&self) -> f64 {
        self.n
    }

    /// `running_scalar_covariance::mean_x()`.
    pub fn mean_x(&self) -> f64 {
        if self.n != 0.0 {
            self.sum_x / self.n
        } else {
            0.0
        }
    }

    /// `running_scalar_covariance::mean_y()`.
    pub fn mean_y(&self) -> f64 {
        if self.n != 0.0 {
            self.sum_y / self.n
        } else {
            0.0
        }
    }

    /// `running_scalar_covariance::covariance()`.
    pub fn covariance(&self) -> f64 {
        assert!(
            self.n > 1.0,
            "running_scalar_covariance::covariance: need n > 1"
        );
        1.0 / (self.n - 1.0) * (self.sum_xy - self.sum_y * self.sum_x / self.n)
    }

    /// `running_scalar_covariance::correlation()`.
    pub fn correlation(&self) -> f64 {
        assert!(
            self.n > 1.0,
            "running_scalar_covariance::correlation: need n > 1"
        );
        self.covariance() / (self.variance_x() * self.variance_y()).sqrt()
    }

    /// `running_scalar_covariance::variance_x()` (clamped at zero).
    pub fn variance_x(&self) -> f64 {
        assert!(
            self.n > 1.0,
            "running_scalar_covariance::variance_x: need n > 1"
        );
        let temp = 1.0 / (self.n - 1.0) * (self.sum_xx - self.sum_x * self.sum_x / self.n);
        // make sure the variance is never negative.
        if temp >= 0.0 {
            temp
        } else {
            0.0
        }
    }

    /// `running_scalar_covariance::variance_y()` (clamped at zero).
    pub fn variance_y(&self) -> f64 {
        assert!(
            self.n > 1.0,
            "running_scalar_covariance::variance_y: need n > 1"
        );
        let temp = 1.0 / (self.n - 1.0) * (self.sum_yy - self.sum_y * self.sum_y / self.n);
        if temp >= 0.0 {
            temp
        } else {
            0.0
        }
    }

    /// `running_scalar_covariance::stddev_x()`.
    pub fn stddev_x(&self) -> f64 {
        assert!(
            self.n > 1.0,
            "running_scalar_covariance::stddev_x: need n > 1"
        );
        self.variance_x().sqrt()
    }

    /// `running_scalar_covariance::stddev_y()`.
    pub fn stddev_y(&self) -> f64 {
        assert!(
            self.n > 1.0,
            "running_scalar_covariance::stddev_y: need n > 1"
        );
        self.variance_y().sqrt()
    }

    /// `running_scalar_covariance::operator+`: merge two accumulators.
    pub fn merge(&self, rhs: &RunningScalarCovariance) -> RunningScalarCovariance {
        let mut temp = rhs.clone();
        temp.sum_xy += self.sum_xy;
        temp.sum_x += self.sum_x;
        temp.sum_y += self.sum_y;
        temp.sum_xx += self.sum_xx;
        temp.sum_yy += self.sum_yy;
        temp.n += self.n;
        temp
    }
}

impl std::ops::Add for RunningScalarCovariance {
    type Output = RunningScalarCovariance;
    fn add(self, rhs: RunningScalarCovariance) -> RunningScalarCovariance {
        self.merge(&rhs)
    }
}

// ----------------------------------------------------------------------------------------

/// Port of `mean_sign_agreement(a, b)` from `dlib/statistics/statistics.h`:
/// the fraction of elements where `a[i]` and `b[i]` share a sign (zero counts
/// as non-negative).
pub fn mean_sign_agreement(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "mean_sign_agreement: a and b must be the same length"
    );
    let mut temp = 0.0;
    for i in 0..a.len() {
        if (a[i] >= 0.0 && b[i] >= 0.0) || (a[i] < 0.0 && b[i] < 0.0) {
            temp += 1.0;
        }
    }
    temp / a.len() as f64
}

/// Port of `correlation(a, b)` from `dlib/statistics/statistics.h`
/// (Pearson correlation via `running_scalar_covariance`).
pub fn correlation(a: &[f64], b: &[f64]) -> f64 {
    assert!(
        a.len() == b.len() && a.len() > 1,
        "correlation: a and b must be the same length and have more than one element"
    );
    let mut rs = RunningScalarCovariance::new();
    for i in 0..a.len() {
        rs.add(a[i], b[i]);
    }
    rs.correlation()
}

/// Port of `covariance(a, b)` from `dlib/statistics/statistics.h`.
pub fn covariance(a: &[f64], b: &[f64]) -> f64 {
    assert!(
        a.len() == b.len() && a.len() > 1,
        "covariance: a and b must be the same length and have more than one element"
    );
    let mut rs = RunningScalarCovariance::new();
    for i in 0..a.len() {
        rs.add(a[i], b[i]);
    }
    rs.covariance()
}

/// Port of `r_squared(a, b)` from `dlib/statistics/statistics.h`:
/// `correlation(a,b)^2`.
pub fn r_squared(a: &[f64], b: &[f64]) -> f64 {
    correlation(a, b).powi(2)
}

/// Port of `mean_squared_error(a, b)` from `dlib/statistics/statistics.h`:
/// `mean((a-b)^2)`.
pub fn mean_squared_error(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "mean_squared_error: a and b must be the same length"
    );
    mean(
        &a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .collect::<Vec<f64>>(),
    )
}

/// Mean of a slice, mirroring `dlib::mean(matrix)` semantics
/// (`sum/size`, 0 when empty).
pub fn mean(a: &[f64]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    a.iter().sum::<f64>() / a.len() as f64
}

/// Sample variance of a slice (n-1 denominator, two-pass), mirroring
/// `dlib::variance(matrix)`.
pub fn variance(a: &[f64]) -> f64 {
    assert!(a.len() > 1, "variance: need more than one element");
    let m = mean(a);
    let s = a.iter().map(|x| (x - m) * (x - m)).sum::<f64>();
    s / (a.len() - 1) as f64
}

/// Sample standard deviation of a slice, mirroring `dlib::standard_deviation`.
pub fn standard_deviation(a: &[f64]) -> f64 {
    variance(a).sqrt()
}

// ----------------------------------------------------------------------------------------

/// Port of `dlib::random_subset_selector<T>` from
/// `dlib/statistics/random_subset_selector.h`: a reservoir-style sampler that
/// keeps a uniformly random subset of at most `max_size` items among all
/// `add()`ed items, with `next_add_accepts()` exposing whether the next add
/// would be accepted (used to feed a paired selector of matching indices).
#[derive(Clone, Debug)]
pub struct RandomSubsetSelector<T> {
    items: Vec<T>,
    max_size: usize,
    count: u64,
    next_add_accepts: bool,
    rnd: Rand,
}

impl<T: Clone> Default for RandomSubsetSelector<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> RandomSubsetSelector<T> {
    /// `random_subset_selector()` (`_max_size == 0`, empty).
    pub fn new() -> Self {
        let mut s = RandomSubsetSelector {
            items: Vec::new(),
            max_size: 0,
            count: 0,
            next_add_accepts: false,
            rnd: Rand::new(),
        };
        s.update_next_add_accepts();
        s
    }

    /// `random_subset_selector::set_seed(value)`.
    pub fn set_seed(&mut self, value: &str) {
        self.rnd.set_seed(value);
    }

    /// `random_subset_selector::make_empty()`.
    pub fn make_empty(&mut self) {
        self.items.clear();
        self.count = 0;
        self.update_next_add_accepts();
    }

    /// `random_subset_selector::to_std_vector()`.
    pub fn to_std_vector(&self) -> &Vec<T> {
        &self.items
    }

    /// `random_subset_selector::size()`.
    pub fn size(&self) -> usize {
        self.items.len()
    }

    /// `random_subset_selector::set_max_size(new_max_size)`.
    pub fn set_max_size(&mut self, new_max_size: usize) {
        self.items.reserve(new_max_size);
        self.make_empty();
        self.max_size = new_max_size;
        self.update_next_add_accepts();
    }

    /// `random_subset_selector::max_size()`.
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// `random_subset_selector::operator[](idx)`.
    pub fn get(&self, idx: usize) -> &T {
        assert!(
            idx < self.size(),
            "random_subset_selector: idx out of range"
        );
        &self.items[idx]
    }

    /// `random_subset_selector::next_add_accepts()`.
    pub fn next_add_accepts(&self) -> bool {
        self.next_add_accepts
    }

    /// `random_subset_selector::add(new_item)`.
    pub fn add(&mut self, new_item: T) {
        if self.items.len() < self.max_size {
            self.items.push(new_item);
            // swap into a random place
            let i = (self.rnd.get_random_32bit_number() as u64 % self.items.len() as u64) as usize;
            let last = self.items.len() - 1;
            self.items.swap(i, last);
        } else if self.next_add_accepts {
            // pick a random element of items and replace it.
            let i = (self.rnd.get_random_32bit_number() as u64 % self.items.len() as u64) as usize;
            self.items[i] = new_item;
        }

        self.update_next_add_accepts();
        self.count += 1;
    }

    fn update_next_add_accepts(&mut self) {
        if self.items.len() < self.max_size {
            self.next_add_accepts = true;
        } else if self.max_size == 0 {
            self.next_add_accepts = false;
        } else {
            // Make a random 64 bit number (num1 << 32 | num2) and accept the
            // next add with probability items.size()/(count+1) by comparing
            // num % (count+1) < items.size().
            let num1 = self.rnd.get_random_32bit_number() as u64;
            let num2 = self.rnd.get_random_32bit_number() as u64;
            let mut num = num1;
            num <<= 32;
            num |= num2;

            num %= self.count + 1;

            self.next_add_accepts = (num as usize) < self.items.len();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_running_stats_one_to_ten() {
        let mut rs = RunningStats::new();
        for x in 1..=10 {
            rs.add(x as f64);
        }
        assert_eq!(rs.current_n(), 10.0);
        assert_eq!(rs.mean(), 5.5);
        // dlib variance uses the n-1 denominator: sum((x-5.5)^2)/9 = 82.5/9.
        assert!((rs.variance() - 82.5 / 9.0).abs() < 1e-12);
        assert!((rs.stddev() - (82.5 / 9.0f64).sqrt()).abs() < 1e-12);
        assert_eq!(rs.min(), 1.0);
        assert_eq!(rs.max(), 10.0);
        assert_eq!(rs.sum(), 55.0);

        // Hand formulas for a uniform 1..n ramp (symmetric => skewness 0).
        let n = 10.0f64;
        let m1 = 5.5;
        let m2: f64 = (1..=10)
            .map(|x| {
                let d = x as f64 - m1;
                d * d
            })
            .sum::<f64>()
            / n;
        let m3: f64 = (1..=10)
            .map(|x| {
                let d = x as f64 - m1;
                d * d * d
            })
            .sum::<f64>()
            / n;
        // dlib's skewness is the bias-corrected g1: sqrt(n(n-1))/(n-2) * m3/m2^{3/2}
        let expected_skew = (n * (n - 1.0)).sqrt() / (n - 2.0) * m3 / m2.powi(3).sqrt();
        // (for the symmetric ramp m3 == 0 so this is exactly 0)
        assert!((rs.skewness() - expected_skew).abs() < 1e-12);
        // dlib's ex_kurtosis: (n-1)((n+1) m4/m2^2 - 3(n-1)) / ((n-2)(n-3)),
        // where m2 and m4 are biased (divide-by-n) central moments computed
        // from the raw power sums exactly as the C++ formula does.
        let cxx_m2 = {
            let sum_sqr: f64 = (1..=10).map(|x| (x * x) as f64).sum();
            sum_sqr / n - m1 * m1
        };
        let cxx_m4 = {
            // m4 from raw moments: E[(X-m)^4] expanded is exactly what C++
            // computes via sum_four etc.; compute directly.
            (1..=10)
                .map(|x| {
                    let d = x as f64 - m1;
                    d * d * d * d
                })
                .sum::<f64>()
                / n
        };
        let expected_kurt = (n - 1.0) * ((n + 1.0) * cxx_m4 / (cxx_m2 * cxx_m2) - 3.0 * (n - 1.0))
            / ((n - 2.0) * (n - 3.0));
        assert!((rs.ex_kurtosis() - expected_kurt).abs() < 1e-12);
        assert!((rs.scale(6.0) - 0.5 / (82.5 / 9.0f64).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_running_stats_merge() {
        let mut a = RunningStats::new();
        let mut b = RunningStats::new();
        for x in 1..=6 {
            a.add(x as f64);
        }
        for x in 7..=10 {
            b.add(x as f64);
        }
        let c = a.merge(&b);
        assert_eq!(c.mean(), 5.5);
        assert!((c.variance() - 82.5 / 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_running_scalar_covariance() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [2.0, 4.1, 5.9, 8.2, 9.8];
        let mut rs = RunningScalarCovariance::new();
        for i in 0..xs.len() {
            rs.add(xs[i], ys[i]);
        }
        // two-pass references
        let mx = xs.iter().sum::<f64>() / xs.len() as f64;
        let my = ys.iter().sum::<f64>() / ys.len() as f64;
        let cov = xs
            .iter()
            .zip(&ys)
            .map(|(x, y)| (x - mx) * (y - my))
            .sum::<f64>()
            / (xs.len() - 1) as f64;
        let vx = xs.iter().map(|x| (x - mx) * (x - mx)).sum::<f64>() / (xs.len() - 1) as f64;
        let vy = ys.iter().map(|y| (y - my) * (y - my)).sum::<f64>() / (ys.len() - 1) as f64;
        assert!((rs.covariance() - cov).abs() < 1e-12);
        assert!((rs.variance_x() - vx).abs() < 1e-12);
        assert!((rs.variance_y() - vy).abs() < 1e-12);
        assert!((rs.correlation() - cov / (vx * vy).sqrt()).abs() < 1e-12);
        assert!((rs.stddev_x() - vx.sqrt()).abs() < 1e-12);
        assert_eq!(rs.current_n(), 5.0);
    }

    #[test]
    fn test_free_functions() {
        let a = [1.0, -2.0, 3.0, -4.0];
        let b = [5.0, -6.0, 7.0, -8.0];
        assert_eq!(mean_sign_agreement(&a, &b), 1.0);
        assert_eq!(mean_sign_agreement(&a, &[-1.0, 2.0, -3.0, 4.0]), 0.0);
        assert_eq!(mean_sign_agreement(&[0.0, 1.0], &[0.0, -1.0]), 0.5);

        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((correlation(&x, &x) - 1.0).abs() < 1e-12);
        assert!((r_squared(&x, &x) - 1.0).abs() < 1e-12);
        assert!(covariance(&x, &x) > 0.0);
        assert_eq!(mean(&x), 3.0);
        assert!((variance(&x) - 2.5).abs() < 1e-12);
        assert!((standard_deviation(&x) - 2.5f64.sqrt()).abs() < 1e-12);
        assert!((mean_squared_error(&x, &x) - 0.0).abs() < 1e-12);
        assert!((mean_squared_error(&x, &[2.0, 3.0, 4.0, 5.0, 6.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_random_subset_selector() {
        let mut sel: RandomSubsetSelector<i32> = RandomSubsetSelector::new();
        sel.set_max_size(5);
        for i in 0..100 {
            sel.add(i);
        }
        assert_eq!(sel.size(), 5);
        assert!(sel.to_std_vector().iter().all(|&v| (0..100).contains(&v)));
        // deterministic given the same default seed
        let mut sel2: RandomSubsetSelector<i32> = RandomSubsetSelector::new();
        sel2.set_max_size(5);
        for i in 0..100 {
            sel2.add(i);
        }
        assert_eq!(sel.to_std_vector(), sel2.to_std_vector());
        // when fewer items than max_size are added, all are kept
        let mut sel3: RandomSubsetSelector<i32> = RandomSubsetSelector::new();
        sel3.set_max_size(10);
        for i in 0..3 {
            sel3.add(i);
        }
        // add() swaps the new item into a random slot even while filling, so
        // only the multiset is guaranteed.
        let mut sorted = sel3.to_std_vector().clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2]);
    }
}
