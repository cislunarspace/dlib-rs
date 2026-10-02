//! Bit-exact Rust port of dlib's random number facilities.
//!
//! Ported from:
//! - `dlib/rand/mersenne_twister.h` (Boost `mersenne_twister`, `mt19937` typedef)
//! - `dlib/rand/rand_kernel_1.h` (class `dlib::rand` and its `serialize`/`deserialize`)
//!
//! All arithmetic is performed with the same widths and evaluation order as the
//! C++ code so that outputs are bit-for-bit identical to dlib.
use crate::serialize::{Deserializer, SerializeError, Serializer};

/// Boost/dlib MT19937 mersenne twister.
///
/// Ported from `dlib::random_helpers::mersenne_twister<uint32,32,624,397,31,
/// 0x9908b0df,11,7,0x9d2c5680,15,0xefc60000,18,3346425566U>` (`mt19937`) in
/// `dlib/rand/mersenne_twister.h`. The state buffer is doubled (`2*n` words)
/// so that the previous block is always available, exactly as in the C++
/// implementation; `twist` replicates the two-block scheme used to avoid
/// modulo operations.
#[derive(Clone, Debug, PartialEq)]
struct Mt19937 {
    x: [u32; Self::N2],
    i: i32,
}

impl Mt19937 {
    const N: usize = 624;
    const M: usize = 397;
    const N2: usize = 2 * 624;
    const A: u32 = 0x9908_b0df;
    const R: u32 = 31;

    /// Default constructor; equivalent to `mersenne_twister()` (`seed(5489)`).
    ///
    /// Ported from `mersenne_twister::mersenne_twister()` in
    /// `dlib/rand/mersenne_twister.h`.
    fn new() -> Self {
        let mut mt = Mt19937 {
            x: [0; Self::N2],
            i: 0,
        };
        mt.seed(5489);
        mt
    }

    /// Ported from `mersenne_twister::seed(UIntType value)`.
    ///
    /// Note: the C++ header leaves `i` untouched here, but compiled dlib
    /// observes the fresh-object value `i == n` (so the first `operator()`
    /// performs `twist(0)`), which we set explicitly to keep the output
    /// stream bit-identical to compiled dlib.
    fn seed(&mut self, value: u32) {
        self.x[0] = value;
        for i in 1..Self::N {
            let prev = self.x[i - 1];
            self.x[i] = 1812433253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        self.i = Self::N as i32;
    }

    /// Ported from `mersenne_twister::twist(int block)`.
    fn twist(&mut self, block: i32) {
        let upper_mask = !0u32 << Self::R;
        let lower_mask = !upper_mask;

        if block == 0 {
            for j in Self::N..Self::N2 {
                let y =
                    (self.x[j - Self::N] & upper_mask) | (self.x[j - (Self::N - 1)] & lower_mask);
                self.x[j] = self.x[j - (Self::N - Self::M)]
                    ^ (y >> 1)
                    ^ (if y & 1 != 0 { Self::A } else { 0 });
            }
        } else if block == 1 {
            for j in 0..Self::N - Self::M {
                let y = (self.x[j + Self::N] & upper_mask) | (self.x[j + Self::N + 1] & lower_mask);
                self.x[j] = self.x[j + Self::N + Self::M]
                    ^ (y >> 1)
                    ^ (if y & 1 != 0 { Self::A } else { 0 });
            }
            for j in Self::N - Self::M..Self::N - 1 {
                let y = (self.x[j + Self::N] & upper_mask) | (self.x[j + Self::N + 1] & lower_mask);
                self.x[j] = self.x[j - (Self::N - Self::M)]
                    ^ (y >> 1)
                    ^ (if y & 1 != 0 { Self::A } else { 0 });
            }
            // last iteration
            let y = (self.x[Self::N2 - 1] & upper_mask) | (self.x[0] & lower_mask);
            self.x[Self::N - 1] =
                self.x[Self::M - 1] ^ (y >> 1) ^ (if y & 1 != 0 { Self::A } else { 0 });
            self.i = 0;
        }
    }

    /// Ported from `mersenne_twister::operator()()` (tempering included).
    fn next(&mut self) -> u32 {
        if self.i == Self::N as i32 {
            self.twist(0);
        } else if self.i >= Self::N2 as i32 {
            self.twist(1);
        }
        let mut z = self.x[self.i as usize];
        self.i += 1;
        z ^= z >> 11;
        z ^= (z << 7) & 0x9d2c_5680;
        z ^= (z << 15) & 0xefc6_0000;
        z ^= z >> 18;
        z
    }
}

/// dlib's random number generator, a bit-exact port of `dlib::rand`
/// (`dlib/rand/rand_kernel_1.h`).
#[derive(Clone, Debug)]
pub struct Rand {
    mt: Mt19937,
    seed: String,
    max_val: f64,
    has_gaussian: bool,
    next_gaussian: f64,
}

impl Rand {
    /// Ported from `rand::rand()` (`init()` primes the generator with 10000
    /// draws and computes `max_val`).
    pub fn new() -> Self {
        // init()
        let mt = Mt19937::new();
        let mut rng = Rand {
            mt,
            seed: String::new(),
            max_val: 0.0,
            has_gaussian: false,
            next_gaussian: 0.0,
        };
        for _ in 0..10000 {
            rng.mt.next();
        }
        rng.max_val = 0xFF_FFFF as f64;
        rng.max_val *= 0x100_0000 as f64;
        rng.max_val += 0xFF_FFFF as f64;
        rng.max_val += 0.05;
        rng
    }

    /// Ported from `rand::rand(const std::string& seed_value)`.
    pub fn with_seed(seed_value: &str) -> Self {
        let mut rng = Rand::new();
        rng.set_seed(seed_value);
        rng
    }

    /// Ported from `rand::rand(time_t seed_value)`
    /// (`set_seed(cast_to_string(seed_value))`).
    pub fn with_seed_i64(seed_value: i64) -> Self {
        let mut rng = Rand::new();
        rng.set_seed(&seed_value.to_string());
        rng
    }

    /// Ported from `rand::clear()`.
    pub fn clear(&mut self) {
        self.mt.seed(5489);
        self.seed.clear();
        self.has_gaussian = false;
        self.next_gaussian = 0.0;
        for _ in 0..10000 {
            self.mt.next();
        }
    }

    /// Ported from `rand::get_seed()`.
    pub fn get_seed(&self) -> &str {
        &self.seed
    }

    /// Ported from `rand::set_seed(const std::string&)`.
    ///
    /// Note: dlib iterates over the seed's `char`s cast to `uint32`; for
    /// non-ASCII seeds on platforms where `char` is signed the C++ behavior
    /// differs, but for ASCII seeds (the only usage dlib makes) bytes are
    /// identical to chars.
    pub fn set_seed(&mut self, value: &str) {
        self.seed = value.to_string();
        if !value.is_empty() {
            let mut s: u32 = 0;
            for b in value.bytes() {
                s = s.wrapping_mul(37).wrapping_add(b as u32);
            }
            self.mt.seed(s);
        } else {
            self.mt.seed(5489);
        }
        for _ in 0..10000 {
            self.mt.next();
        }
        self.has_gaussian = false;
        self.next_gaussian = 0.0;
    }

    /// Ported from `rand::get_random_8bit_number()`.
    pub fn get_random_8bit_number(&mut self) -> u8 {
        self.mt.next() as u8
    }

    /// Ported from `rand::get_random_16bit_number()`.
    pub fn get_random_16bit_number(&mut self) -> u16 {
        self.mt.next() as u16
    }

    /// Ported from `rand::get_random_32bit_number()`.
    pub fn get_random_32bit_number(&mut self) -> u32 {
        self.mt.next()
    }

    /// Ported from `rand::get_random_64bit_number()`.
    pub fn get_random_64bit_number(&mut self) -> u64 {
        let a = self.get_random_32bit_number() as u64;
        let b = self.get_random_32bit_number() as u64;
        (a << 32) | b
    }

    /// Ported from `rand::get_double_in_range(begin, end)`.
    pub fn get_double_in_range(&mut self, begin: f64, end: f64) -> f64 {
        begin + self.get_random_double() * (end - begin)
    }

    /// Ported from `rand::get_integer_in_range(begin, end)` (rejection
    /// sampling over the full 64-bit range).
    pub fn get_integer_in_range(&mut self, begin: i64, end: i64) -> i64 {
        if begin == end {
            return begin;
        }
        let mut r = self.get_random_64bit_number();
        let limit = u64::MAX;
        let range = (end - begin) as u64;
        while r >= (limit / range) * range {
            r = self.get_random_64bit_number();
        }
        begin + (r % range) as i64
    }

    /// Ported from `rand::get_integer(end)`.
    pub fn get_integer(&mut self, end: i64) -> i64 {
        self.get_integer_in_range(0, end)
    }

    /// Ported from `rand::get_random_double()`.
    pub fn get_random_double(&mut self) -> f64 {
        let mut temp = self.get_random_32bit_number();
        temp &= 0xFF_FFFF;

        let mut val = temp as f64;
        val *= 0x100_0000 as f64;

        let mut temp = self.get_random_32bit_number();
        temp &= 0xFF_FFFF;

        val += temp as f64;
        val /= self.max_val;

        if val < 1.0 {
            val
        } else {
            1.0 - f64::EPSILON
        }
    }

    /// Ported from `rand::get_random_float()`.
    pub fn get_random_float(&mut self) -> f32 {
        let mut temp = self.get_random_32bit_number();
        temp &= 0xFF_FFFF;

        let scale = 1.0f32 / 0x100_0000 as f32;
        let val = temp as f32 * scale;
        if val < 1.0f32 {
            val
        } else {
            1.0f32 - f32::EPSILON
        }
    }

    /// Ported from `rand::get_random_complex_gaussian()` (Box-Muller);
    /// returns `(real, imag)`.
    pub fn get_random_complex_gaussian(&mut self) -> (f64, f64) {
        let rndmax = u32::MAX as f64;
        let x1;
        let x2;
        let mut w;
        loop {
            let rnd1 = self.get_random_32bit_number() as f64 / rndmax;
            let rnd2 = self.get_random_32bit_number() as f64 / rndmax;
            let a = 2.0 * rnd1 - 1.0;
            let b = 2.0 * rnd2 - 1.0;
            w = a * a + b * b;
            if w < 1.0 {
                x1 = a;
                x2 = b;
                break;
            }
        }
        w = ((-2.0 * w.ln()) / w).sqrt();
        (x1 * w, x2 * w)
    }

    /// Ported from `rand::get_random_gaussian()`; caches the imaginary part
    /// of the Box-Muller pair and returns it on the next call.
    pub fn get_random_gaussian(&mut self) -> f64 {
        if self.has_gaussian {
            self.has_gaussian = false;
            return self.next_gaussian;
        }
        let (re, im) = self.get_random_complex_gaussian();
        self.next_gaussian = im;
        self.has_gaussian = true;
        re
    }

    /// Ported from `rand::get_random_exponential(lambda)`.
    pub fn get_random_exponential(&mut self, lambda: f64) -> f64 {
        let mut u = 0.0;
        while u == 0.0 {
            u = self.get_random_double();
        }
        -u.ln() / lambda
    }

    /// Ported from `rand::get_random_weibull(lambda, k, gamma)`.
    pub fn get_random_weibull(&mut self, lambda: f64, k: f64, gamma: f64) -> f64 {
        let mut u = 0.0;
        while u == 0.0 {
            u = self.get_random_double();
        }
        gamma + lambda * (-u.ln()).powf(1.0 / k)
    }

    /// Ported from `rand::get_random_beta(alpha, beta)`.
    pub fn get_random_beta(&mut self, alpha: f64, beta: f64) -> f64 {
        let mut u = self.get_random_double().powf(1.0 / alpha);
        let mut v = self.get_random_double().powf(1.0 / beta);
        while (u + v) > 1.0 || (u == 0.0 && v == 0.0) {
            u = self.get_random_double().powf(1.0 / alpha);
            v = self.get_random_double().powf(1.0 / beta);
        }
        u / (u + v)
    }

    /// Ported from `serialize(const rand&, std::ostream&)` in
    /// `dlib/rand/rand_kernel_1.h` (writes the mt19937 state `x` + `i`, the
    /// seed string, `has_gaussian` and `next_gaussian`, version 1).
    pub fn serialize(&self, out: &mut Serializer) {
        out.write_i32(1); // version
                          // dlib::serialize(item.mt, out): C array u32[2*n] then index i
        out.write_u64(Mt19937::N2 as u64);
        for v in &self.mt.x {
            out.write_u32(*v);
        }
        out.write_i32(self.mt.i);
        out.write_string(&self.seed);
        out.write_bool(self.has_gaussian);
        out.write_f64(self.next_gaussian);
    }

    /// Ported from `deserialize(rand&, std::istream&)`.
    pub fn deserialize(inp: &mut Deserializer<'_>) -> Result<Rand, SerializeError> {
        let version = inp.read_i32()?;
        if version != 1 {
            return Err(SerializeError::Version(
                "Error deserializing object of type rand: unexpected version.".to_string(),
            ));
        }
        let len = inp.read_u64()?;
        if len != Mt19937::N2 as u64 {
            return Err(SerializeError::Malformed(
                "rand: mt19937 state array has wrong length",
            ));
        }
        let mut x = [0u32; Mt19937::N2];
        for v in x.iter_mut() {
            *v = inp.read_u32()?;
        }
        let i = inp.read_i32()?;
        let seed = inp.read_string()?;
        let has_gaussian = inp.read_bool()?;
        let next_gaussian = inp.read_f64()?;
        // Reconstruct max_val the same way init() computes it.
        let mut max_val = 0xFF_FFFF as f64;
        max_val *= 0x100_0000 as f64;
        max_val += 0xFF_FFFF as f64;
        max_val += 0.05;
        Ok(Rand {
            mt: Mt19937 { x, i },
            seed,
            max_val,
            has_gaussian,
            next_gaussian,
        })
    }
}

impl Default for Rand {
    fn default() -> Self {
        Rand::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_default_seed_reference_sequence() {
        let mut mt = Mt19937::new();
        mt.seed(5489);
        for _ in 0..10000 {
            mt.next();
        }
        // Values verified by compiling and running the actual dlib headers
        // (dlib::rand default seed): the 10001st..10005th outputs of MT19937.
        let expected = [725333953u32, 251387296, 3200466189, 2466988778, 2049276419];
        let got: Vec<u32> = (0..5).map(|_| mt.next()).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn matches_compiled_dlib_rand_streams() {
        // Values produced by a C++ program built against the local dlib
        // headers (g++ -I dlib rand_kernel_1.h).
        let mut r = Rand::new();
        let default_stream: Vec<u32> = (0..5).map(|_| r.get_random_32bit_number()).collect();
        assert_eq!(
            default_stream,
            vec![725333953, 251387296, 3200466189, 2466988778, 2049276419]
        );

        let mut r = Rand::new();
        r.set_seed("abc");
        let abc_stream: Vec<u32> = (0..5).map(|_| r.get_random_32bit_number()).collect();
        assert_eq!(
            abc_stream,
            vec![2155509415, 946363555, 2146784666, 2833049956, 2618434060]
        );
    }

    #[test]
    fn empty_seed_matches_clear() {
        let mut a = Rand::new();
        a.set_seed("");
        let mut b = Rand::new();
        b.clear();
        for _ in 0..100 {
            assert_eq!(a.get_random_32bit_number(), b.get_random_32bit_number());
        }
    }

    #[test]
    fn gaussian_consumes_pairs() {
        let mut base = Rand::new();
        for _ in 0..31 {
            let _ = base.get_random_32bit_number();
        }
        let mut b = base.clone();
        let mut x = base.clone();
        let mut y = base.clone();
        let _ = x.get_random_gaussian();
        let _ = x.get_random_gaussian(); // served from cache: no extra draws
        let _ = y.get_random_gaussian(); // same single complex draw as x's pair
        for _ in 0..20 {
            let vx = x.get_random_32bit_number();
            assert_eq!(vx, y.get_random_32bit_number());
            assert_ne!(vx, b.get_random_32bit_number()); // the pair did consume draws
        }
    }

    #[test]
    fn serialize_roundtrip_bit_exact() {
        let mut rng = Rand::with_seed("dlib-rs test");
        // advance into an interesting part of the stream
        for _ in 0..77 {
            let _ = rng.get_random_double();
        }
        let _ = rng.get_random_gaussian(); // set has_gaussian
        let mut ser = Serializer::new();
        rng.serialize(&mut ser);
        let bytes = ser.into_inner();
        let mut de = Deserializer::new(&bytes);
        let mut rng2 = Rand::deserialize(&mut de).unwrap();
        for _ in 0..1000 {
            assert_eq!(
                rng.get_random_64bit_number(),
                rng2.get_random_64bit_number()
            );
            assert_eq!(
                rng.get_random_double().to_bits(),
                rng2.get_random_double().to_bits()
            );
            assert_eq!(
                rng.get_random_gaussian().to_bits(),
                rng2.get_random_gaussian().to_bits()
            );
        }
        assert_eq!(rng.get_seed(), rng2.get_seed());
    }

    #[test]
    fn with_seed_matches_manual_set_seed() {
        let mut a = Rand::with_seed_i64(12345);
        let mut b = Rand::new();
        b.set_seed("12345");
        for _ in 0..10 {
            assert_eq!(a.get_random_32bit_number(), b.get_random_32bit_number());
        }
    }

    #[test]
    fn ranges_and_distributions() {
        let mut rng = Rand::new();
        for _ in 0..1000 {
            let v = rng.get_random_double();
            assert!((0.0..=1.0).contains(&v));
            let f = rng.get_random_float();
            assert!((0.0f32..=1.0).contains(&f));
            let d = rng.get_double_in_range(2.0, 5.0);
            assert!((2.0..=5.0).contains(&d));
            let i = rng.get_integer_in_range(3, 17);
            assert!((3..=16).contains(&i));
            assert!(rng.get_random_exponential(2.0) >= 0.0);
            assert!(rng.get_random_weibull(1.0, 2.0, 0.5) >= 0.5);
            let beta = rng.get_random_beta(2.0, 3.0);
            assert!((0.0..=1.0).contains(&beta));
        }
    }

    #[test]
    fn deserialize_version_check() {
        let mut ser = Serializer::new();
        ser.write_i32(2);
        let bytes = ser.into_inner();
        let mut de = Deserializer::new(&bytes);
        assert!(matches!(
            Rand::deserialize(&mut de),
            Err(SerializeError::Version(_))
        ));
    }
}
