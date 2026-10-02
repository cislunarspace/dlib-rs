//! Golden comparison against C++ dlib rand output (`golden/src/rand_golden.cpp`
//! → `tests/golden/rand.txt`). Draw sequences must match bit-for-bit; doubles
//! are compared exactly (17 significant digits round-trip f64), floats within
//! the printing precision of %.9g.

use dlib_rs_core::rand::Rand;
use dlib_rs_core::serialize::Serializer;

fn load() -> String {
    std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/golden/rand.txt"
    ))
    .expect("golden rand.txt present")
}

fn read_block<'a>(lines: &mut std::str::Lines<'a>, header: &str, count: usize) -> Vec<&'a str> {
    loop {
        let line = lines
            .next()
            .unwrap_or_else(|| panic!("missing header {header}"));
        if line.trim() == header {
            break;
        }
    }
    let mut out = Vec::with_capacity(count);
    for line in lines.by_ref() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if t.contains('[') {
            panic!("unexpected header {t:?} while reading {header}");
        }
        out.push(t);
        if out.len() == count {
            break;
        }
    }
    out
}

fn check_exact(want: &str, got: impl FnOnce() -> f64, ctx: &str) {
    let w: f64 = want.parse().unwrap();
    let g = got();
    assert_eq!(g.to_bits(), w.to_bits(), "{ctx}: got {g:e} want {w:e}");
}

#[test]
fn golden_rand_draws() {
    let text = load();
    let mut lines = text.lines();

    let want = read_block(&mut lines, "doubles []", 10_000);
    let mut r = Rand::new();
    for (i, w) in want.iter().enumerate() {
        check_exact(w, || r.get_random_double(), &format!("doubles[\"\"] #{i}"));
    }

    let want = read_block(&mut lines, "gaussians []", 10_000);
    let mut r = Rand::new();
    for (i, w) in want.iter().enumerate() {
        check_exact(
            w,
            || r.get_random_gaussian(),
            &format!("gaussians[\"\"] #{i}"),
        );
    }

    for seed in ["42", "seed123"] {
        let want = read_block(&mut lines, &format!("doubles [{seed}]"), 1_000);
        let mut r = Rand::with_seed(seed);
        for (i, w) in want.iter().enumerate() {
            check_exact(
                w,
                || r.get_random_double(),
                &format!("doubles[{seed}] #{i}"),
            );
        }
        let want = read_block(&mut lines, &format!("gaussians [{seed}]"), 1_000);
        let mut r = Rand::with_seed(seed);
        for (i, w) in want.iter().enumerate() {
            check_exact(
                w,
                || r.get_random_gaussian(),
                &format!("gaussians[{seed}] #{i}"),
            );
        }
    }

    // The C++ generator consumes these sections from a SINGLE dlib::rand
    // instance seeded "42" (blocks are not independent), so mirror that.
    let mut r = Rand::with_seed("42");
    let want = read_block(&mut lines, "u32 [42]", 100);
    for (i, w) in want.iter().enumerate() {
        assert_eq!(r.get_random_32bit_number().to_string(), *w, "u32 #{i}");
    }
    let want = read_block(&mut lines, "u64 [42]", 50);
    for (i, w) in want.iter().enumerate() {
        assert_eq!(r.get_random_64bit_number().to_string(), *w, "u64 #{i}");
    }
    let want = read_block(&mut lines, "u16 [42]", 50);
    for (i, w) in want.iter().enumerate() {
        assert_eq!(r.get_random_16bit_number().to_string(), *w, "u16 #{i}");
    }
    let want = read_block(&mut lines, "u8 [42]", 50);
    for (i, w) in want.iter().enumerate() {
        assert_eq!(r.get_random_8bit_number().to_string(), *w, "u8 #{i}");
    }

    // floats printed with %.9g: compare within print precision
    let want = read_block(&mut lines, "float [42]", 1000);
    for (i, w) in want.iter().enumerate() {
        let wv: f32 = w.parse().unwrap();
        let v = r.get_random_float();
        assert!(
            (v as f64 - wv as f64).abs() <= 1e-8 * (wv.abs() as f64).max(1e-30),
            "float #{i}: got {v:e} want {wv:e}"
        );
    }

    let want = read_block(&mut lines, "in_range [42]", 100);
    for (i, w) in want.iter().enumerate() {
        assert_eq!(
            r.get_integer_in_range(5, 23).to_string(),
            *w,
            "in_range #{i}"
        );
    }
    let want = read_block(&mut lines, "double_in_range [42]", 100);
    for (i, w) in want.iter().enumerate() {
        check_exact(
            w,
            || r.get_double_in_range(-2.5, 7.5),
            &format!("double_in_range #{i}"),
        );
    }
    let want = read_block(&mut lines, "exponential [42]", 100);
    for (i, w) in want.iter().enumerate() {
        check_exact(
            w,
            || r.get_random_exponential(1.7),
            &format!("exponential #{i}"),
        );
    }
    let want = read_block(&mut lines, "weibull [42]", 100);
    for (i, w) in want.iter().enumerate() {
        check_exact(
            w,
            || r.get_random_weibull(2.2, 1.3, 0.4),
            &format!("weibull #{i}"),
        );
    }
    let want = read_block(&mut lines, "beta [42]", 100);
    for (i, w) in want.iter().enumerate() {
        check_exact(w, || r.get_random_beta(2.5, 3.5), &format!("beta #{i}"));
    }
}

#[test]
fn golden_rand_serialized_state() {
    let text = load();
    let mut len = 0usize;
    let mut hex = String::new();
    let mut glen = 0usize;
    let mut ghex = String::new();
    let mut found = 0;
    for line in text.lines() {
        let t = line.trim();
        if let Some(v) = t.strip_prefix("state_hex_len ") {
            len = v.parse().unwrap();
        } else if let Some(v) = t.strip_prefix("state_hex [") {
            hex = v.trim_end_matches(']').to_string();
            found += 1;
        } else if let Some(v) = t.strip_prefix("gauss_state_hex_len ") {
            glen = v.parse().unwrap();
        } else if let Some(v) = t.strip_prefix("gauss_state_hex [") {
            ghex = v.trim_end_matches(']').to_string();
            found += 1;
        }
    }
    assert_eq!(found, 2, "both hex sections found");
    assert_eq!(hex.len() / 2, len);

    let mut r = Rand::with_seed("42");
    for _ in 0..1234 {
        r.get_random_double();
    }
    let mut s = Serializer::new();
    r.serialize(&mut s);
    let bytes = s.into_inner();
    assert_eq!(bytes.len(), len, "serialized length mismatch");
    let got: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(got, hex, "serialized state bytes differ from C++");

    let mut r = Rand::with_seed("seed123");
    r.get_random_gaussian();
    let mut s = Serializer::new();
    r.serialize(&mut s);
    let bytes = s.into_inner();
    assert_eq!(bytes.len(), glen);
    let got: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(got, ghex, "gaussian cached-state bytes differ from C++");
}
