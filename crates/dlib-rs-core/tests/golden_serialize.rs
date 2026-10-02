//! Golden comparison against C++ dlib serialize output
//! (`golden/src/serialize_golden.cpp` → `tests/golden/serialize.bin`). The
//! Rust serialization must be byte-identical and must deserialize the C++
//! stream back to the same values.
#![allow(clippy::approx_constant)]

use dlib_rs_core::matrix::Matrix;
use dlib_rs_core::serialize::{Deserializer, Serializer};

fn golden() -> Vec<u8> {
    std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/golden/serialize.bin"
    ))
    .expect("golden serialize.bin present")
}

#[test]
fn golden_serialize_bytes_match_and_roundtrip() {
    let mut s = Serializer::new();
    s.write_bool(true);
    s.write_bool(false);
    s.write_u8(0);
    s.write_u8(255);
    s.write_u16(65535);
    s.write_u32(0xDEADBEEF);
    s.write_u64(0xFFFF_FFFF_FFFF_FFFF);
    s.write_i16(-12345);
    s.write_i32(-12345678);
    s.write_i64(-12345678901234);
    s.write_f32(0.5);
    s.write_f32(3.14159);
    s.write_f32(-2.5e-30);
    for v in [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        std::f64::consts::PI,
        1e-300,
        1e300,
        5e-324,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        s.write_f64(v);
    }
    s.write_string("");
    s.write_string("hello");
    s.write_byte_vec(&[0xAA, 0xBB, 0x00, 0xFF]);
    s.write_vec(&[-1i32, 0, 1, 65536, -65536]);
    s.write_vec(&[0.25f64, -0.25, 1e100]);
    // std::vector<bool>: ASCII '1'/'0' bytes with u64 length
    s.write_u64(3);
    s.write_raw(b"101");

    let mut md = Matrix::zeros(2, 3);
    for r in 0..2usize {
        for c in 0..3usize {
            md[(r, c)] = 0.1 * (r + 1) as f64 + (c + 1) as f64;
        }
    }
    md.serialize(&mut s);
    let mut mf = Matrix::zeros(2, 2);
    for r in 0..2usize {
        for c in 0..2usize {
            mf[(r, c)] = 0.5 * (r + c) as f32;
        }
    }
    mf.serialize(&mut s);
    let mut mi = Matrix::zeros(2, 2);
    for r in 0..2usize {
        for c in 0..2usize {
            mi[(r, c)] = (10 * r + c) as i64;
        }
    }
    mi.serialize(&mut s);

    let bytes = s.into_inner();
    let want = golden();
    assert_eq!(bytes.len(), want.len(), "stream length mismatch");
    for (i, (a, b)) in bytes.iter().zip(want.iter()).enumerate() {
        assert_eq!(a, b, "byte {i}: got {a:02x} want {b:02x}");
    }

    // Deserialize the C++ stream back.
    let want = golden();
    let mut d = Deserializer::new(&want);
    assert!(d.read_bool().unwrap());
    assert!(!d.read_bool().unwrap());
    assert_eq!(d.read_u8().unwrap(), 0);
    assert_eq!(d.read_u8().unwrap(), 255);
    assert_eq!(d.read_u16().unwrap(), 65535);
    assert_eq!(d.read_u32().unwrap(), 0xDEADBEEF);
    assert_eq!(d.read_u64().unwrap(), u64::MAX);
    assert_eq!(d.read_i16().unwrap(), -12345);
    assert_eq!(d.read_i32().unwrap(), -12345678);
    assert_eq!(d.read_i64().unwrap(), -12345678901234);
    assert_eq!(d.read_f32().unwrap(), 0.5);
    assert_eq!(d.read_f32().unwrap(), 3.14159);
    assert_eq!(d.read_f32().unwrap(), -2.5e-30);
    // -0.0 deserializes as +0.0: dlib's float_details stores the mantissa as
    // an integer, losing the sign of zero (verified byte-identical stream).
    for v in [
        0.0f64,
        0.0,
        1.0,
        -1.0,
        0.5,
        std::f64::consts::PI,
        1e-300,
        1e300,
        5e-324,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        assert_eq!(d.read_f64().unwrap().to_bits(), v.to_bits(), "f64 {v}");
    }
    assert_eq!(d.read_string().unwrap(), "");
    assert_eq!(d.read_string().unwrap(), "hello");
    assert_eq!(d.read_byte_vec().unwrap(), vec![0xAA, 0xBB, 0x00, 0xFF]);
    let vi: Vec<i32> = d.read_vec().unwrap();
    assert_eq!(vi, vec![-1, 0, 1, 65536, -65536]);
    let vd: Vec<f64> = d.read_vec().unwrap();
    assert_eq!(vd, vec![0.25, -0.25, 1e100]);
    // vector<bool>
    let n = d.read_u64().unwrap();
    assert_eq!(n, 3);
    let raw = d.read_raw(3).unwrap();
    assert_eq!(raw, b"101");

    let md2 = Matrix::<f64>::deserialize(&mut d).unwrap();
    assert_eq!(md2, md);
    let mf2 = Matrix::<f32>::deserialize(&mut d).unwrap();
    assert_eq!(mf2, mf);
    let mi2 = Matrix::<i64>::deserialize(&mut d).unwrap();
    assert_eq!(mi2, mi);
    assert_eq!(d.remaining(), 0, "whole stream consumed");
}
