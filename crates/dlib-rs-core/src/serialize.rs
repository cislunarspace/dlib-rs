//! Binary serialization compatible with dlib's `dlib/serialize.h`.
//!
//! Ported from dlib (commit 46fa4a28): `dlib/serialize.h` (integral packing,
//! bool/string/vector formats, legacy ASCII floating point deserialization)
//! and `dlib/float_details.h` (mantissa/exponent floating point format).
//!
//! Format summary (identical to the C++ implementation):
//! - integral types: one control byte (low 4 bits = number of following
//!   little-endian magnitude bytes, high bit = negative flag for signed types)
//!   followed by the magnitude bytes;
//! - `bool`: the single ASCII byte `'1'` or `'0'`;
//! - floats: a `float_details` pair (i64 mantissa, i16 exponent, both in the
//!   packed integral format); deserialization transparently accepts the old
//!   ASCII format used by ancient dlib files;
//! - `String`: packed u64 length + raw bytes;
//! - `Vec<T>`: packed u64 length + elements; `Vec<u8>` stores raw bytes.

use std::fmt;

/// Error type for dlib-compatible (de)serialization.
///
/// Ported from the `serialization_error` exception in `dlib/serialize.h`.
#[derive(thiserror::Error, Debug)]
pub enum SerializeError {
    /// Underlying I/O failure (rare here since sinks are `Vec<u8>`).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// Serialized version string mismatch (`check_serialized_version`).
    #[error("version mismatch: {0}")]
    Version(String),
    /// Ran off the end of the input buffer.
    #[error("unexpected end of input")]
    Eof,
    /// Input is not valid dlib serialization data.
    #[error("malformed data: {0}")]
    Malformed(&'static str),
}

impl PartialEq for SerializeError {
    fn eq(&self, other: &Self) -> bool {
        use SerializeError::*;
        match (self, other) {
            (Io(a), Io(b)) => a.kind() == b.kind(),
            (Version(a), Version(b)) => a == b,
            (Eof, Eof) => true,
            (Malformed(a), Malformed(b)) => a == b,
            _ => false,
        }
    }
}

/// dlib packed-unsigned-int encoding (`ser_helper::pack_int` in `dlib/serialize.h`).
///
/// Writes a control byte holding the minimal number of little-endian bytes of
/// `item`, followed by those bytes.
fn pack_uint(out: &mut Vec<u8>, item: u64) {
    let mut buf = [0u8; 9];
    let mut v = item;
    let mut size = 8u8;
    for i in 1..=8u8 {
        buf[i as usize] = (v & 0xFF) as u8;
        v >>= 8;
        if v == 0 {
            size = i;
            break;
        }
    }
    buf[0] = size;
    out.extend_from_slice(&buf[..size as usize + 1]);
}

/// dlib packed-signed-int encoding (`ser_helper::pack_int` in `dlib/serialize.h`).
///
/// Same as [`pack_uint`] on the magnitude, with the control byte's high bit
/// set when the value is negative. Mirrors the C++ code exactly (including
/// its use of wrapping negation for `i64::MIN`).
fn pack_int(out: &mut Vec<u8>, item: i64) {
    let mut buf = [0u8; 9];
    let neg: u8;
    let mut v = item;
    if v < 0 {
        neg = 0x80;
        v = v.wrapping_neg();
    } else {
        neg = 0;
    }
    let mut size = 8u8;
    for i in 1..=8u8 {
        buf[i as usize] = (v & 0xFF) as u8;
        v >>= 8;
        if v == 0 {
            size = i;
            break;
        }
    }
    buf[0] = size | neg;
    out.extend_from_slice(&buf[..size as usize + 1]);
}

/// dlib packed-unsigned-int decoding (`ser_helper::unpack_int` in `dlib/serialize.h`).
///
/// `max_size` is `sizeof(T)`; the control byte is masked with `0x8F` (3
/// reserved bits ignored, mirroring the C++ code) and sizes of 0 or above
/// `max_size` are rejected.
fn unpack_uint(inp: &mut Deserializer<'_>, max_size: u8) -> Result<u64, SerializeError> {
    let control = inp.read_u8()?;
    let size = control & 0x8F;
    if size == 0 || size > max_size.min(8) {
        return Err(SerializeError::Malformed("integer control byte"));
    }
    let bytes = inp.read_raw(size as usize)?;
    let mut item: u64 = 0;
    for &b in bytes.iter().rev() {
        item <<= 8;
        item |= b as u64;
    }
    Ok(item)
}

/// dlib packed-signed-int decoding (`ser_helper::unpack_int` in `dlib/serialize.h`).
///
/// Negative when the control byte's high bit is set; size is the low 4 bits.
fn unpack_int(inp: &mut Deserializer<'_>, max_size: u8) -> Result<i64, SerializeError> {
    let control = inp.read_u8()?;
    let is_negative = control & 0x80 != 0;
    let size = control & 0x0F;
    if size == 0 || size > max_size.min(8) {
        return Err(SerializeError::Malformed("integer control byte"));
    }
    let bytes = inp.read_raw(size as usize)?;
    let mut item: u64 = 0;
    for &b in bytes.iter().rev() {
        item <<= 8;
        item |= b as u64;
    }
    if is_negative {
        Ok((item as i64).wrapping_neg())
    } else {
        Ok(item as i64)
    }
}

// ----------------------------------------------------------------------------------------
// float_details (dlib/float_details.h)

const IS_INF: i16 = 32000;
const IS_NINF: i16 = 32001;
const IS_NAN: i16 = 32002;

/// `std::frexp` for `f64`: returns `(m, e)` with `x == m * 2^e` and `m` in
/// `[0.5, 1)`. Used by the `float_details` conversion in `dlib/float_details.h`.
fn frexp64(x: f64) -> (f64, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let exp_field = ((bits >> 52) & 0x7FF) as i32;
    let frac = bits & 0x000F_FFFF_FFFF_FFFF;
    if exp_field == 0 {
        // Subnormal: scale up by 2^52 to normalize, then adjust.
        let (m, e) = frexp64(x * 4_503_599_627_370_496.0);
        (m, e - 52)
    } else {
        let m = f64::from_bits(frac | 0x3FE0_0000_0000_0000);
        (if x < 0.0 { -m } else { m }, exp_field - 1022)
    }
}

/// `std::frexp` for `f32`.
fn frexp32(x: f32) -> (f32, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let exp_field = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x007F_FFFF;
    if exp_field == 0 {
        let (m, e) = frexp32(x * 8_388_608.0); // 2^23
        (m, e - 23)
    } else {
        let m = f32::from_bits(frac | 0x3F00_0000);
        (if x < 0.0 { -m } else { m }, exp_field - 126)
    }
}

/// `std::ldexp` for `f64`: computes `m * 2^e` with a single correctly-rounded
/// scaling whenever possible (multiplication by an exact power of two rounds
/// identically to C `ldexp` except on overflow/underflow, where both produce
/// the same saturated values).
fn ldexp64(m: f64, e: i32) -> f64 {
    let mut x = m;
    let mut e = e;
    while e > 1023 {
        x *= 8.98846567431158e307; // 2^1023
        e -= 1023;
        if !x.is_finite() {
            return x;
        }
    }
    while e < -1022 {
        x *= 2.2250738585072014e-308; // 2^-1022
        e += 1022;
        if x == 0.0 {
            return x;
        }
    }
    x * pow2_64(e)
}

fn pow2_64(e: i32) -> f64 {
    // Exact for every e in [-1074, 1023] via bit construction.
    if e >= -1022 {
        let biased = (e + 1023) as u64;
        f64::from_bits(biased << 52)
    } else {
        // Subnormal powers of two.
        let shift = (-1022 - e) as u32;
        f64::from_bits(1u64 << (52 - shift))
    }
}

/// `std::ldexp` for `f32`.
fn ldexp32(m: f32, e: i32) -> f32 {
    let mut x = m;
    let mut e = e;
    while e > 127 {
        x *= 1.7014118e38; // 2^127
        e -= 127;
        if !x.is_finite() {
            return x;
        }
    }
    while e < -126 {
        x *= 1.1754944e-38; // 2^-126
        e += 126;
        if x == 0.0 {
            return x;
        }
    }
    x * pow2_32(e)
}

fn pow2_32(e: i32) -> f32 {
    if e >= -126 {
        let biased = (e + 127) as u32;
        f32::from_bits(biased << 23)
    } else {
        let shift = (-126 - e) as u32;
        f32::from_bits(1u32 << (23 - shift))
    }
}

/// dlib `float_details` from `dlib/float_details.h`: value is
/// `mantissa * 2^exponent`, with sentinel exponents for inf/-inf/NaN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FloatDetails {
    /// Signed mantissa (sign carried here, not in the exponent).
    pub mantissa: i64,
    /// Power-of-two exponent, or one of the inf/-inf/NaN sentinels.
    pub exponent: i16,
}

impl FloatDetails {
    /// Builds from an explicit mantissa/exponent pair.
    pub fn new(mantissa: i64, exponent: i16) -> Self {
        FloatDetails { mantissa, exponent }
    }

    /// Converts an `f64` (port of `float_details::convert_from_T<double>`).
    pub fn from_f64(val: f64) -> Self {
        let digits = 53i32;
        if val == f64::INFINITY {
            return FloatDetails::new(0, IS_INF);
        } else if val == f64::NEG_INFINITY {
            return FloatDetails::new(0, IS_NINF);
        } else if val < f64::INFINITY {
            let (m, e) = frexp64(val);
            let mut mantissa = (m * 9_007_199_254_740_992.0) as i64; // m * 2^53
            let mut exponent_i = e - digits;
            // Compact: shift off low-order zero bytes (up to 8 rounds).
            for _ in 0..8 {
                if (mantissa as u64) & 0xFF == 0 {
                    mantissa >>= 8;
                    exponent_i += 8;
                } else {
                    break;
                }
            }
            return FloatDetails::new(mantissa, exponent_i as i16);
        }
        FloatDetails::new(0, IS_NAN)
    }

    /// Converts an `f32` (port of `float_details::convert_from_T<float>`).
    pub fn from_f32(val: f32) -> Self {
        let digits = 24i32;
        if val == f32::INFINITY {
            return FloatDetails::new(0, IS_INF);
        } else if val == f32::NEG_INFINITY {
            return FloatDetails::new(0, IS_NINF);
        } else if val < f32::INFINITY {
            let (m, e) = frexp32(val);
            let mut mantissa = (m * 16_777_216.0f32) as i64; // m * 2^24
            let mut exponent_i = e - digits;
            for _ in 0..8 {
                if (mantissa as u64) & 0xFF == 0 {
                    mantissa >>= 8;
                    exponent_i += 8;
                } else {
                    break;
                }
            }
            return FloatDetails::new(mantissa, exponent_i as i16);
        }
        FloatDetails::new(0, IS_NAN)
    }

    /// Converts back to `f64` (port of `float_details::convert_to_T<double>`).
    pub fn to_f64(&self) -> f64 {
        if self.exponent < IS_INF {
            ldexp64(self.mantissa as f64, self.exponent as i32)
        } else if self.exponent == IS_INF {
            f64::INFINITY
        } else if self.exponent == IS_NINF {
            f64::NEG_INFINITY
        } else {
            f64::NAN
        }
    }

    /// Converts back to `f32` (port of `float_details::convert_to_T<float>`:
    /// the mantissa is cast to `float` first, then scaled).
    pub fn to_f32(&self) -> f32 {
        if self.exponent < IS_INF {
            ldexp32(self.mantissa as f32, self.exponent as i32)
        } else if self.exponent == IS_INF {
            f32::INFINITY
        } else if self.exponent == IS_NINF {
            f32::NEG_INFINITY
        } else {
            f32::NAN
        }
    }
}

// ----------------------------------------------------------------------------------------
// Legacy ASCII floating point deserialization (dlib/serialize.h)

/// Port of `old_deserialize_floating_point` from `dlib/serialize.h`: parses an
/// ASCII float (or `inf`/`n...`/`N...` sentinels) followed by one space.
fn old_deserialize_floating_point64(inp: &mut Deserializer<'_>) -> Result<f64, SerializeError> {
    let b = inp.peek_byte().ok_or(SerializeError::Eof)?;
    let val = match b {
        b'i' => {
            inp.read_raw(3)?;
            f64::INFINITY
        }
        b'n' => {
            inp.read_raw(4)?;
            f64::NEG_INFINITY
        }
        b'N' => {
            inp.read_raw(3)?;
            f64::NAN
        }
        _ => parse_ascii_float64(inp)?,
    };
    // C++: `return (in.get() != ' ')` — a trailing space is required and consumed.
    let trailer = inp.read_u8()?;
    if trailer == b' ' {
        Ok(val)
    } else {
        Err(SerializeError::Malformed("legacy float"))
    }
}

fn old_deserialize_floating_point32(inp: &mut Deserializer<'_>) -> Result<f32, SerializeError> {
    let b = inp.peek_byte().ok_or(SerializeError::Eof)?;
    let val = match b {
        b'i' => {
            inp.read_raw(3)?;
            f32::INFINITY
        }
        b'n' => {
            inp.read_raw(4)?;
            f32::NEG_INFINITY
        }
        b'N' => {
            inp.read_raw(3)?;
            f32::NAN
        }
        _ => parse_ascii_float32(inp)?,
    };
    let trailer = inp.read_u8()?;
    if trailer == b' ' {
        Ok(val)
    } else {
        Err(SerializeError::Malformed("legacy float"))
    }
}

/// Minimal ASCII float parser mirroring what `in >> item` accepts for the old
/// dlib format: optional sign, digits with optional decimal point, optional
/// exponent.
fn parse_ascii_float64(inp: &mut Deserializer<'_>) -> Result<f64, SerializeError> {
    let s = scan_ascii_float(inp)?;
    s.parse::<f64>()
        .map_err(|_| SerializeError::Malformed("legacy float"))
}

fn parse_ascii_float32(inp: &mut Deserializer<'_>) -> Result<f32, SerializeError> {
    let s = scan_ascii_float(inp)?;
    s.parse::<f32>()
        .map_err(|_| SerializeError::Malformed("legacy float"))
}

fn scan_ascii_float(inp: &mut Deserializer<'_>) -> Result<String, SerializeError> {
    let mut s = String::new();
    // Optional sign.
    if let Some(b'-') | Some(b'+') = inp.peek_byte() {
        s.push(inp.read_u8()? as char);
    }
    // Digits and at most one '.'.
    let mut digits = 0usize;
    let mut dot = false;
    loop {
        match inp.peek_byte() {
            Some(b'0'..=b'9') => {
                s.push(inp.read_u8()? as char);
                digits += 1;
            }
            Some(b'.') if !dot => {
                dot = true;
                s.push(inp.read_u8()? as char);
            }
            _ => break,
        }
    }
    if digits == 0 {
        return Err(SerializeError::Malformed("legacy float"));
    }
    // Optional exponent.
    if let Some(b'e') | Some(b'E') = inp.peek_byte() {
        let save = inp.pos;
        let mut exp = String::from("e");
        inp.read_u8()?;
        if let Some(b'-') | Some(b'+') = inp.peek_byte() {
            exp.push(inp.read_u8()? as char);
        }
        let mut edigits = 0usize;
        while let Some(b'0'..=b'9') = inp.peek_byte() {
            exp.push(inp.read_u8()? as char);
            edigits += 1;
        }
        if edigits == 0 {
            inp.pos = save; // not an exponent after all
        } else {
            s.push_str(&exp);
        }
    }

    Ok(s)
}

// ----------------------------------------------------------------------------------------
// Serializer

/// Output sink producing dlib-compatible bytes (`serialize.h` write side).
#[derive(Debug, Default, Clone)]
pub struct Serializer {
    data: Vec<u8>,
}

impl Serializer {
    /// Creates an empty serializer.
    pub fn new() -> Self {
        Serializer { data: Vec::new() }
    }

    /// Consumes the serializer, returning the written bytes.
    pub fn into_inner(self) -> Vec<u8> {
        self.data
    }

    /// Returns the bytes written so far.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Writes a `bool` as ASCII `'1'`/`'0'` (`serialize(bool)` in `dlib/serialize.h`).
    pub fn write_bool(&mut self, v: bool) {
        self.data.push(if v { b'1' } else { b'0' });
    }

    /// Writes a raw byte (`serialize(unsigned char)`).
    pub fn write_u8(&mut self, v: u8) {
        self.data.push(v);
    }

    /// Writes a packed unsigned integer (`ser_helper::pack_int`).
    pub fn write_u16(&mut self, v: u16) {
        pack_uint(&mut self.data, v as u64);
    }

    /// Writes a packed unsigned integer (`ser_helper::pack_int`).
    pub fn write_u32(&mut self, v: u32) {
        pack_uint(&mut self.data, v as u64);
    }

    /// Writes a packed unsigned integer (`ser_helper::pack_int`).
    pub fn write_u64(&mut self, v: u64) {
        pack_uint(&mut self.data, v);
    }

    /// Writes a packed signed integer (`ser_helper::pack_int`).
    pub fn write_i16(&mut self, v: i16) {
        pack_int(&mut self.data, v as i64);
    }

    /// Writes a packed signed integer (`ser_helper::pack_int`).
    pub fn write_i32(&mut self, v: i32) {
        pack_int(&mut self.data, v as i64);
    }

    /// Writes a packed signed integer (`ser_helper::pack_int`).
    pub fn write_i64(&mut self, v: i64) {
        pack_int(&mut self.data, v);
    }

    /// Writes an `f32` via `float_details` (`serialize_floating_point`).
    pub fn write_f32(&mut self, v: f32) {
        let fd = FloatDetails::from_f32(v);
        self.write_i64(fd.mantissa);
        self.write_i16(fd.exponent);
    }

    /// Writes an `f64` via `float_details` (`serialize_floating_point`).
    pub fn write_f64(&mut self, v: f64) {
        let fd = FloatDetails::from_f64(v);
        self.write_i64(fd.mantissa);
        self.write_i16(fd.exponent);
    }

    /// Writes a `String`: packed u64 length + raw bytes (`serialize(std::string)`).
    pub fn write_string(&mut self, v: &str) {
        self.write_u64(v.len() as u64);
        self.data.extend_from_slice(v.as_bytes());
    }

    /// Writes a `std::vector<unsigned char>`: packed u64 length + raw bytes.
    pub fn write_byte_vec(&mut self, v: &[u8]) {
        self.write_u64(v.len() as u64);
        self.data.extend_from_slice(v);
    }

    /// Writes a `std::vector<T>`: packed u64 length + elements.
    pub fn write_vec<T: DlibSerialize>(&mut self, v: &[T]) {
        self.write_u64(v.len() as u64);
        for x in v {
            x.dlib_serialize(self);
        }
    }

    /// Appends raw bytes.
    pub fn write_raw(&mut self, bytes: &[u8]) {
        self.data.extend_from_slice(bytes);
    }
}

// ----------------------------------------------------------------------------------------
// Deserializer

/// Input reader for dlib-compatible bytes (`serialize.h` read side).
#[derive(Clone)]
pub struct Deserializer<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Deserializer<'a> {
    /// Creates a deserializer over `data`.
    pub fn new(data: &'a [u8]) -> Self {
        Deserializer { data, pos: 0 }
    }

    /// Number of unread bytes.
    pub fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    /// Peeks the next byte without consuming it.
    pub fn peek_byte(&self) -> Option<u8> {
        self.data.get(self.pos).copied()
    }

    /// Reads a `bool` (ASCII `'1'`/`'0'`).
    pub fn read_bool(&mut self) -> Result<bool, SerializeError> {
        match self.read_u8()? {
            b'1' => Ok(true),
            b'0' => Ok(false),
            _ => Err(SerializeError::Malformed("bool byte")),
        }
    }

    /// Reads a raw byte.
    pub fn read_u8(&mut self) -> Result<u8, SerializeError> {
        if self.pos >= self.data.len() {
            return Err(SerializeError::Eof);
        }
        let b = self.data[self.pos];
        self.pos += 1;
        Ok(b)
    }

    /// Reads a packed unsigned integer (must fit in `u16`).
    pub fn read_u16(&mut self) -> Result<u16, SerializeError> {
        unpack_uint(self, 2).map(|v| v as u16)
    }

    /// Reads a packed unsigned integer (must fit in `u32`).
    pub fn read_u32(&mut self) -> Result<u32, SerializeError> {
        unpack_uint(self, 4).map(|v| v as u32)
    }

    /// Reads a packed unsigned integer.
    pub fn read_u64(&mut self) -> Result<u64, SerializeError> {
        unpack_uint(self, 8)
    }

    /// Reads a packed signed integer (must fit in `i16`).
    pub fn read_i16(&mut self) -> Result<i16, SerializeError> {
        unpack_int(self, 2).map(|v| v as i16)
    }

    /// Reads a packed signed integer (must fit in `i32`).
    pub fn read_i32(&mut self) -> Result<i32, SerializeError> {
        unpack_int(self, 4).map(|v| v as i32)
    }

    /// Reads a packed signed integer.
    pub fn read_i64(&mut self) -> Result<i64, SerializeError> {
        unpack_int(self, 8)
    }

    /// Reads an `f32` via `float_details`, accepting the legacy ASCII format
    /// (`deserialize_floating_point` in `dlib/serialize.h`).
    pub fn read_f32(&mut self) -> Result<f32, SerializeError> {
        let first = self.peek_byte().ok_or(SerializeError::Eof)?;
        if (first & 0x70) == 0 {
            let mantissa = self.read_i64()?;
            let exponent = self.read_i16()?;
            Ok(FloatDetails::new(mantissa, exponent).to_f32())
        } else {
            old_deserialize_floating_point32(self)
        }
    }

    /// Reads an `f64` via `float_details`, accepting the legacy ASCII format
    /// (`deserialize_floating_point` in `dlib/serialize.h`).
    pub fn read_f64(&mut self) -> Result<f64, SerializeError> {
        let first = self.peek_byte().ok_or(SerializeError::Eof)?;
        if (first & 0x70) == 0 {
            let mantissa = self.read_i64()?;
            let exponent = self.read_i16()?;
            Ok(FloatDetails::new(mantissa, exponent).to_f64())
        } else {
            old_deserialize_floating_point64(self)
        }
    }

    /// Reads a `String`'s raw bytes (packed u64 length + bytes).
    pub fn read_string_bytes(&mut self) -> Result<Vec<u8>, SerializeError> {
        self.read_byte_vec()
    }

    /// Reads a UTF-8 `String`; invalid UTF-8 is `Malformed`.
    pub fn read_string(&mut self) -> Result<String, SerializeError> {
        let bytes = self.read_string_bytes()?;
        String::from_utf8(bytes).map_err(|_| SerializeError::Malformed("string utf8"))
    }

    /// Reads a `std::vector<unsigned char>` (packed u64 length + raw bytes).
    pub fn read_byte_vec(&mut self) -> Result<Vec<u8>, SerializeError> {
        let len = self.read_u64()? as usize;
        if len > self.remaining() {
            return Err(SerializeError::Eof);
        }
        Ok(self.read_raw(len)?.to_vec())
    }

    /// Reads a `std::vector<T>` (packed u64 length + elements).
    pub fn read_vec<T: DlibSerialize>(&mut self) -> Result<Vec<T>, SerializeError> {
        let len = self.read_u64()? as usize;
        let mut v = Vec::with_capacity(len.min(1024));
        for _ in 0..len {
            v.push(T::dlib_deserialize(self)?);
        }
        Ok(v)
    }

    /// Consumes and returns the next `n` bytes.
    pub fn read_raw(&mut self, n: usize) -> Result<&'a [u8], SerializeError> {
        if self.remaining() < n {
            return Err(SerializeError::Eof);
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
}

// ----------------------------------------------------------------------------------------
// DlibSerialize trait

/// Types that can be written to / read from dlib serialization streams.
///
/// Rust counterpart of the `serialize`/`deserialize` overload set in
/// `dlib/serialize.h`.
pub trait DlibSerialize: Sized {
    /// Writes `self` to `out`.
    fn dlib_serialize(&self, out: &mut Serializer);
    /// Reads a value of this type from `inp`.
    fn dlib_deserialize(inp: &mut Deserializer) -> Result<Self, SerializeError>;
}

impl DlibSerialize for bool {
    fn dlib_serialize(&self, out: &mut Serializer) {
        out.write_bool(*self);
    }
    fn dlib_deserialize(inp: &mut Deserializer) -> Result<Self, SerializeError> {
        inp.read_bool()
    }
}

impl DlibSerialize for u8 {
    fn dlib_serialize(&self, out: &mut Serializer) {
        out.write_u8(*self);
    }
    fn dlib_deserialize(inp: &mut Deserializer) -> Result<Self, SerializeError> {
        inp.read_u8()
    }
}

impl DlibSerialize for u16 {
    fn dlib_serialize(&self, out: &mut Serializer) {
        out.write_u16(*self);
    }
    fn dlib_deserialize(inp: &mut Deserializer) -> Result<Self, SerializeError> {
        inp.read_u16()
    }
}

impl DlibSerialize for u32 {
    fn dlib_serialize(&self, out: &mut Serializer) {
        out.write_u32(*self);
    }
    fn dlib_deserialize(inp: &mut Deserializer) -> Result<Self, SerializeError> {
        inp.read_u32()
    }
}

impl DlibSerialize for u64 {
    fn dlib_serialize(&self, out: &mut Serializer) {
        out.write_u64(*self);
    }
    fn dlib_deserialize(inp: &mut Deserializer) -> Result<Self, SerializeError> {
        inp.read_u64()
    }
}

impl DlibSerialize for i16 {
    fn dlib_serialize(&self, out: &mut Serializer) {
        out.write_i16(*self);
    }
    fn dlib_deserialize(inp: &mut Deserializer) -> Result<Self, SerializeError> {
        inp.read_i16()
    }
}

impl DlibSerialize for i32 {
    fn dlib_serialize(&self, out: &mut Serializer) {
        out.write_i32(*self);
    }
    fn dlib_deserialize(inp: &mut Deserializer) -> Result<Self, SerializeError> {
        inp.read_i32()
    }
}

impl DlibSerialize for i64 {
    fn dlib_serialize(&self, out: &mut Serializer) {
        out.write_i64(*self);
    }
    fn dlib_deserialize(inp: &mut Deserializer) -> Result<Self, SerializeError> {
        inp.read_i64()
    }
}

impl DlibSerialize for f32 {
    fn dlib_serialize(&self, out: &mut Serializer) {
        out.write_f32(*self);
    }
    fn dlib_deserialize(inp: &mut Deserializer) -> Result<Self, SerializeError> {
        inp.read_f32()
    }
}

impl DlibSerialize for f64 {
    fn dlib_serialize(&self, out: &mut Serializer) {
        out.write_f64(*self);
    }
    fn dlib_deserialize(inp: &mut Deserializer) -> Result<Self, SerializeError> {
        inp.read_f64()
    }
}

impl DlibSerialize for String {
    fn dlib_serialize(&self, out: &mut Serializer) {
        out.write_string(self);
    }
    fn dlib_deserialize(inp: &mut Deserializer) -> Result<Self, SerializeError> {
        inp.read_string()
    }
}

impl DlibSerialize for Vec<u8> {
    fn dlib_serialize(&self, out: &mut Serializer) {
        out.write_byte_vec(self);
    }
    fn dlib_deserialize(inp: &mut Deserializer) -> Result<Self, SerializeError> {
        inp.read_byte_vec()
    }
}

/// `std::vector<bool>` uses ASCII `'0'`/`'1'` bytes with a packed u64 length
/// (`serialize(std::vector<bool>)` in `dlib/serialize.h`).
impl DlibSerialize for Vec<bool> {
    fn dlib_serialize(&self, out: &mut Serializer) {
        out.write_u64(self.len() as u64);
        for &b in self {
            out.write_bool(b);
        }
    }
    fn dlib_deserialize(inp: &mut Deserializer) -> Result<Self, SerializeError> {
        let len = inp.read_u64()? as usize;
        let mut v = Vec::with_capacity(len.min(1024));
        for _ in 0..len {
            v.push(inp.read_bool()?);
        }
        Ok(v)
    }
}

impl fmt::Debug for Deserializer<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Deserializer")
            .field("pos", &self.pos)
            .field("len", &self.data.len())
            .finish()
    }
}

// ----------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ser_i64(v: i64) -> Vec<u8> {
        let mut s = Serializer::new();
        s.write_i64(v);
        s.into_inner()
    }

    fn roundtrip_i64(v: i64) {
        let bytes = ser_i64(v);
        let mut d = Deserializer::new(&bytes);
        assert_eq!(d.read_i64().unwrap(), v);
        assert_eq!(d.remaining(), 0);
    }

    #[test]
    fn packed_int_bytes() {
        assert_eq!(ser_i64(0), vec![0x01, 0x00]);
        assert_eq!(ser_i64(1), vec![0x01, 0x01]);
        assert_eq!(ser_i64(-1), vec![0x81, 0x01]);
        assert_eq!(ser_i64(127), vec![0x01, 0x7F]);
        assert_eq!(ser_i64(128), vec![0x01, 0x80]);
        assert_eq!(ser_i64(256), vec![0x02, 0x00, 0x01]);
        assert_eq!(ser_i64(-256), vec![0x82, 0x00, 0x01]);
        assert_eq!(
            ser_i64(i64::MIN),
            std::iter::once(0x88u8)
                .chain([0u8, 0, 0, 0, 0, 0, 0, 0x80])
                .collect::<Vec<_>>()
        );
        roundtrip_i64(i64::MIN);
        roundtrip_i64(i64::MAX);

        let mut s = Serializer::new();
        s.write_u64(0xFFFF_FFFF_FFFF_FFFF);
        assert_eq!(
            s.into_inner(),
            vec![0x08, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]
        );

        let mut s = Serializer::new();
        s.write_u32(0x1234_5678);
        assert_eq!(s.into_inner(), vec![0x04, 0x78, 0x56, 0x34, 0x12]);

        let mut s = Serializer::new();
        s.write_i32(-256);
        assert_eq!(s.into_inner(), vec![0x82, 0x00, 0x01]);
    }

    #[test]
    fn int_type_width_checks() {
        // Size byte larger than target type is malformed.
        let mut s = Serializer::new();
        s.write_u32(0x1234_5678);
        let mut d = Deserializer::new(s.as_bytes());
        assert!(matches!(d.read_u16(), Err(SerializeError::Malformed(_))));
        // Reserved-bit-masked control byte (0x8F mask semantics).
        let mut d = Deserializer::new(&[0x02, 0x01, 0x00][..]);
        assert_eq!(d.read_u64().unwrap(), 1);
        // 0x10..0x70 reserved bits set → masked out on unsigned read.
        let mut d = Deserializer::new(&[0x12, 0x01, 0x00]);
        assert_eq!(d.read_u64().unwrap(), 1);
    }

    #[test]
    fn bool_bytes() {
        let mut s = Serializer::new();
        s.write_bool(true);
        s.write_bool(false);
        assert_eq!(s.as_bytes(), b"10");
        let mut d = Deserializer::new(b"10");
        assert!(d.read_bool().unwrap());
        assert!(!d.read_bool().unwrap());
        let mut d = Deserializer::new(b"2");
        assert!(matches!(d.read_bool(), Err(SerializeError::Malformed(_))));
    }

    #[test]
    fn float_details_0_5() {
        // Compaction shifts 6 zero low bytes off 2^52: mantissa 16, exponent -53+48 = -5.
        let fd = FloatDetails::from_f64(0.5);
        assert_eq!(fd.mantissa, 16);
        assert_eq!(fd.exponent, -5);
        let mut s = Serializer::new();
        s.write_f64(0.5);
        assert_eq!(s.into_inner(), vec![0x01, 0x10, 0x81, 0x05]);
        assert_eq!(FloatDetails::new(16, -5).to_f64(), 0.5);
    }

    #[test]
    fn f64_roundtrip() {
        let vals = [
            0.0,
            -0.0, // dlib drops the sign of zero (mantissa 0, exponent 0)
            1.0,
            -1.0,
            0.5,
            1e-300,
            1e300,
            f64::MAX,
            f64::MIN_POSITIVE,
            5e-324,
            -5e-324,
            f64::EPSILON,
            std::f64::consts::PI,
            std::f64::consts::E,
            1234.5678,
        ];
        for v in vals {
            let mut s = Serializer::new();
            s.write_f64(v);
            let mut d = Deserializer::new(s.as_bytes());
            let back = d.read_f64().unwrap();
            assert_eq!(d.remaining(), 0);
            if v == 0.0 {
                assert_eq!(back.to_bits(), 0.0f64.to_bits());
            } else {
                assert_eq!(back.to_bits(), v.to_bits(), "value {v:e}");
            }
        }
        // -0.0 round-trips to +0.0, exactly like dlib's float_details.
        let mut s = Serializer::new();
        s.write_f64(-0.0);
        let mut d = Deserializer::new(s.as_bytes());
        assert_eq!(d.read_f64().unwrap().to_bits(), 0.0f64.to_bits());

        for v in [f64::INFINITY, f64::NEG_INFINITY] {
            let mut s = Serializer::new();
            s.write_f64(v);
            let mut d = Deserializer::new(s.as_bytes());
            assert_eq!(d.read_f64().unwrap(), v);
        }
        let mut s = Serializer::new();
        s.write_f64(f64::NAN);
        let mut d = Deserializer::new(s.as_bytes());
        assert!(d.read_f64().unwrap().is_nan());
    }

    #[test]
    fn f32_roundtrip() {
        let vals = [
            0.0f32,
            1.0,
            -1.0,
            0.5,
            1e-30,
            1e30,
            f32::MAX,
            f32::MIN_POSITIVE,
            1e-42, // subnormal f32
            f32::EPSILON,
            std::f32::consts::PI,
        ];
        for v in vals {
            let mut s = Serializer::new();
            s.write_f32(v);
            let mut d = Deserializer::new(s.as_bytes());
            assert_eq!(d.read_f32().unwrap().to_bits(), v.to_bits(), "value {v:e}");
            assert_eq!(d.remaining(), 0);
        }
        for v in [f32::INFINITY, f32::NEG_INFINITY] {
            let mut s = Serializer::new();
            s.write_f32(v);
            let mut d = Deserializer::new(s.as_bytes());
            assert_eq!(d.read_f32().unwrap(), v);
        }
        let mut s = Serializer::new();
        s.write_f32(f32::NAN);
        let mut d = Deserializer::new(s.as_bytes());
        assert!(d.read_f32().unwrap().is_nan());
    }

    #[test]
    fn cross_type_float_read() {
        // An f64-written value can be read as f32 (float_details is type-agnostic).
        let mut s = Serializer::new();
        s.write_f64(0.5);
        let mut d = Deserializer::new(s.as_bytes());
        assert_eq!(d.read_f32().unwrap(), 0.5f32);
    }

    #[test]
    fn legacy_ascii_floats() {
        let mut d = Deserializer::new(b"0.5 ");
        assert_eq!(d.read_f64().unwrap(), 0.5);
        assert_eq!(d.remaining(), 0);

        let mut d = Deserializer::new(b"inf ");
        assert_eq!(d.read_f64().unwrap(), f64::INFINITY);

        // dlib's old ASCII format writes -inf as "ninf" (peek 'n', 4 bytes).
        let mut d = Deserializer::new(b"ninf ");
        assert_eq!(d.read_f64().unwrap(), f64::NEG_INFINITY);

        // NaN is "Nan" (peek 'N', 3 bytes); "nan" would hit the 'n' branch.
        let mut d = Deserializer::new(b"Nan ");
        assert!(d.read_f64().unwrap().is_nan());

        let mut d = Deserializer::new(b"-1.25e3 ");
        assert_eq!(d.read_f64().unwrap(), -1250.0);

        let mut d = Deserializer::new(b"3.141592653589793 ");
        assert_eq!(d.read_f64().unwrap(), std::f64::consts::PI);

        // f32 reads of legacy data.
        let mut d = Deserializer::new(b"0.5 ");
        assert_eq!(d.read_f32().unwrap(), 0.5f32);
        let mut d = Deserializer::new(b"inf ");
        assert_eq!(d.read_f32().unwrap(), f32::INFINITY);

        // Missing trailing space → malformed.
        let mut d = Deserializer::new(b"0.5x");
        assert!(matches!(d.read_f64(), Err(SerializeError::Malformed(_))));
        // Garbage → malformed.
        let mut d = Deserializer::new(b"zz ");
        assert!(matches!(d.read_f64(), Err(SerializeError::Malformed(_))));
    }

    #[test]
    fn string_roundtrip() {
        for s in ["", "hello", "unicode: 你好 🌍"] {
            let mut ser = Serializer::new();
            ser.write_string(s);
            let mut d = Deserializer::new(ser.as_bytes());
            assert_eq!(d.read_string().unwrap(), s);
            assert_eq!(d.remaining(), 0);
        }
        // Layout: packed u64 length + raw bytes.
        let mut ser = Serializer::new();
        ser.write_string("hi");
        assert_eq!(ser.into_inner(), vec![0x01, 0x02, b'h', b'i']);
        // Invalid utf8 → Malformed.
        let mut ser = Serializer::new();
        ser.write_byte_vec(&[0xFF, 0xFE]);
        let mut d = Deserializer::new(ser.as_bytes());
        assert!(matches!(d.read_string(), Err(SerializeError::Malformed(_))));
        let mut d2 = Deserializer::new(ser.as_bytes());
        assert_eq!(d2.read_string_bytes().unwrap(), vec![0xFF, 0xFE]);
    }

    #[test]
    fn byte_vec_layout() {
        let mut ser = Serializer::new();
        ser.write_byte_vec(&[0xAA, 0xBB]);
        assert_eq!(ser.into_inner(), vec![0x01, 0x02, 0xAA, 0xBB]);

        let mut ser = Serializer::new();
        ser.write_byte_vec(&[]);
        assert_eq!(ser.into_inner(), vec![0x01, 0x00]);

        let mut ser = Serializer::new();
        ser.write_byte_vec(&[1, 2, 3]);
        let mut d = Deserializer::new(ser.as_bytes());
        assert_eq!(d.read_byte_vec().unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn vec_generic_roundtrip() {
        let v = vec![1i32, -2, 3];
        let mut ser = Serializer::new();
        ser.write_vec(&v);
        let mut d = Deserializer::new(ser.as_bytes());
        assert_eq!(d.read_vec::<i32>().unwrap(), v);

        let vs = vec!["a".to_string(), "bc".to_string()];
        let mut ser = Serializer::new();
        ser.write_vec(&vs);
        let mut d = Deserializer::new(ser.as_bytes());
        assert_eq!(d.read_vec::<String>().unwrap(), vs);

        // Vec<bool> uses ASCII bytes.
        let vb = vec![true, false, true];
        let mut ser = Serializer::new();
        vb.dlib_serialize(&mut ser);
        assert_eq!(ser.as_bytes(), &[0x01, 0x03, b'1', b'0', b'1']);
        let mut d = Deserializer::new(ser.as_bytes());
        assert_eq!(Vec::<bool>::dlib_deserialize(&mut d).unwrap(), vb);
    }

    #[test]
    fn malformed_inputs() {
        // Empty input → Eof.
        let mut d = Deserializer::new(&[]);
        assert!(matches!(d.read_i64(), Err(SerializeError::Eof)));
        assert!(matches!(d.read_f64(), Err(SerializeError::Eof)));

        // Control byte 0 → Malformed.
        let mut d = Deserializer::new(&[0x00, 0x00]);
        assert!(matches!(d.read_u64(), Err(SerializeError::Malformed(_))));

        // Truncated payload → Eof.
        let mut d = Deserializer::new(&[0x04, 0x01, 0x02]);
        assert!(matches!(d.read_u32(), Err(SerializeError::Eof)));

        // String length longer than remaining → Eof.
        let mut ser = Serializer::new();
        ser.write_u64(100);
        ser.write_raw(b"short");
        let mut d = Deserializer::new(ser.as_bytes());
        assert!(matches!(d.read_string_bytes(), Err(SerializeError::Eof)));
    }

    #[test]
    fn multiple_objects_stream() {
        let mut ser = Serializer::new();
        ser.write_i32(-7);
        ser.write_f64(2.5);
        ser.write_string("abc");
        ser.write_bool(true);
        let mut d = Deserializer::new(ser.as_bytes());
        assert_eq!(d.read_i32().unwrap(), -7);
        assert_eq!(d.read_f64().unwrap(), 2.5);
        assert_eq!(d.read_string().unwrap(), "abc");
        assert!(d.read_bool().unwrap());
        assert_eq!(d.remaining(), 0);
    }
}
