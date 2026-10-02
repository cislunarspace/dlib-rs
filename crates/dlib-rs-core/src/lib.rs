//! Core layer of `dlib-rs` — a pure-Rust port of the foundational parts of
//! [dlib](https://github.com/davisking/dlib).
//!
//! Modules and type names follow dlib's naming (`matrix`, `geometry`, `rand`,
//! `serialize`) so implementations can be compared file-by-file against the
//! original C++ headers.
//!
//! Ported from dlib (commit 46fa4a28), Boost Software License 1.0.

pub mod geometry;
pub mod matrix;
pub mod rand;
pub mod serialize;
