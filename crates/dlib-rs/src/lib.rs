//! `dlib-rs` — pure-Rust port of the core of
//! [dlib](https://github.com/davisking/dlib): matrix linear algebra, geometry,
//! image processing (FHOG face detection, shape predictor landmarks) and
//! classic machine learning (BFGS/L-BFGS/CG/BOBYQA optimizers, SMO SVMs,
//! clustering, statistics).
//!
//! This umbrella crate re-exports the three layer crates; see each crate's
//! docs for details. Ported from dlib (commit 46fa4a28), Boost Software
//! License 1.0.

pub use dlib_rs_core as core;
pub use dlib_rs_image as image;
pub use dlib_rs_ml as ml;
