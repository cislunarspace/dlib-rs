# dlib-rs

A pure-Rust port of the core of [dlib](https://github.com/davisking/dlib)
(tested against commit `46fa4a28`, dlib 20.0). No C/C++ dependency; CPU-only
with `rayon` parallelism.

Ported from dlib, which is distributed under the Boost Software License 1.0;
this crate keeps the same license. All credit for the original algorithms and
implementations goes to [Davis King](https://github.com/davisking) and the
dlib contributors.

## Scope

| dlib module | dlib-rs crate | Status |
|---|---|---|
| `matrix` (linear algebra, LU/Cholesky/QR/eig/SVD) | `dlib-rs-core` | ported |
| `geometry` (vector, rectangle, transforms, polygon) | `dlib-rs-core` | ported |
| `rand` (MT19937, bit-compatible with dlib) | `dlib-rs-core` | ported |
| `serialize` (dlib binary format, `.dat` model loading) | `dlib-rs-core` | ported |
| `image_loader`/`image_saver` (BMP, DNG, PNG, JPEG, WebP) | `dlib-rs-image` | ported |
| `image_transforms` (resize, pyramids, FHOG, edge detectors, drawing) | `dlib-rs-image` | ported |
| `image_keypoint` (FHOG, SURF) | `dlib-rs-image` | ported |
| `image_processing` (`frontal_face_detector`, `shape_predictor`) | `dlib-rs-image` | ported |
| `optimization` (BFGS, L-BFGS, CG, BOBYQA, line search) | `dlib-rs-ml` | ported |
| `svm` (kernels, SMO trainers, decision functions) | `dlib-rs-ml` | ported |
| clustering (`chinese_whispers`, kmeans, `bottom_up_cluster`, `spectral_cluster`) | `dlib-rs-ml` | ported |
| `statistics` (`running_stats`, correlations, …) | `dlib-rs-ml` | ported |

Out of scope for now: DNN, face recognition (ResNet), GUI, networking,
SSE/AVX hand-tuning (portable scalar code today), Python bindings.

## Platform support

| Target | CI |
|---|---|
| `x86_64-unknown-linux-gnu` | build + test |
| `aarch64-unknown-linux-gnu` | build + test |
| `x86_64-pc-windows-msvc` | build + test |
| `aarch64-pc-windows-msvc` | `cargo check` |

## Loading dlib models

```rust
use dlib_rs::image::{frontal_face_detector, shape_predictor};

let detector = frontal_face_detector()?;
let predictor = shape_predictor::ShapePredictor::load_from_file(
    "shape_predictor_68_face_landmarks.dat")?;
```

## Layout

- `crates/dlib-rs-core` — matrix, geometry, rand, serialize
- `crates/dlib-rs-image` — pixels, codecs, transforms, keypoints, detectors
- `crates/dlib-rs-ml` — optimizers, SVM, clustering, statistics
- `crates/dlib-rs` — umbrella crate re-exporting the three layers
- `golden/` — C++ golden-output generators used to verify numerical parity
  with dlib during development (not part of the published crates)

## License

Boost Software License 1.0 — see `LICENSE.txt`.
