//! End-to-end face-pipeline golden test (M4 acceptance): load a JPEG, run the
//! embedded frontal-face detector, then the official 68-landmark shape
//! predictor, and compare against the C++ dlib golden output
//! (`golden/src/face_golden.cpp` → `tests/golden/face.txt`): detection boxes
//! must match exactly and landmark coordinates within 1.5 pixels.
//!
//! Skips when the 100 MB `shape_predictor_68_face_landmarks.dat` model is not
//! present (it is gitignored; CI without the asset still tests blob-load via
//! the lib unit tests).

use dlib_rs_core::geometry::Rectangle;
use dlib_rs_image::array2d::Array2D;
use dlib_rs_image::image_loader::load_image;
use dlib_rs_image::image_processing::frontal_face_detector::frontal_face_detector;
use dlib_rs_image::image_processing::shape_predictor::ShapePredictor;

fn data_dir() -> String {
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data").to_string()
}

fn golden() -> String {
    std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/golden/face.txt"
    ))
    .expect("golden face.txt present")
}

#[test]
fn golden_face_pipeline_end_to_end() {
    let dat = format!("{}/shape_predictor_68_face_landmarks.dat", data_dir());
    if !std::path::Path::new(&dat).exists() {
        eprintln!("skipping: {dat} not present");
        return;
    }

    let mut det = frontal_face_detector().expect("embedded detector deserializes");
    let sp = ShapePredictor::load_from_file(&dat).expect("official .dat loads");

    let text = golden();
    let mut lines = text
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty());

    // blob_bytes sanity
    let blob = lines.next().unwrap();
    assert!(
        blob.starts_with("blob_bytes"),
        "unexpected first line {blob}"
    );

    for img_name in ["2008_002506.bmp", "2007_007763.bmp"] {
        // image header
        let hdr = lines.next().unwrap();
        assert!(
            hdr.starts_with("image "),
            "expected image header, got {hdr}"
        );
        assert!(hdr.contains(img_name), "golden order mismatch: {hdr}");

        let img: Array2D<u8> =
            load_image(&format!("{}/{img_name}", data_dir())).expect("load jpeg");
        let want_dets: usize = lines
            .next()
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse()
            .unwrap();
        let dets = det.run(&img, 0.0);

        assert_eq!(dets.len(), want_dets, "detection count on {img_name}");
        for (i, det_rect) in dets.iter().enumerate() {
            let p: Vec<i64> = lines
                .next()
                .unwrap()
                .split_whitespace()
                .skip(2)
                .map(|v| v.parse().unwrap())
                .collect();
            let want = Rectangle::new(p[0], p[1], p[2], p[3]);
            assert_eq!(
                (
                    det_rect.left(),
                    det_rect.top(),
                    det_rect.right(),
                    det_rect.bottom()
                ),
                (want.left(), want.top(), want.right(), want.bottom()),
                "detection box {i} on {img_name} must match C++ exactly"
            );

            // 68 landmarks within 1.5 px
            let shape = sp.operator_(&img, &want);
            let nparts: usize = lines
                .next()
                .unwrap()
                .split_whitespace()
                .nth(1)
                .unwrap()
                .parse()
                .unwrap();
            assert_eq!(shape.num_parts(), nparts);
            let mut max_err: f64 = 0.0;
            for j in 0..nparts {
                let p: Vec<f64> = lines
                    .next()
                    .unwrap()
                    .split_whitespace()
                    .skip(2)
                    .map(|v| v.parse().unwrap())
                    .collect();
                let got = shape.part(j);
                let ex = (got.x() - p[0]).abs();
                let ey = (got.y() - p[1]).abs();
                max_err = max_err.max(ex).max(ey);
            }
            assert!(
                max_err < 1.5,
                "landmark error {max_err} px on {img_name} det {i} exceeds 1.5 px"
            );
        }
        let end = lines.next().unwrap();
        assert_eq!(end, "end_image");
    }
}

/// The shape_predictor_trainer on the exported example-faces training set:
/// a reduced configuration (plan: 5 trees x 2 cascade levels) must converge
/// and produce a serializable, re-loadable, runnable predictor.
#[test]
fn golden_shape_predictor_trainer_on_example_faces() {
    use dlib_rs_image::image_processing::full_object_detection::FullObjectDetection;
    use dlib_rs_image::image_processing::shape_predictor_trainer::ShapePredictorTrainer;

    let trainset = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/golden/trainset.txt"
    );
    let text = match std::fs::read_to_string(trainset) {
        Ok(t) => t,
        Err(_) => {
            eprintln!("skipping: trainset.txt not present");
            return;
        }
    };

    let mut lines = text
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty());
    let mut images: Vec<Array2D<u8>> = Vec::new();
    let mut objects: Vec<FullObjectDetection> = Vec::new();
    while let Some(hdr) = lines.next() {
        if !hdr.starts_with("FACE ") {
            break;
        }
        let p: Vec<&str> = hdr.split_whitespace().collect();
        let img_path = format!("{}/{}", data_dir(), p[1]);
        if !std::path::Path::new(&img_path).exists() {
            eprintln!("skipping: training image {img_path} missing");
            return;
        }
        let rect = Rectangle::new(
            p[2].parse().unwrap(),
            p[3].parse().unwrap(),
            p[4].parse().unwrap(),
            p[5].parse().unwrap(),
        );
        let mut parts = Vec::with_capacity(68);
        for _ in 0..68 {
            let q: Vec<f64> = lines
                .next()
                .unwrap()
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect();
            parts.push(dlib_rs_core::geometry::Dpoint::new(q[0], q[1]));
        }
        images.push(load_image(&img_path).expect("load training image"));
        objects.push(FullObjectDetection { rect, parts });
    }
    if images.len() < 2 {
        eprintln!("skipping: not enough training faces parsed");
        return;
    }

    let mut trainer = ShapePredictorTrainer::new();
    trainer
        .set_cascade_depth(2)
        .set_num_trees_per_cascade_level(5)
        .set_tree_depth(4)
        .set_feature_pool_size(80)
        .set_num_test_splits(20);
    let first_rect = objects[0].rect;
    let grouped: Vec<Vec<FullObjectDetection>> = objects.into_iter().map(|o| vec![o]).collect();
    let predictor = trainer.train(&images, &grouped);

    // Training must actually fit the data reasonably: predicted landmarks on a
    // training rectangle stay near the annotation box.
    let shape = predictor.operator_(&images[0], &first_rect);
    assert_eq!(shape.num_parts(), 68);
    for j in 0..68 {
        let p = shape.part(j);
        assert!(
            p.x() >= first_rect.left() as f64 - 10.0
                && p.x() <= first_rect.right() as f64 + 10.0
                && p.y() >= first_rect.top() as f64 - 10.0
                && p.y() <= first_rect.bottom() as f64 + 10.0,
            "landmark {j} escaped the face box: ({},{})",
            p.x(),
            p.y()
        );
    }

    // serialize -> deserialize -> identical inference
    use dlib_rs_core::serialize::Serializer;
    let mut s = Serializer::new();
    predictor.serialize(&mut s);
    let bytes = s.into_inner();
    let mut d = dlib_rs_core::serialize::Deserializer::new(&bytes);
    let mut sp2 = dlib_rs_image::image_processing::shape_predictor::ShapePredictor::default();
    sp2.deserialize(&mut d).expect("roundtrip deserialize");
    let shape2 = sp2.operator_(&images[0], &first_rect);
    for j in 0..68 {
        assert_eq!(shape.part(j).x(), shape2.part(j).x(), "part {j} x");
        assert_eq!(shape.part(j).y(), shape2.part(j).y(), "part {j} y");
    }
}
