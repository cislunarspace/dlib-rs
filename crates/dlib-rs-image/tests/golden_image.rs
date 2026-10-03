//! Golden comparison against C++ dlib image-layer output
//! (`golden/src/image_golden.cpp` → `tests/golden/image.txt`).

use dlib_rs_image::array2d::Array2D;
use dlib_rs_image::image_keypoint::{extract_fhog_features, get_surf_points};
use dlib_rs_image::image_loader::dng::save_dng;
use dlib_rs_image::image_saver::save_bmp;
use dlib_rs_image::image_transforms::colormaps::jet;
use dlib_rs_image::image_transforms::edge_detector::sobel_edge_detector;
use dlib_rs_image::image_transforms::equalize_hist::equalize_hist;
use dlib_rs_image::image_transforms::interpolation::{pyramid_up_with, PyramidDown};
use dlib_rs_image::image_transforms::resize_image::resize_image;
use dlib_rs_image::image_transforms::rotation::{flip_image_left_right, rotate_image};
use dlib_rs_image::image_transforms::threshold_image::threshold_image;
use dlib_rs_image::pixel::RgbPixel;

fn load() -> String {
    std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/golden/image.txt"
    ))
    .expect("golden image.txt present")
}

fn make_img() -> Array2D<RgbPixel> {
    let mut img = Array2D::zeros(64, 64);
    for r in 0..64usize {
        for c in 0..64usize {
            img[(r, c)] = RgbPixel {
                r: (r * 4) as u8,
                g: (c * 4) as u8,
                b: ((r + c) * 2 % 256) as u8,
            };
        }
    }
    img
}

fn make_gray() -> Array2D<u8> {
    let mut img = Array2D::zeros(64, 64);
    for r in 0..64usize {
        for c in 0..64usize {
            img[(r, c)] = ((r * 3 + c * 5) % 256) as u8;
        }
    }
    img
}

struct Lines {
    lines: Vec<String>,
    idx: usize,
}

impl Lines {
    fn new() -> Self {
        Lines {
            lines: load()
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect(),
            idx: 0,
        }
    }
    fn next(&mut self) -> Vec<String> {
        let l = self.lines[self.idx].clone();
        self.idx += 1;
        l.split_whitespace().map(|s| s.to_string()).collect()
    }
    fn expect_header(&mut self, prefix: &str) -> Vec<String> {
        loop {
            let parts = self.next();
            if parts.first().map(|s| s.as_str()) == Some(prefix) {
                return parts;
            }
            assert!(self.idx < self.lines.len() + 1, "never found {prefix}");
            if self.idx >= self.lines.len() {
                panic!("never found header {prefix}");
            }
        }
    }
}

fn rel_close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
}

#[test]
fn golden_image_transforms_and_keypoints() {
    let img = make_img();
    let gray = make_gray();
    let mut g = Lines::new();

    // ---- FHOG ----
    let parts = g.expect_header("FHOG_dims");
    assert_eq!(parts[1].parse::<usize>().unwrap(), 31);
    let gnr = parts[2].parse::<usize>().unwrap();
    let gnc = parts[3].parse::<usize>().unwrap();
    let hog = extract_fhog_features(&img, 8);
    assert_eq!((hog.nr, hog.nc), (gnr, gnc), "FHOG dims");
    let mut expect: Vec<(usize, usize, usize, f64)> = Vec::new();
    for _ in 0..31 * gnr * gnc {
        let p = g.next();
        expect.push((
            p[1].parse().unwrap(),
            p[2].parse().unwrap(),
            p[3].parse().unwrap(),
            p[4].parse().unwrap(),
        ));
    }
    for (ch, r, c, want) in expect {
        let got = hog.at(r, c, ch) as f64;
        assert!(
            rel_close(got, want, 1e-5),
            "FHOG[{ch}][{r}][{c}]: got {got:.9e} want {want:.9e}"
        );
    }

    // ---- resize_image (rgb 64x64 -> 50x100; no dims line in golden) ----
    let mut out = Array2D::zeros(50, 100);
    resize_image(&img, 50, 100, &mut out);
    assert_eq!((out.nr(), out.nc()), (50, 100));
    check_rgb_block(&mut g, "RESIZE", &out);

    // ---- pyramid_up with pyramid_down<6> (77x77) ----
    let parts = g.expect_header("PYRAMID_UP_dims");
    let pnr = parts[1].parse::<usize>().unwrap();
    let pnc = parts[2].parse::<usize>().unwrap();
    let mut out = Array2D::<RgbPixel>::new();
    pyramid_up_with(&img, &mut out, &PyramidDown::<6>);
    assert_eq!((out.nr(), out.nc()), (pnr, pnc), "pyramid_up dims");
    check_rgb_block(&mut g, "PYRAMID_UP", &out);

    // ---- pyramid_down<6> (53x53) ----
    let parts = g.expect_header("PYRAMID_dims");
    let pnr = parts[1].parse::<usize>().unwrap();
    let pnc = parts[2].parse::<usize>().unwrap();
    let mut out = Array2D::<RgbPixel>::new();
    PyramidDown::<6>.apply(&img, &mut out);
    assert_eq!((out.nr(), out.nc()), (pnr, pnc), "pyramid_down dims");
    check_rgb_block(&mut g, "PYRAMID", &out);

    // ---- rotate 30deg ----
    let mut out = Array2D::<RgbPixel>::new();
    rotate_image(&img, 30.0_f64.to_radians(), &mut out);
    check_rgb_block(&mut g, "ROTATE", &out);

    // ---- flip left right ----
    let mut out = Array2D::<RgbPixel>::new();
    flip_image_left_right(&img, &mut out);
    check_rgb_block(&mut g, "FLIPLR", &out);

    // ---- equalize / threshold / sobel / jet ----
    let mut eq = Array2D::<u8>::new();
    equalize_hist(&gray, &mut eq);
    check_u8_block(&mut g, "EQUALIZE", &eq);

    let mut thr = Array2D::<u8>::new();
    threshold_image(&gray, &mut thr, 100.0);
    check_u8_block(&mut g, "THRESH", &thr);

    let (gx, gy) = sobel_edge_detector(&gray);
    let n = gx.nr() * gx.nc();
    for _ in 0..n {
        let p = g.expect_header("SOBEL");
        let r = p[1].parse::<usize>().unwrap();
        let c = p[2].parse::<usize>().unwrap();
        let wx = p[3].parse::<f64>().unwrap();
        let wy = p[4].parse::<f64>().unwrap();
        assert!(rel_close(gx[(r, c)], wx, 1e-12), "SOBEL gx[{r}][{c}]");
        assert!(rel_close(gy[(r, c)], wy, 1e-12), "SOBEL gy[{r}][{c}]");
    }

    let mut jm = Array2D::<RgbPixel>::new();
    jet(&gray, &mut jm);
    check_rgb_block(&mut g, "JET", &jm);

    // ---- SURF on gradient (0 points) and on blob image ----
    let parts = g.expect_header("SURF_count");
    let cnt = parts[1].parse::<usize>().unwrap();
    let pts = get_surf_points(&gray);
    assert_eq!(pts.len(), cnt, "SURF count on gradient");
    for _ in 0..cnt {
        let p = g.expect_header("SURF");
        check_surf_line(&p, &pts);
    }

    let parts = g.expect_header("SURF_BLOB_count");
    let cnt = parts[1].parse::<usize>().unwrap();
    let mut blob = Array2D::zeros(96, 96);
    for r in 0..96usize {
        for c in 0..96usize {
            blob[(r, c)] = ((r * 2 + c * 3) % 97) as u8;
        }
    }
    for r in 28..44usize {
        for c in 30..48usize {
            blob[(r, c)] = 240;
        }
    }
    for r in 60..78usize {
        for c in 56..70usize {
            blob[(r, c)] = 10;
        }
    }
    let pts = get_surf_points(&blob);
    assert_eq!(pts.len(), cnt, "SURF blob count");
    for _ in 0..cnt {
        let p = g.expect_header("SURFB");
        check_surf_line(&p, &pts);
    }

    // ---- BMP hex parity ----
    let parts = g.expect_header("BMP_hex_len");
    let want_len = parts[1].parse::<usize>().unwrap();
    let parts = g.expect_header("BMP_hex");
    let want_hex = parts[1].trim_matches('[').trim_end_matches(']').to_string();
    let mut small = Array2D::zeros(16, 16);
    for r in 0..16usize {
        for c in 0..16usize {
            small[(r, c)] = (r * 16 + c) as u8;
        }
    }
    let mut buf = Vec::new();
    save_bmp(&small, &mut buf).unwrap();
    assert_eq!(buf.len(), want_len, "BMP stream length");
    let got: String = buf.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(got, want_hex, "BMP bytes differ from C++ dlib save_bmp");

    // ---- DNG hex parity (gray + rgb) ----
    let parts = g.expect_header("DNG_gray_hex_len");
    let want_len = parts[1].parse::<usize>().unwrap();
    let parts = g.expect_header("DNG_gray_hex");
    let want_hex = parts[1].trim_matches('[').trim_end_matches(']').to_string();
    let mut buf = Vec::new();
    save_dng(&small, &mut buf).unwrap();
    assert_eq!(buf.len(), want_len, "DNG gray stream length");
    let got: String = buf.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(got, want_hex, "DNG gray bytes differ from C++ save_dng");

    let parts = g.expect_header("DNG_rgb_hex_len");
    let want_len = parts[1].parse::<usize>().unwrap();
    let parts = g.expect_header("DNG_rgb_hex");
    let want_hex = parts[1].trim_matches('[').trim_end_matches(']').to_string();
    let mut small = Array2D::zeros(16, 16);
    for r in 0..16usize {
        for c in 0..16usize {
            small[(r, c)] = RgbPixel {
                r: (r * 16 + c) as u8,
                g: (255 - r * 16 - c) as u8,
                b: ((r * 8 + c * 4) % 256) as u8,
            };
        }
    }
    let mut buf = Vec::new();
    save_dng(&small, &mut buf).unwrap();
    assert_eq!(buf.len(), want_len, "DNG rgb stream length");
    let got: String = buf.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(got, want_hex, "DNG rgb bytes differ from C++ save_dng");

    assert_eq!(g.idx, g.lines.len(), "all golden image sections consumed");
}

fn check_rgb_block(g: &mut Lines, tag: &str, img: &Array2D<RgbPixel>) {
    let n = img.nr() * img.nc();
    for _ in 0..n {
        let p = g.expect_header(tag);
        let r = p[1].parse::<usize>().unwrap();
        let c = p[2].parse::<usize>().unwrap();
        assert_eq!(
            p[3].parse::<u32>().unwrap(),
            img[(r, c)].r as u32,
            "{tag} r{r}c{c}"
        );
        assert_eq!(
            p[4].parse::<u32>().unwrap(),
            img[(r, c)].g as u32,
            "{tag} r{r}c{c}"
        );
        assert_eq!(
            p[5].parse::<u32>().unwrap(),
            img[(r, c)].b as u32,
            "{tag} r{r}c{c}"
        );
    }
}

fn check_u8_block(g: &mut Lines, tag: &str, img: &Array2D<u8>) {
    let n = img.nr() * img.nc();
    for _ in 0..n {
        let p = g.expect_header(tag);
        let r = p[1].parse::<usize>().unwrap();
        let c = p[2].parse::<usize>().unwrap();
        assert_eq!(
            p[3].parse::<u32>().unwrap(),
            img[(r, c)] as u32,
            "{tag} r{r}c{c}"
        );
    }
}

fn check_surf_line(p: &[String], pts: &[dlib_rs_image::image_keypoint::surf::SurfPoint]) {
    let i = p[1].parse::<usize>().unwrap();
    let pt = &pts[i];
    let vals: Vec<f64> = p[2..8].iter().map(|s| s.parse().unwrap()).collect();
    assert!(
        rel_close(pt.center.x(), vals[0], 1e-9),
        "SURF center.x #{}",
        i
    );
    assert!(
        rel_close(pt.center.y(), vals[1], 1e-9),
        "SURF center.y #{}",
        i
    );
    assert!(rel_close(pt.scale, vals[2], 1e-9), "SURF scale #{}", i);
    assert!(rel_close(pt.angle, vals[3], 1e-9), "SURF angle #{}", i);
    assert!(
        rel_close(pt.response, vals[4], 1e-9),
        "SURF response #{}",
        i
    );
    assert!(
        rel_close(pt.laplacian as f64, vals[5], 1e-9),
        "SURF laplacian #{}",
        i
    );
    for (j, w) in p[8..].iter().enumerate() {
        let w: f64 = w.parse().unwrap();
        assert!(rel_close(pt.vector[j], w, 1e-9), "SURF vec[{j}] #{}", i);
    }
}
