//! Port of `dlib/image_transforms/draw.h`: `draw_line`, `draw_rectangle`,
//! `fill_rect` and `draw_solid_circle`.
//!
//! All routines are generic over any [`GenericImage`] and assign pixels with
//! dlib's `assign_pixel` semantics (including the alpha blending dlib uses for
//! diagonal lines and the anti-aliased solid circle).

use dlib_rs_core::geometry::{grow_rect, shrink_rect, Dpoint, Point, Rectangle};

use crate::array2d::GenericImage;
use crate::pixel::{assign_pixel, Pixel, RgbAlphaPixel};

fn get_rect<I: GenericImage>(img: &I) -> Rectangle {
    // dlib's get_rect(): rectangle(0, 0, nc-1, nr-1); empty when the image is empty.
    Rectangle::new(
        0,
        0,
        img.num_columns() as i64 - 1,
        img.num_rows() as i64 - 1,
    )
}

/// Port of `dlib::draw_pixel`-style single pixel assignment (dlib draws pixels
/// through `assign_pixel(img[r][c], val)` everywhere in `draw.h`).
///
/// Panics when the coordinate is outside the image, mirroring dlib's checked
/// array access semantics.
pub fn draw_pixel<I, P>(img: &mut I, r: i64, c: i64, val: &P)
where
    I: GenericImage,
    I::PixelType: Pixel,
    P: Pixel,
{
    assert!(
        r >= 0 && r < img.num_rows() as i64 && c >= 0 && c < img.num_columns() as i64,
        "draw_pixel: coordinate ({r},{c}) is outside the image"
    );
    assign_pixel(img.pixel_mut(r as usize, c as usize), val);
}

/// Port of `dlib::draw_line(x1, y1, x2, y2, img, val)` from `draw.h`.
///
/// Straight lines are drawn with plain pixel assignment; diagonal lines are
/// alpha blended exactly like dlib does (each stepped pixel is written twice
/// with complementary alpha weights derived from the sub-pixel position).
#[allow(clippy::too_many_arguments)]
pub fn draw_line<I, P>(img: &mut I, x1: i64, y1: i64, x2: i64, y2: i64, val: &P)
where
    I: GenericImage,
    I::PixelType: Pixel,
    P: Pixel,
{
    if x1 == x2 {
        // make sure y1 comes before y2
        let (y1, y2) = if y1 > y2 { (y2, y1) } else { (y1, y2) };

        if x1 < 0 || x1 >= img.num_columns() as i64 {
            return;
        }

        // this is a vertical line
        for y in y1..=y2 {
            if y < 0 || y >= img.num_rows() as i64 {
                continue;
            }
            assign_pixel(img.pixel_mut(y as usize, x1 as usize), val);
        }
    } else if y1 == y2 {
        // make sure x1 comes before x2
        let (x1, x2) = if x1 > x2 { (x2, x1) } else { (x1, x2) };

        if y1 < 0 || y1 >= img.num_rows() as i64 {
            return;
        }

        // this is a horizontal line
        for x in x1..=x2 {
            if x < 0 || x >= img.num_columns() as i64 {
                continue;
            }
            assign_pixel(img.pixel_mut(y1 as usize, x as usize), val);
        }
    } else {
        // This part is a little more complicated because we are going to
        // perform alpha blending so the diagonal lines look nice.
        let valid_area = get_rect(img);
        let mut alpha_pixel = RgbAlphaPixel::default();
        assign_pixel(&mut alpha_pixel, val);
        let max_alpha = alpha_pixel.a;

        let rise = y2 - y1;
        let run = x2 - x1;
        if rise.abs() < run.abs() {
            let slope = rise as f64 / run as f64;

            let (first, last) = if x1 > x2 {
                (
                    x2.max(valid_area.left()) as f64,
                    x1.min(valid_area.right()) as f64,
                )
            } else {
                (
                    x1.max(valid_area.left()) as f64,
                    x2.min(valid_area.right()) as f64,
                )
            };

            let x1f = x1 as f64;
            let y1f = y1 as f64;
            let mut i = first;
            while i <= last {
                let dy = slope * (i - x1f) + y1f;
                let dx = i;

                let y = dy as i64;
                let x = dx as i64;

                if y >= valid_area.top() && y <= valid_area.bottom() {
                    alpha_pixel.a = ((1.0 - (dy - y as f64)) * max_alpha as f64) as u8;
                    assign_pixel(img.pixel_mut(y as usize, x as usize), &alpha_pixel);
                }
                if y + 1 >= valid_area.top() && y < valid_area.bottom() {
                    alpha_pixel.a = ((dy - y as f64) * max_alpha as f64) as u8;
                    assign_pixel(img.pixel_mut((y + 1) as usize, x as usize), &alpha_pixel);
                }
                i += 1.0;
            }
        } else {
            let slope = run as f64 / rise as f64;

            let (first, last) = if y1 > y2 {
                (
                    y2.max(valid_area.top()) as f64,
                    y1.min(valid_area.bottom()) as f64,
                )
            } else {
                (
                    y1.max(valid_area.top()) as f64,
                    y2.min(valid_area.bottom()) as f64,
                )
            };

            let x1f = x1 as f64;
            let y1f = y1 as f64;
            let mut i = first;
            while i <= last {
                let dx = slope * (i - y1f) + x1f;
                let dy = i;

                let y = dy as i64;
                let x = dx as i64;

                if x >= valid_area.left() && x <= valid_area.right() {
                    alpha_pixel.a = ((1.0 - (dx - x as f64)) * max_alpha as f64) as u8;
                    assign_pixel(img.pixel_mut(y as usize, x as usize), &alpha_pixel);
                }
                if x + 1 >= valid_area.left() && x < valid_area.right() {
                    alpha_pixel.a = ((dx - x as f64) * max_alpha as f64) as u8;
                    assign_pixel(img.pixel_mut(y as usize, (x + 1) as usize), &alpha_pixel);
                }
                i += 1.0;
            }
        }
    }
}

/// Port of `dlib::draw_line(img, p1, p2, val)`.
pub fn draw_line_points<I, P>(img: &mut I, p1: &Point, p2: &Point, val: &P)
where
    I: GenericImage,
    I::PixelType: Pixel,
    P: Pixel,
{
    draw_line(img, p1.x(), p1.y(), p2.x(), p2.y(), val);
}

/// Port of `dlib::draw_rectangle(img, rect, val)`: draws the four outline
/// edges of `rect` via `draw_line`, clipped to the image bounds.
pub fn draw_rectangle<I, P>(img: &mut I, rect: &Rectangle, val: &P)
where
    I: GenericImage,
    I::PixelType: Pixel,
    P: Pixel,
{
    draw_line_points(img, &rect.tl_corner(), &rect.tr_corner(), val);
    draw_line_points(img, &rect.bl_corner(), &rect.br_corner(), val);
    draw_line_points(img, &rect.tl_corner(), &rect.bl_corner(), val);
    draw_line_points(img, &rect.tr_corner(), &rect.br_corner(), val);
}

/// Port of `dlib::draw_rectangle(img, rect, val, thickness)`: alternately
/// shrinks/grows the rectangle like dlib does (`(i+1)/2` offset per ring).
pub fn draw_rectangle_thickness<I, P>(img: &mut I, rect: &Rectangle, val: &P, thickness: u32)
where
    I: GenericImage,
    I::PixelType: Pixel,
    P: Pixel,
{
    for i in 0..thickness {
        if i % 2 == 0 {
            // (i + 1) / 2 == i.div_ceil(2)
            draw_rectangle(img, &shrink_rect(rect, i.div_ceil(2) as i64), val);
        } else {
            draw_rectangle(img, &grow_rect(rect, i.div_ceil(2) as i64), val);
        }
    }
}

/// Port of `dlib::fill_rect(img, rect, pixel)`: assigns `pixel` to every image
/// pixel inside `rect` intersected with the image bounds.
pub fn fill_rect<I, P>(img: &mut I, rect: &Rectangle, val: &P)
where
    I: GenericImage,
    I::PixelType: Pixel,
    P: Pixel,
{
    let area = rect.intersect(&get_rect(img));
    if area.is_empty() {
        return;
    }
    for r in area.top()..=area.bottom() {
        for c in area.left()..=area.right() {
            assign_pixel(img.pixel_mut(r as usize, c as usize), val);
        }
    }
}

/// dlib's `std::lround` semantics: round half away from zero.
fn lround(v: f64) -> i64 {
    v.round() as i64
}

/// Port of `dlib::draw_solid_circle(img, center_point, radius, pixel)` from
/// `draw.h`: draws the filled disk scanline by scanline through vertical
/// `draw_line` calls, exactly following dlib's two half-loops; sub-pixel
/// circles are alpha blended in proportion to their size.
pub fn draw_solid_circle<I, P>(img: &mut I, center_point: &Dpoint, radius: f64, val: &P)
where
    I: GenericImage,
    I::PixelType: Pixel,
    P: Pixel,
{
    let valid_area = get_rect(img);
    let x = center_point.x();
    let y = center_point.y();
    let cp = Point::new(x as i64, y as i64);
    if radius > 1.0 {
        let mut first_x = (x - radius + 0.5) as i64;
        let mut last_x = (x + radius + 0.5) as i64;
        let rs = radius * radius;

        // ensure that we only loop over the part of the x dimension that this
        // image contains.
        if first_x < valid_area.left() {
            first_x = valid_area.left();
        }
        if last_x > valid_area.right() {
            last_x = valid_area.right();
        }

        let mut top;
        let mut bottom;

        top = lround(
            (rs - (first_x as f64 - x - 0.5) * (first_x as f64 - x - 0.5))
                .max(0.0)
                .sqrt(),
        );
        top += y as i64;
        let mut last = top;

        // draw the left half of the circle
        let middle = (cp.x() - 1).min(last_x);
        for i in first_x..=middle {
            let a = i as f64 - x + 0.5;
            // find the top of the arc
            top = lround((rs - a * a).max(0.0).sqrt());
            top += y as i64;
            let temp = top;

            while top >= last {
                bottom = y as i64 - top + y as i64;
                draw_line_points(img, &Point::new(i, top), &Point::new(i, bottom), val);
                top -= 1;
            }

            last = temp;
        }

        let middle = cp.x().max(first_x);
        top = lround(
            (rs - (last_x as f64 - x + 0.5) * (last_x as f64 - x + 0.5))
                .max(0.0)
                .sqrt(),
        );
        top += y as i64;
        last = top;
        // draw the right half of the circle
        let mut i = last_x;
        while i >= middle {
            let a = i as f64 - x - 0.5;
            // find the top of the arc
            top = lround((rs - a * a).max(0.0).sqrt());
            top += y as i64;
            let temp = top;

            while top >= last {
                bottom = y as i64 - top + y as i64;
                draw_line_points(img, &Point::new(i, top), &Point::new(i, bottom), val);
                top -= 1;
            }

            last = temp;
            i -= 1;
        }
    } else if valid_area.contains(&cp) {
        // For circles smaller than a pixel we will just alpha blend them in
        // proportion to how small they are.
        let mut temp = RgbAlphaPixel::default();
        assign_pixel(&mut temp, val);
        temp.a = (255.0 * radius + 0.5) as u8;
        assign_pixel(img.pixel_mut(cp.y() as usize, cp.x() as usize), &temp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array2d::Array2D;

    #[test]
    fn draw_line_hits_endpoints() {
        let mut img: Array2D<u8> = Array2D::zeros(5, 5);
        draw_line(&mut img, 0, 0, 4, 0, &9u8);
        for c in 0..5 {
            assert_eq!(img[(0, c)], 9);
        }
        draw_line(&mut img, 2, 0, 2, 4, &7u8);
        for r in 0..5 {
            assert_eq!(img[(r, 2)], 7);
        }
        // diagonal line touches both endpoints
        let mut img: Array2D<u8> = Array2D::zeros(5, 5);
        draw_line(&mut img, 0, 0, 4, 4, &200u8);
        assert_eq!(img[(0, 0)], 200);
        assert_eq!(img[(4, 4)], 200);
        // clipping: nothing outside bounds panics or writes
        draw_line(&mut img, -3, -3, 20, 20, &5u8);
        assert_eq!(img[(0, 0)], 5); // overwritten by out-of-bounds-ish diagonal start
    }

    #[test]
    fn fill_rect_covers_exact_pixel_set() {
        let mut img: Array2D<u8> = Array2D::zeros(10, 10);
        // 6 wide x 5 tall rectangle at (2,1)
        fill_rect(&mut img, &Rectangle::new(2, 1, 7, 5), &42u8);
        let mut expected = Array2D::<u8>::zeros(10, 10);
        for r in 1..=5 {
            for c in 2..=7 {
                expected[(r, c)] = 42;
            }
        }
        for r in 0..10 {
            for c in 0..10 {
                assert_eq!(img[(r, c)], expected[(r, c)], "mismatch at ({r},{c})");
            }
        }
        // clipped fill stays inside the image
        let mut img2: Array2D<u8> = Array2D::zeros(4, 4);
        fill_rect(&mut img2, &Rectangle::new(-5, -5, 20, 20), &1u8);
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(img2[(r, c)], 1);
            }
        }
    }

    #[test]
    fn draw_rectangle_outline_sets_border_only() {
        let mut img: Array2D<u8> = Array2D::zeros(8, 8);
        draw_rectangle(&mut img, &Rectangle::new(1, 1, 5, 4), &3u8);
        for r in 0..8 {
            for c in 0..8 {
                let on_border = (r == 1 || r == 4) && (1..=5).contains(&c)
                    || (c == 1 || c == 5) && (1..=4).contains(&r);
                assert_eq!(img[(r, c)], if on_border { 3 } else { 0 }, "at ({r},{c})");
            }
        }
        // thickness 2 adds the adjacent ring
        let mut img2: Array2D<u8> = Array2D::zeros(8, 8);
        draw_rectangle_thickness(&mut img2, &Rectangle::new(2, 2, 5, 5), &1u8, 2);
        assert_eq!(img2[(2, 2)], 1);
        assert_eq!(img2[(1, 1)], 1); // grown ring
        assert_eq!(img2[(3, 3)], 0); // interior untouched
    }

    #[test]
    fn draw_solid_circle_radius3_symmetric() {
        let mut img: Array2D<u8> = Array2D::zeros(11, 11);
        draw_solid_circle(&mut img, &Dpoint::new(5.0, 5.0), 3.0, &77u8);
        let set = |r: usize, c: usize| img[(r, c)] == 77;
        // symmetric under both mirror axes
        for r in 0..11 {
            for c in 0..11 {
                assert_eq!(set(r, c), set(r, 10 - c), "col mirror at ({r},{c})");
                assert_eq!(set(r, c), set(10 - r, c), "row mirror at ({r},{c})");
            }
        }
        // center and extremes of the vertical span are filled
        assert!(set(5, 5));
        assert!(set(2, 5) && set(8, 5));
        assert!(set(2, 3) && set(2, 7) && set(8, 3) && set(8, 7));
        // outside the disk
        assert!(!set(1, 5) && !set(9, 5));
        assert!(!set(5, 1) && !set(5, 9));
        assert!(!set(2, 2) && !set(8, 8));
    }

    #[test]
    fn draw_pixel_assigns_and_panics_out_of_bounds() {
        let mut img: Array2D<u8> = Array2D::zeros(3, 3);
        draw_pixel(&mut img, 1, 2, &99u8);
        assert_eq!(img[(1, 2)], 99);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut img: Array2D<u8> = Array2D::zeros(3, 3);
            draw_pixel(&mut img, 3, 0, &1u8);
        }));
        assert!(result.is_err());
    }
}
