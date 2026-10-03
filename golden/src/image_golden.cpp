// Golden outputs for the dlib-rs image layer: deterministic input image, then
// FHOG features, resize/pyramid/rotate/flip results, equalize_hist,
// threshold, sobel edges, jet colormap, SURF points, and hex dumps of dlib's
#include <dlib/image_transforms/interpolation.h>
#include <dlib/array2d.h>
#include <dlib/array.h>
#include <dlib/image_transforms/fhog.h>
#include <dlib/image_transforms/interpolation.h>
#include <dlib/image_transforms/equalize_histogram.h>
#include <dlib/image_transforms/thresholding.h>
#include <dlib/image_transforms/edge_detector.h>
#include <dlib/image_transforms/assign_image.h>
#include <dlib/image_transforms/colormaps.h>
#include <dlib/image_keypoint/surf.h>
#include <dlib/image_saver/image_saver.h>
#include <dlib/pixel.h>
#include <dlib/matrix.h>
#include <cstdio>
#include <sstream>
#include <iomanip>

using namespace dlib;

static void make_img(array2d<rgb_pixel>& img) {
    img.set_size(64, 64);
    for (long r = 0; r < img.nr(); ++r)
        for (long c = 0; c < img.nc(); ++c) {
            img[r][c].red = (unsigned char)(r * 4);
            img[r][c].green = (unsigned char)(c * 4);
            img[r][c].blue = (unsigned char)((r + c) * 2 % 256);
        }
}

static void make_gray(array2d<unsigned char>& img) {
    img.set_size(64, 64);
    for (long r = 0; r < img.nr(); ++r)
        for (long c = 0; c < img.nc(); ++c)
            img[r][c] = (unsigned char)((r * 3 + c * 5) % 256);
}

template <typename T>
static void print_val(const char* name, const T& v) {
    std::printf("%s %.17g\n", name, (double)v);
}

int main() {
    array2d<rgb_pixel> img;
    make_img(img);
    array2d<unsigned char> gray;
    make_gray(gray);

    // ---- FHOG (planar form) ----
    {
        dlib::array<array2d<double> > hog;
        extract_fhog_features(img, hog, 8);
        std::printf("FHOG_dims %zu %ld %ld\n", hog.size(), hog[0].nr(), hog[0].nc());
        for (int ch = 0; ch < 31; ++ch)
            for (long r = 0; r < hog[ch].nr(); ++r)
                for (long c = 0; c < hog[ch].nc(); ++c)
                    std::printf("FHOG %d %ld %ld %.17g\n", ch, r, c, hog[ch][r][c]);
    }

    // ---- resize_image ----
    {
        array2d<rgb_pixel> out(50, 100);
        resize_image(img, out);
        for (long r = 0; r < out.nr(); ++r)
            for (long c = 0; c < out.nc(); ++c)
                std::printf("RESIZE %ld %ld %d %d %d\n", r, c,
                            (int)out[r][c].red, (int)out[r][c].green, (int)out[r][c].blue);
    }

    // ---- pyramid_up (covers scaled growth) ----
    {
        array2d<rgb_pixel> out;
        pyramid_up(img, out, pyramid_down<6>());
        std::printf("PYRAMID_UP_dims %ld %ld\n", out.nr(), out.nc());
        for (long r = 0; r < out.nr(); ++r)
            for (long c = 0; c < out.nc(); ++c)
                std::printf("PYRAMID_UP %ld %ld %d %d %d\n", r, c,
                            (int)out[r][c].red, (int)out[r][c].green, (int)out[r][c].blue);
    }


    // ---- pyramid_down<6> ----
    {
        pyramid_down<6> pyr;
        array2d<rgb_pixel> out;
        pyr(img, out);
        std::printf("PYRAMID_dims %ld %ld\n", out.nr(), out.nc());
        for (long r = 0; r < out.nr(); ++r)
            for (long c = 0; c < out.nc(); ++c)
                std::printf("PYRAMID %ld %ld %d %d %d\n", r, c,
                            (int)out[r][c].red, (int)out[r][c].green, (int)out[r][c].blue);
    }

    // ---- rotate / flip ----
    {
        array2d<rgb_pixel> out;
        rotate_image(img, out, 30.0 * 3.141592653589793238462643383279502884 / 180.0);
        for (long r = 0; r < out.nr(); ++r)
            for (long c = 0; c < out.nc(); ++c)
                std::printf("ROTATE %ld %ld %d %d %d\n", r, c,
                            (int)out[r][c].red, (int)out[r][c].green, (int)out[r][c].blue);
        array2d<rgb_pixel> f;
        flip_image_left_right(img, f);
        for (long r = 0; r < f.nr(); ++r)
            for (long c = 0; c < f.nc(); ++c)
                std::printf("FLIPLR %ld %ld %d %d %d\n", r, c,
                            (int)f[r][c].red, (int)f[r][c].green, (int)f[r][c].blue);
    }

    // ---- equalize_hist / threshold / sobel / jet on gray ----
    {
        array2d<unsigned char> eq;
        equalize_histogram(gray, eq);
        for (long r = 0; r < eq.nr(); ++r)
            for (long c = 0; c < eq.nc(); ++c)
                std::printf("EQUALIZE %ld %ld %d\n", r, c, (int)eq[r][c]);

        array2d<unsigned char> thr;
        threshold_image(gray, thr, 100.0);
        for (long r = 0; r < thr.nr(); ++r)
            for (long c = 0; c < thr.nc(); ++c)
                std::printf("THRESH %ld %ld %d\n", r, c, (int)thr[r][c]);

        array2d<double> gx, gy;
        sobel_edge_detector(gray, gx, gy);
        for (long r = 0; r < gx.nr(); ++r)
            for (long c = 0; c < gx.nc(); ++c)
                std::printf("SOBEL %ld %ld %.17g %.17g\n", r, c, gx[r][c], gy[r][c]);

        array2d<rgb_pixel> jm;
        assign_image(jm, jet(gray));
        for (long r = 0; r < jm.nr(); ++r)
            for (long c = 0; c < jm.nc(); ++c)
                std::printf("JET %ld %ld %d %d %d\n", r, c,
                            (int)jm[r][c].red, (int)jm[r][c].green, (int)jm[r][c].blue);
    }

    // ---- SURF on gradient image ----
    {
        std::vector<surf_point> points = get_surf_points(gray);
        std::printf("SURF_count %zu\n", points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            const surf_point& p = points[i];
            std::printf("SURF %zu %.17g %.17g %.17g %.17g %.17g %.17g",
                        i, p.p.center.x(), p.p.center.y(), p.p.scale, p.angle, p.p.score,
                        p.p.laplacian);
            for (long j = 0; j < p.des.size(); ++j)
                std::printf(" %.17g", p.des(j));
            std::printf("\n");
        }
    }

    // ---- SURF on blob image (deterministic corners/blobs) ----
    {
        array2d<unsigned char> blob(96, 96);
        for (long r = 0; r < blob.nr(); ++r)
            for (long c = 0; c < blob.nc(); ++c)
                blob[r][c] = (unsigned char)((r * 2 + c * 3) % 97);
        for (long r = 28; r < 44; ++r)
            for (long c = 30; c < 48; ++c)
                blob[r][c] = 240;
        for (long r = 60; r < 78; ++r)
            for (long c = 56; c < 70; ++c)
                blob[r][c] = 10;
        std::vector<surf_point> points = get_surf_points(blob);
        std::printf("SURF_BLOB_count %zu\n", points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            const surf_point& p = points[i];
            std::printf("SURFB %zu %.17g %.17g %.17g %.17g %.17g %.17g",
                        i, p.p.center.x(), p.p.center.y(), p.p.scale, p.angle, p.p.score,
                        p.p.laplacian);
            for (long j = 0; j < p.des.size(); ++j)
                std::printf(" %.17g", p.des(j));
            std::printf("\n");
        }
    }

    // ---- BMP encoder hex (16x16 gray) ----
    {
        array2d<unsigned char> small(16, 16);
        for (long r = 0; r < 16; ++r)
            for (long c = 0; c < 16; ++c)
                small[r][c] = (unsigned char)(r * 16 + c);
        std::ostringstream sout;
        save_bmp(small, sout);
        const std::string bytes = sout.str();
        std::printf("BMP_hex_len %zu\n", bytes.size());
        std::printf("BMP_hex [");
        for (unsigned char ch : bytes) std::printf("%02x", ch);
        std::printf("]\n");
    }

    // ---- DNG encoder hex (16x16 gray and 16x16 rgb) ----
    {
        array2d<unsigned char> small(16, 16);
        for (long r = 0; r < 16; ++r)
            for (long c = 0; c < 16; ++c)
                small[r][c] = (unsigned char)(r * 16 + c);
        std::ostringstream sout;
        save_dng(small, sout);
        const std::string bytes = sout.str();
        std::printf("DNG_gray_hex_len %zu\n", bytes.size());
        std::printf("DNG_gray_hex [");
        for (unsigned char ch : bytes) std::printf("%02x", ch);
        std::printf("]\n");
    }
    {
        array2d<rgb_pixel> small(16, 16);
        for (long r = 0; r < 16; ++r)
            for (long c = 0; c < 16; ++c) {
                small[r][c].red = (unsigned char)(r * 16 + c);
                small[r][c].green = (unsigned char)(255 - r * 16 - c);
                small[r][c].blue = (unsigned char)((r * 8 + c * 4) % 256);
            }
        std::ostringstream sout;
        save_dng(small, sout);
        const std::string bytes = sout.str();
        std::printf("DNG_rgb_hex_len %zu\n", bytes.size());
        std::printf("DNG_rgb_hex [");
        for (unsigned char ch : bytes) std::printf("%02x", ch);
        std::printf("]\n");
    }

    return 0;
}
