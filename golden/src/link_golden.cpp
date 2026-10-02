// Smoke-checks that the golden generators compile against dlib headers and
// link against the local static libdlib.a (save_bmp is a compiled symbol).
#include <dlib/array2d.h>
#include <dlib/image_saver/image_saver.h>
#include <iostream>

int main() {
    dlib::array2d<unsigned char> img(2, 2);
    img[0][0] = 10;
    img[0][1] = 20;
    img[1][0] = 30;
    img[1][1] = 40;
    dlib::save_bmp(img, "golden_smoke_out.bmp");
    std::cout << "golden_link_ok\n";
    return 0;
}
