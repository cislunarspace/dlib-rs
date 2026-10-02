// Golden binary output for the dlib-rs serialize port: serializes a battery
// of values with dlib::serialize and writes the raw bytes to stdout. The Rust
// test suite compares its own byte stream against this file exactly, and
// deserializes it to check the parsed values.
//
// Value order (must be mirrored by the Rust test):
//   bool true, bool false,
//   uint8 0, 255,
//   uint16 65535, uint32 0xDEADBEEFu, uint64 0xFFFF'FFFF'FFFF'FFFF,
//   int16 -12345, int32 -12345678, int64 -12345678901234,
//   float 0.5f, float 3.14159f, float -2.5e-30f,
//   double 0.0, -0.0, 1.0, -1.0, 0.5, 3.141592653589793, 1e-300, 1e300,
//          5e-324, infinity, -infinity,
//   string "", "hello",
//   vector<unsigned char> {0xAA, 0xBB, 0x00, 0xFF},
//   vector<int32> {-1, 0, 1, 65536, -65536},
//   vector<double> {0.25, -0.25, 1e100},
//   vector<bool> {true, false, true},
//   matrix<double> 2x3 with (r,c) = 0.1*(r+1)+(c+1),
//   matrix<float> 2x2 with (r,c) = 0.5f*(r+c),
//   matrix<int32> 2x2 with (r,c) = 10*r+c  (serialized as long elements)
#include <dlib/matrix.h>
#include <dlib/serialize.h>
#include <cstdint>
#include <vector>
#include <iostream>

using namespace dlib;

int main() {
    std::ostringstream sout;

    bool b = true;
    serialize(b, sout);
    b = false;
    serialize(b, sout);

    uint8_t u8 = 0;
    serialize(u8, sout);
    u8 = 255;
    serialize(u8, sout);

    uint16_t u16 = 65535;
    serialize(u16, sout);
    uint32_t u32 = 0xDEADBEEFu;
    serialize(u32, sout);
    uint64_t u64 = 0xFFFFFFFFFFFFFFFFull;
    serialize(u64, sout);

    int16_t i16 = -12345;
    serialize(i16, sout);
    int32_t i32 = -12345678;
    serialize(i32, sout);
    int64_t i64 = -12345678901234LL;
    serialize(i64, sout);

    float f = 0.5f;
    serialize(f, sout);
    f = 3.14159f;
    serialize(f, sout);
    f = -2.5e-30f;
    serialize(f, sout);

    double d = 0.0;
    serialize(d, sout);
    d = -0.0;
    serialize(d, sout);
    d = 1.0;
    serialize(d, sout);
    d = -1.0;
    serialize(d, sout);
    d = 0.5;
    serialize(d, sout);
    d = 3.141592653589793;
    serialize(d, sout);
    d = 1e-300;
    serialize(d, sout);
    d = 1e300;
    serialize(d, sout);
    d = 5e-324;
    serialize(d, sout);
    d = std::numeric_limits<double>::infinity();
    serialize(d, sout);
    d = -std::numeric_limits<double>::infinity();
    serialize(d, sout);

    std::string s = "";
    serialize(s, sout);
    s = "hello";
    serialize(s, sout);

    std::vector<unsigned char> vuc = {0xAA, 0xBB, 0x00, 0xFF};
    serialize(vuc, sout);

    std::vector<int32_t> vi = {-1, 0, 1, 65536, -65536};
    serialize(vi, sout);

    std::vector<double> vd = {0.25, -0.25, 1e100};
    serialize(vd, sout);

    std::vector<bool> vb = {true, false, true};
    serialize(vb, sout);

    matrix<double> md(2, 3);
    for (long r = 0; r < 2; ++r)
        for (long c = 0; c < 3; ++c)
            md(r, c) = 0.1 * (r + 1) + (c + 1);
    serialize(md, sout);

    matrix<float> mf(2, 2);
    for (long r = 0; r < 2; ++r)
        for (long c = 0; c < 2; ++c)
            mf(r, c) = 0.5f * (r + c);
    serialize(mf, sout);

    matrix<long> mi(2, 2);
    for (long r = 0; r < 2; ++r)
        for (long c = 0; c < 2; ++c)
            mi(r, c) = 10 * r + c;
    serialize(mi, sout);

    const std::string bytes = sout.str();
    std::cout.write(bytes.data(), bytes.size());
    return 0;
}
