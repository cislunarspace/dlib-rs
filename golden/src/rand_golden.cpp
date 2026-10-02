// Golden outputs for the dlib-rs rand port: bit-exact draw sequences for
// fixed seeds, plus a hex dump of the serialized generator state mid-stream.
#include <dlib/rand/rand_kernel_1.h>
#include <dlib/serialize.h>
#include <cstdio>
#include <sstream>
#include <iomanip>

using namespace dlib;

static void doubles(const char* seed, int n) {
    dlib::rand rnd(seed);
    std::printf("doubles [%s]\n", seed);
    for (int i = 0; i < n; ++i)
        std::printf("%.17g\n", rnd.get_random_double());
}

static void gaussians(const char* seed, int n) {
    dlib::rand rnd(seed);
    std::printf("gaussians [%s]\n", seed);
    for (int i = 0; i < n; ++i)
        std::printf("%.17g\n", rnd.get_random_gaussian());
}

int main() {
    doubles("", 10000);
    gaussians("", 10000);
    doubles("42", 1000);
    gaussians("42", 1000);
    doubles("seed123", 1000);
    gaussians("seed123", 1000);

    {
        dlib::rand rnd("42");
        std::printf("u32 [42]\n");
        for (int i = 0; i < 100; ++i)
            std::printf("%u\n", rnd.get_random_32bit_number());
        std::printf("u64 [42]\n");
        for (int i = 0; i < 50; ++i)
            std::printf("%llu\n", (unsigned long long)rnd.get_random_64bit_number());
        std::printf("u16 [42]\n");
        for (int i = 0; i < 50; ++i)
            std::printf("%u\n", (unsigned)rnd.get_random_16bit_number());
        std::printf("u8 [42]\n");
        for (int i = 0; i < 50; ++i)
            std::printf("%u\n", (unsigned)rnd.get_random_8bit_number());
        std::printf("float [42]\n");
        for (int i = 0; i < 1000; ++i)
            std::printf("%.9g\n", rnd.get_random_float());
        std::printf("in_range [42]\n");
        for (int i = 0; i < 100; ++i)
            std::printf("%lld\n", rnd.get_integer_in_range(5, 23));
        std::printf("double_in_range [42]\n");
        for (int i = 0; i < 100; ++i)
            std::printf("%.17g\n", rnd.get_double_in_range(-2.5, 7.5));
        std::printf("exponential [42]\n");
        for (int i = 0; i < 100; ++i)
            std::printf("%.17g\n", rnd.get_random_exponential(1.7));
        std::printf("weibull [42]\n");
        for (int i = 0; i < 100; ++i)
            std::printf("%.17g\n", rnd.get_random_weibull(2.2, 1.3, 0.4));
        std::printf("beta [42]\n");
        for (int i = 0; i < 100; ++i)
            std::printf("%.17g\n", rnd.get_random_beta(2.5, 3.5));
    }

    // Serialized state after 1234 doubles from seed "42".
    {
        dlib::rand rnd("42");
        for (int i = 0; i < 1234; ++i)
            rnd.get_random_double();
        std::ostringstream sout;
        serialize(rnd, sout);
        const std::string bytes = sout.str();
        std::printf("state_hex_len %zu\n", bytes.size());
        std::printf("state_hex [");
        for (unsigned char ch : bytes)
            std::printf("%02x", ch);
        std::printf("]\n");
    }

    // State after a gaussian draw leaves a cached value (has_gaussian = true).
    {
        dlib::rand rnd("seed123");
        rnd.get_random_gaussian();
        std::ostringstream sout;
        serialize(rnd, sout);
        const std::string bytes = sout.str();
        std::printf("gauss_state_hex_len %zu\n", bytes.size());
        std::printf("gauss_state_hex [");
        for (unsigned char ch : bytes)
            std::printf("%02x", ch);
        std::printf("]\n");
    }

    return 0;
}
