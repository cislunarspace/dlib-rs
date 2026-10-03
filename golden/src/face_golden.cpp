// Golden outputs for the dlib-rs face pipeline:
//  1. dumps the embedded frontal-face-detector blob (byte-identity target),
//  2. runs detection + shape_predictor on two fixed test images,
//  3. dumps a small parsed training set (from dlib's example XML) for the
//     Rust shape_predictor_trainer test.
// argv[1] = golden dir, argv[2] = dlib examples faces dir, argv[3] = .dat path.
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/data_io/image_dataset_metadata.h>
#include <dlib/serialize.h>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

using namespace dlib;

static void detect_and_predict(
    const char* path,
    frontal_face_detector& detector,
    const shape_predictor& sp) {
    array2d<unsigned char> img;
    load_image(img, path);
    std::printf("image %s %ld %ld\n", path, img.nr(), img.nc());
    std::vector<rectangle> dets = detector(img);
    std::printf("dets %zu\n", dets.size());
    for (size_t i = 0; i < dets.size(); ++i) {
        std::printf("det %zu %ld %ld %ld %ld\n", i,
                    dets[i].left(), dets[i].top(), dets[i].right(), dets[i].bottom());
        full_object_detection shape = sp(img, dets[i]);
        std::printf("parts %zu\n", shape.num_parts());
        for (unsigned long j = 0; j < shape.num_parts(); ++j)
            std::printf("part %zu %.17g %.17g\n", j, shape.part(j).x(), shape.part(j).y());
    }
    std::printf("end_image\n");
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: face_golden <golden_dir> <faces_dir> <dat_path>\n");
        return 2;
    }
    const std::string golden_dir = argv[1];
    const std::string faces_dir = argv[2];
    const std::string dat_path = argv[3];

    // 1. embedded detector blob
    frontal_face_detector detector = get_frontal_face_detector();
    {
        const std::string blob = get_serialized_frontal_faces();
        std::ofstream out(golden_dir + "/frontal_faces.dat", std::ios::binary);
        out.write(blob.data(), (std::streamsize)blob.size());
        std::printf("blob_bytes %zu\n", blob.size());
    }

    // 2. detection + landmarks on two fixed images
    shape_predictor sp;
    {
        std::ifstream in(dat_path, std::ios::binary);
        deserialize(sp, in);
    }
    detect_and_predict((faces_dir + "/2008_002506.bmp").c_str(), detector, sp);
    detect_and_predict((faces_dir + "/2007_007763.bmp").c_str(), detector, sp);

    // 3. parsed training set: first 8 fully-annotated faces
    {
        image_dataset_metadata::dataset data;
        image_dataset_metadata::load_image_dataset_metadata(
            data, faces_dir + "/training_with_face_landmarks.xml");
        std::ofstream out(golden_dir + "/trainset.txt");
        int dumped = 0;
        for (unsigned long i = 0; i < data.images.size() && dumped < 8; ++i) {
            const auto& im = data.images[i];
            for (unsigned long f = 0; f < im.boxes.size() && dumped < 8; ++f) {
                const auto& box = im.boxes[f];
                if (box.parts.size() != 68) continue;
                if (box.rect.is_empty()) continue;
                std::vector<dlib::point> pts(68);
                bool ok = true;
                for (const auto& kv : box.parts) {
                    long idx = -1;
                    try {
                        idx = std::stol(kv.first);
                    } catch (...) {
                        ok = false;
                        break;
                    }
                    if (idx < 0 || idx >= 68) {
                        ok = false;
                        break;
                    }
                    pts[idx] = kv.second;
                }
                if (!ok) continue;
                out << "FACE " << im.filename << " " << box.rect.left() << " " << box.rect.top()
                    << " " << box.rect.right() << " " << box.rect.bottom() << "\n";
                for (unsigned long p = 0; p < 68; ++p)
                    out << pts[p].x() << " " << pts[p].y() << "\n";
                ++dumped;
            }
        }
        std::printf("train_faces %d\n", dumped);
    }

    return 0;
}
