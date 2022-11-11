#include "cpu.hpp"

game::game(int width, int height, int *state, int step) {
    this->width = width;
    this->height = height;
    this->pitch = step;
    this->frame_counter = 0;
    for (int i = 0; i < width * height; i++) {
        this->buf1.push_back(state[i]);
    }
    this->buf2 = std::vector<int>(width * height);
}

bool game::step() {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            int neighbor = 0;
            int self = buf1[i + j * width];
            for (int lx = -1; lx < 2; lx++) {
                for (int ly = -1; ly < 2; ly++) {
                    neighbor += buf1[(i + lx + width) % width +
                                     ((j + ly + height) % height) * width] >= 1
                                    ? 1
                                    : 0;
                }
            }
            if (self >= 1) {
                buf2[i + j * width] =
                    neighbor == 3 || neighbor == 4 ? self + 1 : 0;
            } else {
                buf2[i + j * width] = neighbor == 3 ? 1 : 0;
            }
        }
    }
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            buf1[i + j * width] = buf2[i + j * width];
        }
    }
}

bool game::render(unsigned int *p, int pitch) {
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            p[i + j * width] = 0;
            if (buf1[i + j * width] == 0)
                continue;
            // float h = buf1[i + j * width] % 360;
            // unsigned int d = 0;
            // if (0 <= h && h < 60) {
            //     d = (255 << 16) + ((unsigned int)(h / 60 * 255) << 8);
            // } else if (60 <= h && h < 120) {
            //     d = ((unsigned int)((120 - h) / 60 * 255) << 16) + (255 <<
            //     8);
            // } else if (120 <= h && h < 180) {
            //     d = (255 << 8) + ((unsigned int)((h - 120) / 60 * 255));
            // } else if (180 <= h && h < 240) {
            //     d = ((unsigned int)((240 - h) / 60 * 255) << 8) + (255);
            // } else if (240 <= h && h < 300) {
            //     d = ((unsigned int)((h - 240) / 60 * 255) << 16) + (255);
            // } else if (300 <= h && h < 360) {
            //     d = (255 << 16) + (((unsigned int)(360 - h) / 60 * 255));
            // }
            // p[i + j * width] = d << 8;
        }
    }
    return true;
}