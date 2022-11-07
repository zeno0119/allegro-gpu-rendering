#include "kernel.cuh"
#include <allegro5/allegro5.h>
#include <allegro5/allegro_font.h>
#include <allegro5/allegro_primitives.h>
#include <chrono>
#include <iostream>
#include <stdlib.h>

int main() {
    int width = 1024;
    int height = 1024;
    if (!al_init()) {
        printf("couldn't initialize allegro\n");
        return 1;
    }

    if (!al_init_primitives_addon()) {
        printf("couldn't initialize allegro primitives\n");
    }

    if (!al_install_keyboard() || !al_install_mouse()) {
        printf("couldn't initialize keyboard\n");
        return 1;
    }

    ALLEGRO_TIMER *timer = al_create_timer(1.0 / 144.0);
    if (!timer) {
        printf("couldn't initialize timer\n");
        return 1;
    }

    ALLEGRO_EVENT_QUEUE *queue = al_create_event_queue();
    if (!queue) {
        printf("couldn't initialize queue\n");
        return 1;
    }

    ALLEGRO_DISPLAY *disp = al_create_display(width, height);
    if (!disp) {
        printf("couldn't initialize display\n");
        return 1;
    }

    ALLEGRO_FONT *font = al_create_builtin_font();
    if (!font) {
        printf("couldn't initialize font\n");
        return 1;
    }

    al_register_event_source(queue, al_get_keyboard_event_source());
    al_register_event_source(queue, al_get_display_event_source(disp));
    al_register_event_source(queue, al_get_timer_event_source(timer));

    std::cout << "first end" << std::endl;
    int time = 0;
    int frame_counter = 0;

    auto m = al_create_bitmap(width, height);
    int *map = (int *)malloc(sizeof(int) * width * height);
    for (int i = 0; i < width * height; i++) {
        map[i] = 0;
        if (i % width == 0 && i < width * 128) {
            map[i] = 1;
        }
    }
    {
        auto lock = al_lock_bitmap(m, ALLEGRO_PIXEL_FORMAT_RGB_888,
                                   ALLEGRO_LOCK_READWRITE);
        game((unsigned char *)lock->data, 0, map, width, height, lock->pitch);
        al_unlock_bitmap(m);
    }
    bool done = false;
    bool redraw = true;
    ALLEGRO_EVENT event;

    al_start_timer(timer);
    while (1) {
        al_wait_for_event(queue, &event);

        switch (event.type) {
        case ALLEGRO_EVENT_TIMER:
            // game logic goes here.

            redraw = true;
            break;

        case ALLEGRO_EVENT_DISPLAY_CLOSE:
            done = true;
            break;
        }

        if (done) {
            std::cout << time / frame_counter << std::endl;
            break;
        }

        if (redraw && al_is_event_queue_empty(queue)) {
            auto start = std::chrono::system_clock::now();
            al_clear_to_color(al_map_rgb(0, 0, 0));
            al_draw_bitmap(m, 0, 0, 0);
            auto lock = al_lock_bitmap(m, ALLEGRO_PIXEL_FORMAT_RGB_888, 0);
            game((unsigned char *)lock->data, 0, nullptr, width, height,
                 lock->pitch);
            al_unlock_bitmap(m);
            al_flip_display();
            redraw = false;
            auto end = std::chrono::system_clock::now();
            time += std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                          start)
                        .count();
            frame_counter++;
        }
    }

    al_destroy_font(font);
    al_destroy_display(disp);
    al_destroy_timer(timer);
    al_destroy_event_queue(queue);

    return 0;
}