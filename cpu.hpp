#include <allegro5/allegro5.h>
#include <iostream>
#include <vector>
class game {
  public:
    game(int width, int height, int *state, int step);
    bool render(unsigned int *p, int pitch);
    bool step();

  private:
    int frame_counter;
    int width, height, pitch;
    std::vector<int> buf1;
    std::vector<int> buf2;
};