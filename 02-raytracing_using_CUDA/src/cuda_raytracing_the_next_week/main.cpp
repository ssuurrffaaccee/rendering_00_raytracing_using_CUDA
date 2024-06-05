#include "game.h"
#include "util/check.h"
int main() {
  try {
    Game game;
    game.init();
    game.run();
  } catch (MyExceptoin& e) {
    std::cout << e.what() << "\n";
  } catch (std::exception& e) {
    std::cout << e.what() << "\n";
  }
  return 0;
}