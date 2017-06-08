#include <cstdlib>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <fcntl.h>
#include <unistd.h>

void set_seed() {
  int fd = open("/dev/random", O_RDONLY);
  if (fd == -1) abort();
  unsigned seed, pos = 0;
  while (pos < sizeof(seed)) {
    int a = read(fd, (char *)&seed + pos, sizeof(seed) - pos);
    if (a <= 0) abort();
    pos += a;
  }
  srand(seed);
  close(fd);
}

int main(int argc, char *argv[]) {
  int seed = time(NULL) / 30;
  srand(seed);
  /* set_seed(); */
  std::cout << "seed: " << seed << std::endl;
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
