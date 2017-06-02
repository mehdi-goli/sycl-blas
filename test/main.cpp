#include <cstdlib>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

int main(int argc, char *argv[]) {
  srand(time(NULL)/10*10);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
