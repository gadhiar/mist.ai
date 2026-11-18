// SPDX-License-Identifier: MIT
// Part of MIST.AI, released under the MIT License; see LICENSE at the repository root.
//
// A minimal test registry for the host tests of `mist_core`.
//
// The core builds without exceptions, so a failed check records the failure and returns from the
// test function instead of throwing. Each binary runs every registered case, or the one named on
// its command line, and `--list` prints the names; `tests/unit/firmware/test_core.py` reads the
// names from the sources and runs one case per pytest item.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace mist_test {

using TestFn = void (*)();

struct Case {
  const char* name;
  TestFn fn;
};

constexpr int kMaxCases = 256;

inline Case* cases() {
  static Case all[kMaxCases];
  return all;
}

inline int& case_count() {
  static int count = 0;
  return count;
}

inline int& failures() {
  static int count = 0;
  return count;
}

struct Registrar {
  Registrar(const char* name, TestFn fn) {
    if (case_count() < kMaxCases) {
      cases()[case_count()++] = Case{name, fn};
    }
  }
};

inline void fail(const char* file, int line, const char* what) {
  ++failures();
  std::printf("FAIL %s:%d: %s\n", file, line, what);
}

inline void fail_values(const char* file, int line, const char* what, long long a, long long b) {
  ++failures();
  std::printf("FAIL %s:%d: %s (%lld vs %lld)\n", file, line, what, a, b);
}

inline void fail_strings(const char* file, int line, const char* what, const char* a,
                         const char* b) {
  ++failures();
  std::printf("FAIL %s:%d: %s (\"%s\" vs \"%s\")\n", file, line, what, a ? a : "(null)",
              b ? b : "(null)");
}

inline int run(int argc, char** argv) {
  if (argc > 1 && std::strcmp(argv[1], "--list") == 0) {
    for (int i = 0; i < case_count(); ++i) {
      std::printf("%s\n", cases()[i].name);
    }
    return 0;
  }
  const char* only = argc > 1 ? argv[1] : nullptr;
  int ran = 0;
  for (int i = 0; i < case_count(); ++i) {
    if (only != nullptr && std::strcmp(only, cases()[i].name) != 0) {
      continue;
    }
    const int before = failures();
    cases()[i].fn();
    ++ran;
    std::printf("%s %s\n", failures() == before ? "PASS" : "FAILED", cases()[i].name);
  }
  if (ran == 0) {
    std::printf("no test case named %s\n", only != nullptr ? only : "(any)");
    return 2;
  }
  return failures() == 0 ? 0 : 1;
}

}  // namespace mist_test

#define MIST_TEST(name)                                              \
  static void name();                                                \
  static const ::mist_test::Registrar name##_registrar(#name, &name); \
  static void name()

#define CHECK(cond)                                    \
  do {                                                 \
    if (!(cond)) {                                     \
      ::mist_test::fail(__FILE__, __LINE__, #cond);    \
      return;                                          \
    }                                                  \
  } while (0)

#define CHECK_EQ(a, b)                                                               \
  do {                                                                               \
    const long long check_a_ = static_cast<long long>(a);                            \
    const long long check_b_ = static_cast<long long>(b);                            \
    if (check_a_ != check_b_) {                                                      \
      ::mist_test::fail_values(__FILE__, __LINE__, #a " == " #b, check_a_, check_b_); \
      return;                                                                        \
    }                                                                                \
  } while (0)

#define CHECK_STREQ(a, b)                                                       \
  do {                                                                          \
    const char* check_a_ = (a);                                                 \
    const char* check_b_ = (b);                                                 \
    if (check_a_ == nullptr || check_b_ == nullptr ||                           \
        std::strcmp(check_a_, check_b_) != 0) {                                 \
      ::mist_test::fail_strings(__FILE__, __LINE__, #a " == " #b, check_a_, check_b_); \
      return;                                                                   \
    }                                                                           \
  } while (0)

// The start marker is main's first output: a run that crashed without it crashed before main, in
// the sanitizer runtime's start-up, and the pytest driver may start it again.
#define MIST_TEST_START_MARKER()      \
  do {                                \
    std::printf("MIST_TEST_START\n"); \
    std::fflush(stdout);              \
  } while (0)

int main(int argc, char** argv) {
  MIST_TEST_START_MARKER();
  return mist_test::run(argc, argv);
}
