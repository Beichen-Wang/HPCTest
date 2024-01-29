#pragma once
#include <cstddef>
#include <memory>
#include <stdio.h>

#define MEM_ASSERT(condition, message) \
      ((condition)                         \
           ? void()                        \
           : MyVector::assert_fail(__FILE__, __LINE__, (message)))

namespace MyVector{
    void assert_fail(const char* file, int line, const char* message) {
        printf("{%s}:{%d}: assertion failed: {%s}", file, line, message);
        std::abort();
    }
    template <typename Int>
    constexpr typename std::make_unsigned<Int>::type to_unsigned(Int value) {
        MEM_ASSERT(value >= 0, "negative value");
        return static_cast<typename std::make_unsigned<Int>::type>(value);
    }
}