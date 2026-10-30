#pragma once

#include <math.h>

#ifndef INFINITY
#define INFINITY as_type<float>(0x7f800000u)
#endif
#ifndef NAN
#define NAN as_type<float>(0x7fc00000u)
#endif

using uchar = unsigned char;
using uint = unsigned int;
using uint64 = unsigned long long;
using int64 = long long;

template <class A, class B> constexpr bool is_same_v = false;
template <class A> constexpr bool is_same_v<A, A> = true;

template <class T> struct vec3 {
    T x, y, z;
    __device__ vec3() {}
    __device__ explicit vec3(T v) : x(v), y(v), z(v) {}
    __device__ vec3(T a, T b, T c) : x(a), y(b), z(c) {}
    __device__ T &operator[](uint i) {
        return i == 0 ? x : i == 1 ? y : z;
    }
    __device__ const T &operator[](uint i) const {
        return i == 0 ? x : i == 1 ? y : z;
    }
};
template <class T> struct vec4 {
    T values[4];
    __device__ T &operator[](uint i) {
        return values[i];
    }
    __device__ const T &operator[](uint i) const {
        return values[i];
    }
};
__device__ float dot(vec3<float> a, vec3<float> b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ float mix(float a, float b, float t) {
    return a + (b - a) * t;
}
template <class A, class B> __device__ auto min(A a, B b) -> decltype(a + b) {
    return a < b ? a : b;
}
template <class A, class B> __device__ auto max(A a, B b) -> decltype(a + b) {
    return a > b ? a : b;
}
template <class T, class A, class B> __device__ T clamp(T value, A low, B high) {
    return min(max(value, T(low)), T(high));
}
template <class D, class S> __device__ D as_type(S value) {
    static_assert(sizeof(D) == sizeof(S), "bit casts must retain sample width");
    union {
        S input;
        D output;
    } bits;
    bits.input = value;
    return bits.output;
}
