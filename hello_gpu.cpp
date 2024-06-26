// circle -O3 -std=c++20 -Iinclude -sm_60 hello_gpu.cpp -o hello_gpu
#include <cassert>
#include <iostream>
#include <nv/target>

constexpr void fmt_assert_fail(const char* file, int line, const char* message, const char* function) noexcept {
#if defined(__circle_lang__) and defined(__CUDACC__)
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (
    // GPU code has no access to either std::fprintf or std::terminate
    __assert_fail(message, file, line, function);
  ), (
    // Use unchecked std::fprintf to avoid triggering another assertion when
    // writing to stderr fails
    std::fprintf(stderr, "%s:%d: assertion failed: %s", file, line, message);
    // Chosen instead of std::abort to satisfy Clang in CUDA mode during device
    // code pass.
    std::terminate();
  ))
#else
  // Use unchecked std::fprintf to avoid triggering another assertion when
  // writing to stderr fails
  std::fprintf(stderr, "%s:%d: assertion failed: %s", file, line, message);
  // Chosen instead of std::abort to satisfy Clang in CUDA mode during device
  // code pass.
  std::terminate();
#endif
}

// configure fmt
#define FMT_HEADER_ONLY
#define FMT_USE_INT128 0 // XXX LLVM ERROR: Undefined external symbol "__udivti3"
#define FMT_STATIC_THOUSANDS_SEPARATOR ',' // avoid <locale>
#define FMT_ASSERT(condition, message) ((condition) ? (void) 0 : fmt_assert_fail(__FILE__, __LINE__, (message), __FUNCTION__))
#include <fmt/format.h>
#include <fmt/compile.h>

template<auto N>
struct fixed_string
{
  constexpr fixed_string(const char (&str)[N])
  {
    std::copy_n(str, N, value);
  }

  constexpr const operator std::string_view () const
  {
    return {value, N};
  }

  constexpr const operator fmt::basic_string_view<char> () const
  {
    return {value, N};
  }

  constexpr const char& operator[](decltype(auto) i) const
  {
    return value[i];
  }

  constexpr static auto size()
  {
    return N;
  }

  char value[N];
};

template<fixed_string s>
constexpr auto operator "" cs() noexcept
{
  return FMT_COMPILE(s);
}

template<class S, class... Args>
constexpr std::string format(const S& fmt_str, Args&&... args)
{
  auto sz = fmt::formatted_size(fmt_str, args...);
  std::string result(sz, 0);
  fmt::format_to(result.data(), fmt_str, args...);
  return result;
}

template<class S, class... Args>
constexpr void print(const S& fmt_str, Args&&... args)
{
  printf(format(fmt_str, args...).c_str());
}

template<class S, class... Args>
constexpr auto println(const S& fmt_str, Args&&... args)
{
  print(fmt_str, args...);
  printf("\n");
}

__global__ void foo()
{
  print("hello\n"cs);
  println("hello again: {}, {}, {}"cs, 1, 2, "string");
  println("float: {}"cs, 1.234e-20f);
}

int main()
{
  foo<<<1,1>>>();
  if(cudaError_t e = cudaDeviceSynchronize())
  {
    throw std::runtime_error(cudaGetErrorString(e));
  }

  std::cout << "OK" << std::endl;

  return 0;
}

