// circle -O3 -std=c++20 -Iinclude -sm_60 hello_gpu.cpp -o hello_gpu
#include <cassert>
#include <cstdio>

__forceinline__ __device__ void panic()
{
  printf("panic\n");
  assert(false);
}

// configure fmt
#define FMT_HEADER_ONLY
#define FMT_USE_INT128 0 // XXX LLVM ERROR: Undefined external symbol "__udivti3"
#define FMT_STATIC_THOUSANDS_SEPARATOR ',' // avoid <locale>
#define FMT_THROW(exception) panic
#include <fmt/format.h>
#include <fmt/compile.h>

#include <iostream>

template<class S, class... Args>
  requires fmt::detail::is_compiled_string<S>::value
constexpr std::string my_format(const S& fmt_str, Args&&... args)
{
  auto sz = fmt::formatted_size(fmt_str, args...);
  std::string result(sz, 0);
  fmt::format_to(result.data(), fmt_str, args...);
  return result;
}

template<class S, class... Args>
  requires fmt::detail::is_compiled_string<S>::value
constexpr void my_print(const S& fmt_str, Args&&... args)
{
  printf(my_format(fmt_str, args...).c_str());
}

__global__ void foo()
{
  my_print(FMT_COMPILE("hello\n"));
  my_print(FMT_COMPILE("hello again: {}, {}, {}\n"), 1, 2, "string");
  my_print(FMT_COMPILE("float: {}\n"), 1.234e-20f);
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

