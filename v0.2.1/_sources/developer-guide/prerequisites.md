# Development prerequisites

Development requires Git, Python 3.10–3.13, a C compiler, a C++17 compiler and
CMake. The extension `pulserver._ext` compiles `src/cpp/` and `src/c/`; the
latter is compiled with `-std=c90 -pedantic-errors`.

The test that compiles `src/c/` as a 32-bit scanner build requires a 32-bit C
toolchain, which `gcc-multilib` provides on Debian and Ubuntu. It skips when
none is available.
