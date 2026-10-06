/**
 * @file bindings.hpp
 * @brief Python bindings of the Fourier engine's compiled parts.
 */

#ifndef PULSERVER_FOURIER_BINDINGS_HPP
#define PULSERVER_FOURIER_BINDINGS_HPP

#include <pybind11/pybind11.h>

void bind_fourier(pybind11::module_& module);

#endif /* PULSERVER_FOURIER_BINDINGS_HPP */
