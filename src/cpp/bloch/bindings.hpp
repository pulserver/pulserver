/**
 * @file bindings.hpp
 * @brief The isochromat engine, bound as `pulserver._ext.bloch`.
 */

#pragma once

#include <pybind11/pybind11.h>

void bind_bloch(pybind11::module_& module);
