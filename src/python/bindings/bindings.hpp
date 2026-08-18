/**
 * @file bindings.hpp
 * @brief The five binding groups that make up `pulserver._ext`.
 *
 * Each is defined in its own translation unit and populates the submodule
 * named after it; module.cpp creates the submodules and calls these in turn.
 */

#pragma once

#include <pybind11/pybind11.h>

void pulserver_bind_pulseg(pybind11::module_& m);
void pulserver_bind_pulseqpp(pybind11::module_& m);
void pulserver_bind_arbgrad(pybind11::module_& m);
void pulserver_bind_sampling(pybind11::module_& m);
void pulserver_bind_recon_cpu(pybind11::module_& m);
