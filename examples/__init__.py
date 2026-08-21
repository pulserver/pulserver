"""Standalone Pulserver example modules.

``sequence`` and ``recon`` are the two directories the zoo is written in;
``pyproject.toml`` installs both into :mod:`pulserver.app`, which is the one
namespace a plugin is reached through. This package marker also makes them
importable from an uninstalled source checkout.
"""
