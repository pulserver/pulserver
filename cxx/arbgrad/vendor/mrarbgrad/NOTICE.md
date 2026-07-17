# Vendored third-party code: MRArbGrad

The C++ sources under this directory (`mag/`, `traj/`, `utility/`) are vendored
from **MRArbGrad** (Magnetic Resonance Arbitrary Gradient Toolbox):

- Repository: https://github.com/RyanShanghaitech/MRArbGrad
- Reference: Luo R, Huang H, Miao Q, Xu J, Hu P, Qi H. "Real-Time Gradient
  Waveform Design for Arbitrary k-Space Trajectories." IEEE Transactions on
  Biomedical Engineering. 2026;1-12.

Only the core solver (`Mag`) and the built-in trajectory definitions
(`TrajFunc`, `MrTraj`, `MrTraj_2D`, `VDSpiral`, `Rosette`) and their
supporting utilities are vendored. The upstream CPython C-API bindings
(`main.cpp`), the external-function/-samples entry points, and the optional
`mtg/` (Lustig min-time-gradient) solver are intentionally **not** vendored —
`pulserver` provides its own pybind11 bindings
(`../../../python/pulserver/_ext/_arbgrad_wrapper.cpp`). `Spiral`/
`Spiral_TrajFunc` (plain constant-pitch Archimedean spiral) is also **not**
vendored: `VDSpiral_TrajFunc::getK` algebraically reduces to the exact same
`rho = kRhoPhi * phi` formula when its two shape parameters are equal, so it
is a redundant special case of `VDSpiral` and not exposed separately.

## License (MIT)

```
Copyright 2026 Rui Luo

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
