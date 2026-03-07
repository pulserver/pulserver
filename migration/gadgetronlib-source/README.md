# gadgetronlib-source

This scaffold contains public C++ source components for Gadgetron-side integration.

## Included from monorepo
- `extensions/gadgetronlib/`
- `extensions/ge2mrdlib/`
- `LICENSE.txt`

## Important build note
- `extensions/ge2mrdlib/` may require private/proprietary GE Orchestra SDK components for full builds.
- Public CI for this repo should default to jobs that do not require private SDK dependencies.
