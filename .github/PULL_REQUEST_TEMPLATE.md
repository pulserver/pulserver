# What this changes

<!-- What the code does now that it did not before, in a sentence or two.
     Link the issue if there is one. -->

## Why

<!-- The problem it solves. For a physics or safety change, say what makes
     the new behaviour correct: the reference, the invariant, or the
     measurement. -->

## How it was checked

<!-- Tick what you ran. `bash scripts/run_tests.sh` covers all of it, and the
     native lanes skip themselves when a toolchain is missing. -->

- [ ] `bash scripts/format_and_lint.sh` — formatting and lint are clean
- [ ] `bash scripts/run_tests.sh` — Python, C, C++ and Nim
- [ ] New behaviour has a test whose name states the invariant
- [ ] Fixtures regenerated with `bash scripts/regenerate_fixtures.sh` (if any changed)
- [ ] Documentation updated (if this changes what a user does or sees)

## Anything a reviewer should look at first

<!-- The part you are least sure about, a deliberate trade-off, or a number
     that moved. If a benchmark changed, the before and after. -->
