# Contributing to dmipy-fit

Thanks for your interest. dmipy-fit is the analytical-inverse engine; physics is the
specification, so contributions are judged first on physical correctness (analytical limits,
invariants, and — where applicable — parity with the dmipy-sim Monte-Carlo forward model),
not just on passing tests.

## Development setup

```bash
git clone https://github.com/dmrai-lab/dmipy-fit.git
cd dmipy-fit
pip install -e ".[jax]"        # add [data] for the cached MC reference arrays
pytest -q                      # analytical + MISST/dipy reference tests
pytest -q -m "not slow"        # skip the heavy GPU battery
```

Test in **float64 first** (reference/correctness); float32 is production speed and is only
acceptable when the difference from float64 is below the physical noise floor.

## Guidelines

- Match the surrounding code — naming, comment density, and idiom.
- There is one source of truth for every physical constant (tissue constants in the sim
  `Substrate`; scanner limits in `dmipy_sim.sequences.scanner_constants`). Read them via their
  accessors; do not hard-code.
- Add a test that fails without your change. For a new physical effect, add a validation
  example against an exact analytical result, not only a regression test.

## Contributor License Agreement

dmipy is **dual-licensed** (AGPL-3.0 OR commercial), so we need an explicit relicensing grant
from contributors — see the
[CLA](https://github.com/dmrai-lab/dmipy/blob/main/licensing/CLA.md). For now, add this line
to your first pull request:

> I have read the CLA and I agree to it on behalf of myself (and my employer if applicable).
> Signed, [your name] <[your email]>

You keep the copyright to your work. Please open an issue before starting anything large.
