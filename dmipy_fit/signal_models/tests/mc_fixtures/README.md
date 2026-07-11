# Monte-Carlo signal fixtures

These `.npz`/`.yaml` files are **offline-generated Monte-Carlo signals from dmipy-sim**, committed
here as cached ground truth so the analytical cylinder/dispersion models in dmipy-fit can be
validated against MC **parity** in CI without running any live Monte Carlo.

This is the same pattern as the flagship (`examples/flagship_canonical_wm/`): the expensive,
non-deterministic-runtime MC is generated once on a GPU and frozen; the fast, deterministic
analytical forward is what CI actually re-runs, asserting it reproduces the frozen MC. That way a
change in dmipy-fit is caught by a parity regression here, while the heavy MC generation (and
dmipy-sim's own MC-validation tests) live in dmipy-sim and run offline.

Each fixture is a restricted-cylinder signal for one radius and one waveform family:

- `fixtures_R<radius>um_pgse_short.npz`, `..._pgse_finite_delta.npz`
- `fixtures_R<radius>um_ogse_cosine.npz`, `..._ogse_trap.npz`
- `fixtures_R2.0um_gamma_lm_validation.npz`, `fixtures_ogse_dispersed.npz`, `fixtures_now_gamma_lm.npz`

radii ∈ {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0} µm. The `.yaml` sidecar records the waveform and
substrate parameters used to generate the paired `.npz`.

## Regenerating

The generators live in the development repo (they drive dmipy-sim on a GPU). To validate against a
fresh regeneration, point the tests at it:

```bash
DMIPY_FIXTURE_DIR=/path/to/fresh/fixtures pytest dmipy_fit/signal_models/tests/test_cylinder_fixtures.py
```
