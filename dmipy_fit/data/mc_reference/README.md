# Monte Carlo reference datasets

Small, pre-reduced ground-truth arrays produced by the **dmipy-sim** Monte Carlo
diffusion-MRI engine (the numerical ground truth that dmipy-fit's analytical
models are validated against).  They are bundled so the tutorials and the test
suite can overlay the numerical ground truth on the analytical white-matter model
**without re-running any random walk** — load them with
`dmipy_fit.data.saved_data.mc_reference(name)`.

Each was measured off a cached canonical white-matter master walk (g-ratio 0.70,
packing fraction 0.55, Gamma(α=2) axon-diameter distribution) and reduced to the
curves below. Regenerate with `tools/precompute_mc_reference.py` (needs a CUDA GPU
and the `dmipy-sim` package + its warm master-walk cache).

| file | what it holds | paired analytic API |
|------|---------------|---------------------|
| `surface_relaxivity.npz` | per-compartment surface attenuation `B_ia/B_ea` (MC, self-consistent analytic, idealised) vs echo time `TES`; realised `S_ext/V` | `dmipy_fit.white_matter.surface.b_hat_ia / b_hat_ea_long` |
| `parity_overlap.npz` | susceptibility `sin⁴θ` factor `A_xe*` vs angle, extra-axonal attenuation `B_ea*` vs b, EA signal `C_ea*` vs TE — MC and analytic, at 3T and 7T | `OccupancyGatedModel(G2Zeppelin, [Susceptibility, …])` |
| `crossterm.npz` | diffusion×susceptibility apparent-D⊥ warp `dD` and exponent `lnX` vs field/b, for a fine and a large-calibre pack (seed-averaged) | `UnifiedWhiteMatterParameters.xi_cross_factor` |

**Provenance.** These are the validated arrays from the coherence-pathway paper's
parity figures (512-cylinder packs for surface/parity; seed-averaged 128-cylinder
packs for the heavy-tail cross-term). See the dmipy-sim `CrossTermProbe` /
`UnifiedWhiteMatterModel` docs and the pack-size convention (≥512 cylinders +
many seeds for any reported cross-term magnitude).
