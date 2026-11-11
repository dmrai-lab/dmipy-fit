# Reference catalogue audit, batch 1

`dmipy_fit/custom_optimizers/reference_models.py` is meant to hold published
models only, each faithful to its paper's equations. This batch checks nine
factories against their papers: B1-B5 and C1-C4 of the catalogue table.
The audit itself changed nothing; the fixes that followed are listed under
"Status after the fixes" at the end of this file. Checks are in
`tests/test_catalog_audit_batch1.py`; each test's prediction and tolerance is
in its docstring. (The audit first recorded every mismatch as a strict xfail
named `CATALOG-AUDIT-<id>`; the text below describes the factories as audited.)

**How each source was read.** "Full text" means the paper's own equations were
read in a raw copy (PDF text extraction or the PMC page) and searched; the
equation numbers below are the paper's. "Secondary" means a later paper that
restates the model was read; it is cited as such and its equation numbers are
never presented as the original's. Papers are not committed.

## Summary

| ID | Model | Source read | Verdict | One line |
|---|---|---|---|---|
| B1 | `ball_and_stick()` | Behrens 2003: full text (Eq. 12-13) | differs: changes recovery | Published model has one diffusivity d shared by stick and ball; the factory frees two. |
| B2 | `ball_and_zeppelin()` | Panagiotaki 2012: not accessible; taxonomy described by Ferizi 2014 (secondary) | source not verifiable | Composition arithmetic checks; whether this pair is in the taxonomy cannot be confirmed. |
| B3 | `stick_tortuous_zeppelin()` | Novikov 2019: full text (Eqs. 3.4, 3.5) | differs: changes recovery | The Standard Model has four free scalars and an ODF; the factory has De_par = Da, De_perp = Da(1-f), no ODF. |
| B4 | `free_water_elimination()` | Pasternak 2009: abstract + secondary only | differs: changes recovery (primary equations unread) | Published tissue compartment is a full tensor; the factory uses an axially symmetric Zeppelin. |
| B5 | `ivim()` | Le Bihan 1988: not accessible | source not verifiable | Biexponential arithmetic checks; the fixed D* and the bounds could not be checked against the paper. Citation volume wrong. |
| C1 | `noddi()` | Zhang 2012: full text (Eqs. 1-9) | differs: changes recovery | Dispersed extracellular signal is the average of zeppelin signals; Eq. 4 is the exponential of the averaged tensor. Up to 7e-3 of S0. |
| C2 | `bingham_noddi()` | Tariq 2016: full text (Eqs. 3-5, 10-16) | differs: changes recovery | Same extracellular defect as C1; kappa capped at 31.8, paper's constraint is 64. |
| C3 | `noddida()` | Jelescu 2016 (NMR Biomed 29:33): full text (Eqs. 2-5) | differs: changes recovery | Different model: no dispersion, tied diffusivities, an added free ball; plus a tortuosity-helper bug with three compartments. |
| C4 | `mcsmt()` | Kaden 2016: full text (Eqs. 4-8) | faithful | Equals Eqs. 6-8 to 1e-9. Bounds and citation authors differ cosmetically. |

Citation errors in the factory docstrings (all cosmetic): the table and B5
docstring cite Radiology 161 for Le Bihan 1988, which is Radiology 168 (the
DOI in the docstring is the 168 one); C4 lists the authors as "Kaden, Kelm,
Carson, McKinstry & Rueckert" but the paper's authors are Kaden, Kelm, Carson,
Does and Alexander; C3 cites "Jelescu, Veraart, Adisetiyo, Milla, Novikov &
Fieremans, NMR Biomed 28" but the paper at that DOI (10.1002/nbm.3450) is
Jelescu, Veraart, Fieremans and Novikov, NMR Biomed 29(1):33-47, epub 2015.

## B1 ball_and_stick: Behrens et al. 2003

Source: Behrens et al., MRM 50:1077-1088, "Simple Partial Volume Model",
Eqs. 12-13 (full text, from the author's PDF; text extraction is garbled but
the equations are legible).

Published model: S_i = S0[(1-f) exp(-b_i d) + f exp(-b_i d r_i^T R A R^T r_i)],
A = diag(1,0,0) (Eq. 12-13). Two compartments (stick along (theta,phi),
isotropic), f the signal fraction of the stick, **one diffusivity d** shared by
both. Free parameters: sigma, S0, d, f, theta, phi (the six listed after
Eq. 14). No exchange; the Gaussian-noise likelihood is part of the paper's
inference, not the signal model.

Factory: `C1Stick` + `G1Ball`, independent `C1Stick_1_lambda_par` and
`G1Ball_1_lambda_iso` (both 0.1-3 um^2/ms), fractions `partial_volume_0/1`.

| Difference | Effect |
|---|---|
| Two diffusivities instead of one shared d | Changes recovery: one extra free parameter; the stick can be faster than the ball, which the published model excludes. With equal values the factory equals Eq. 12 to 1e-12 (test passes). |
| Fraction naming: `partial_volume_0` = f | Cosmetic. |

Tests: `test_b1_matches_behrens_eq12_when_diffusivities_are_equal` passes;
`test_b1_has_a_single_shared_diffusivity` is a strict xfail (B1).

## B2 ball_and_zeppelin: Panagiotaki et al. 2012

Source: the 2012 paper could not be read (publisher and repository copies were
refused; no open copy found). **Not verified.** Secondary: Ferizi et al.,
MRM 2014 (open, read) restates the taxonomy: the extracellular compartment is
a Tensor, Zeppelin or Ball; the intracellular compartment is a Stick, Cylinder
or GDR-cylinders; a third compartment is a Dot, Sphere, Astrostick or
Astrocylinder. Under that description a Zeppelin and a Ball are both
*extracellular* options, so "Ball + Zeppelin" as two compartments is not an
obvious taxonomy member, but I cannot see the paper's model list. What the
factory computes is a Zeppelin plus an isotropic Gaussian, i.e. the same
structure as B4 with a free instead of a fixed free-water diffusivity.

Tests: `test_b2_is_a_zeppelin_plus_ball_sum` (composition arithmetic only,
passes to 1e-12). No claim about the paper is tested.

Verdict: source not verifiable.

## B3 stick_tortuous_zeppelin: Novikov et al. 2019

Source: Novikov, Fieremans, Jespersen, Kiselev, NMR Biomed 32:e3998, author
manuscript on PMC, Section 3.2 (full text).

Published Standard Model: S_g(b) = integral P(n) K(b, g.n) dn (Eq. 3.4),
K = S0[f exp(-b Da xi^2) + (1-f-f_CSF) exp(-b De_perp - b(De_par - De_perp)
xi^2) + f_CSF exp(-b D_CSF)] (Eq. 3.5), xi = g.n. At least four independent
scalar parameters f, Da, De_par, De_perp, plus the ODF P(n) (spherical-tensor
coefficients); f and 1-f are signal fractions; CSF optional; Gaussian
(long-time) regime. Constraints (tortuosity, Da = De_par, a fixed Da) are
discussed in Sec. 3.4 as optional additions used by particular methods, not
part of the model. The paper states the name covers NODDI, WMTI, LEMONADE and
others "with differences in the parameterization of the ODF ... and parameter
constraints".

Factory: stick + zeppelin with a single shared orientation (delta ODF),
Da = De_par linked, De_perp = Da(1-f) linked, so free parameters are f, Da and
two angles.

| Difference | Effect |
|---|---|
| De_par = Da imposed | Changes recovery (one diffusivity fewer; WMTI-style Da < De_par is excluded). |
| Tortuosity De_perp = Da(1-f) imposed | Changes recovery (couples f to anisotropy). |
| No ODF (delta function) | Changes recovery for dispersed tissue; equivalent to the SM only for perfectly parallel fibres. |
| Docstring calls the constrained form "the Standard Model parametrisation" | Misleading name; the constrained model is NODDI-without-dispersion. |

Tests: `test_b3_equals_standard_model_kernel_under_the_tortuosity_constraint`
passes (1e-12); strict xfails `test_b3_has_three_independent_diffusivities`
and `test_b3_has_an_orientation_distribution`.

## B4 free_water_elimination: Pasternak et al. 2009

Source: Pasternak et al., MRM 62:717-730. The paper is behind a paywall and
no open copy was found. **I read the abstract (bi-tensor model, tissue tensor
plus free water, free-water volume mapped) and a secondary** (Neto Henriques
et al., bioRxiv 2017 restating the model): S_i = S0[f exp(-b D_iso) +
(1-f) exp(-b g^T D g)], D the *full* tissue diffusion tensor, f the free-water
fraction, D_iso = 3.0e-3 mm^2/s. Pasternak's own equation numbers, regularised
single-shell fit and priors were not verified.

Factory: Zeppelin (two diffusivities, two angles) + Ball fixed at 3e-9 m^2/s
(`parameter_ranges` has no ball diffusivity). `partial_volume_0` is the tissue
fraction (published f = 1 - partial_volume_0, cosmetic).

| Difference | Effect |
|---|---|
| Axially symmetric tissue Zeppelin instead of a six-parameter tensor | Changes recovery: a tissue tensor with distinct radial eigenvalues is not representable. Measured: tensor eigenvalues (1.7, 0.9, 0.4)e-9, f_water 0.2, shells 1-3e9 s/m^2: best Zeppelin fit leaves rms 4.4e-2 (predicted > 1e-3). |
| D_iso = 3e-9 fixed | Matches (secondary). |
| Spatial regularisation of the original single-shell fit | Not part of the factory (a fitting choice, not the signal model). |

Tests: `test_b4_free_water_ball_is_fixed_at_3e_minus_3_mm2_per_s` passes;
strict xfail `test_b4_tissue_compartment_represents_a_general_tensor`.

Verdict is conditional on the secondary being right about the tensor, which
the abstract's "bi-tensor" supports.

## B5 ivim: Le Bihan et al. 1988

Source: Radiology 168:497-505. Not accessible; no open copy found.
**Nothing was verified from the primary text.** The test uses the standard
biexponential (1-f) exp(-bD) + f exp(-bD*) as the IVIM statement, flagged as
unverified against the 1988 equations. Whether the 1988 text writes the
perfusion term with D* or with D + D* was not checked.

Factory: two Balls; the blood Ball fixed at 7e-9 m^2/s; the tissue Ball's
bounds [0.5, 6]e-9 m^2/s. The fixed D* and the bounds are choices I could not
trace to the paper; a fixed D* removes a published fit parameter if the paper
fits it. `partial_volume_0` is the tissue fraction (1 - f, cosmetic). The tissue
upper bound 6e-9 is twice free-water diffusivity (not physical, harmless for
fitting). Docstring cites Radiology 161 (should be 168).

Tests: `test_b5_is_a_biexponential_with_fixed_pseudodiffusion` passes (1e-12)
and pins the parameter inventory and bounds.

Verdict: source not verifiable.

## C1 noddi: Zhang et al. 2012

Source: Zhang, Schneider, Wheeler-Kingshott, Alexander, NeuroImage 61:1000
(author PDF, full text), Sec. 2.1-2.2, 2.6.1.

Published model: A = (1-nu_iso)(nu_ic A_ic + (1-nu_ic) A_ec) + nu_iso A_iso
(Eq. 1). A_ic = integral f(n) exp(-b d_par (q.n)^2) dn (Eq. 2); f the Watson
ODF M(1/2,3/2,kappa)^-1 exp(kappa (mu.n)^2) (Eq. 3). **A_ec: log A_ec =
-b q^T (integral f(n) D(n) dn) q (Eq. 4)**, D(n) cylindrically symmetric with
d_par and d_perp = d_par(1-nu_ic) (tortuosity); the dispersed apparent
diffusivities are d'_par = d_par - d_par nu_ic (1-tau1) and d'_perp = d_par -
d_par nu_ic (1+tau1)/2 (Eqs. 5-6), tau1 from Eqs. 7-8. A_iso = exp(-b d_iso).
OD = (2/pi) arctan(1/kappa) (Eq. 9). Parameters nu_ic, d_par, kappa, mu, nu_iso,
d_iso; d_par = 1.7e-3 and d_iso = 3.0e-3 mm^2/s fixed (Sec. 2.6.1); myelin
water absent; Gaussian-phase style compartments.

Factory: `SD1WatsonDistributed([C1Stick, G2Zeppelin])` (bundle fraction =
nu_ic), tortuosity link, d_par fixed 1.7e-9 and shared, Ball fixed 3e-9;
free parameters mu, odi, bundle fraction, tissue/CSF fractions. ODI mapping
is Eq. 9 (tested).

| Difference | Effect |
|---|---|
| Dispersed extracellular signal is the ODF average of per-orientation zeppelin signals, not exp(-b g^T <D> g) | **Changes recovery.** Max deviation of the full signal from Eq. 4 (nu_ic 0.6, nu_iso 0.2, shells 1/2/3e9): 4.5e-3 (ODI 0.1), 6.9e-3 (0.3), 7.3e-3 (0.7) of S0; with the factory's own average-of-signals reference they agree to 3e-5. Both are exact for aligned fibres. |
| odi >= 0.02 (kappa <= 31.8) | Not in the NODDI paper (its toolbox bound was not read); see C2 for the published bound in the Bingham paper. Narrows the recoverable ODI for very coherent tissue. |
| Fractions as `partial_volume_*` | Cosmetic. |

Passing checks: intracellular Eq. 2, tortuosity, fixed d_par/d_iso, Eq. 1
composition, ODI Eq. 9, Eq. 8 against quadrature, free parameter set. Strict
xfail: `test_c1_extracellular_is_exponential_of_averaged_tensor` (three ODIs).

## C2 bingham_noddi: Tariq et al. 2016

Source: Tariq, Schneider, Alexander, Gandini Wheeler-Kingshott, Zhang,
NeuroImage 133:207 (accepted manuscript, full text), Eqs. 3-5, 10-16 and
"Model fitting procedure".

Published model: NODDI with the Bingham ODF f(n;B) = c_B^-1 exp(kappa
(mu1.n)^2 + beta (mu2.n)^2) (Eq. 10), beta <= kappa; A_in Eq. 3; A_en = exp(-b
q^T D_en q) with D_en the ODF-averaged tensor (Eqs. 4-5), d_perp =
d_par(1-nu_in); d_par = d_i = 1.7e-9, d_iso = 3.0e-9 fixed. Free: nu_in,
kappa, beta, psi, theta, phi, nu_iso. "The only explicit constraint ... is
applied to kappa and beta (<= 64)."

Factory: `SD2BinghamDistributed`, odi, `beta_fraction` (beta = beta_fraction *
kappa, so beta <= kappa by construction), psi, mu.

| Difference | Effect |
|---|---|
| Averaged-signal extracellular (as C1) | Changes recovery; same defect. |
| kappa <= 31.8 from odi >= 0.02, versus 64 | Changes recovery only for very coherent tissue (kappa between 32 and 64). |
| Euler-angle (psi) convention | Not verified; the tests use the factory's own density to avoid depending on it. |

Passing: Watson limit (beta_fraction = 0 reproduces `noddi()` to 1.5e-3);
composition with signal-averaged extracellular. Strict xfails: tensor-average
(two beta fractions) and the kappa bound.

## C3 noddida: Jelescu et al. 2016

Source: Jelescu, Veraart, Fieremans, Novikov, "Degeneracy in model parameter
estimation for multi-compartmental diffusion in neuronal tissue", NMR Biomed
29:33-47 (PMC author manuscript, full text), Theory and Methods, Eqs. 2-5.
The paper names NODDIDA (introduced in its Ref. 6).

Published model: S/S0 = f integral W(n) exp(-b Da (g.n)^2) dn + (1-f)
exp(-b g^T (integral W(n) D_e(n) dn) g) (Eq. 2), W the Watson ODF, D_e(n) =
De_par nn^T + De_perp(1 - nn^T); free parameters f, Da, De_par, De_perp,
kappa, S0 (mu from the overall tensor); **no CSF compartment** (f_iso = 0);
Da, De_par, De_perp, bounds [0, 4] um^2/ms; NODDI's constraints Da = De_par =
1.7 and De_perp = De_par(1-f) (Eq. 5) are *removed* in NODDIDA.

Factory: aligned stick + zeppelin + free-diffusivity Ball; Da = De_par linked;
De_perp = Da(1-f) linked; no Watson dispersion.

| Difference | Effect |
|---|---|
| No orientation dispersion (kappa absent) | Changes recovery. |
| De_par = Da and De_perp tortuosity tied (published: independent) | Changes recovery; this is the constraint set NODDIDA exists to drop. |
| An isotropic Ball with free D_iso added (published: none) | Changes recovery. |
| Diffusivity bounds 0.1-3 versus 0-4 um^2/ms | Changes recovery for CSF-contaminated voxels (the paper's reason for 4). |
| Docstring says "all diffusivities free" | False: only Da and D_iso are free. |

Overall the factory is not NODDIDA; it is an aligned NODDI with free Da and a
free ball.

**C3b, a defect in the tortuosity helper.** `T1_tortuosity.__call__`
(`dmipy_fit/utils/utils.py`) normalises `vf_intra` by `vf_intra + vf_extra`
but then uses the un-normalised `vf_extra`, so for `vf_intra + vf_extra != 1`
it does not return `lambda_par * vf_extra/(vf_intra+vf_extra)`. With a third
compartment (as in `noddida()`) at fractions (0.4, 0.35, 0.25) the factory
signal implies De_perp = 0.396 Da, not 0.467 Da; the signal differs by up to
2e-2. Found by inverting the factory signal; the helper's unit test
(`utils/tests/test_tortuosity_multi_tissue.py`) multiplies the expected ratio
by lambda_par = 1.7e-9 and compares at 7 decimals, so it passes for any
value. The same helper serves any other three-compartment factory with a
tortuosity link (not checked in this batch). Two-compartment models
(sum = 1) are not affected.

Tests: `test_c3_factory_is_the_aligned_special_case_without_the_ball` passes;
strict xfails for three independent diffusivities, a dispersion parameter, no
isotropic compartment, `test_c3b_tortuosity_with_a_third_compartment` and
`test_c3b_noddida_signal_with_ball_fraction`.

## C4 mcsmt: Kaden et al. 2016

Source: Kaden, Kelm, Carson, Does, Alexander, NeuroImage 139:346 (PMC, full
text), "Multi-compartment microscopic diffusion imaging", Eqs. 4-8.

Published model: intra Eq. 4 exp(-b <g,w>^2 lambda), extra Eq. 5 with
lambda_perp_ext = (1-v_int) lambda (first-order tortuosity), lambda_par_ext =
lambda. Spherical mean e_b = v e_int + (1-v) e_ext, e_int = sqrt(pi)
erf(sqrt(b lambda)) / (2 sqrt(b lambda)) (Eq. 7), e_ext = exp(-b lambda_perp)
sqrt(pi) erf(sqrt(b(lambda - lambda_perp))) / (2 sqrt(b(lambda-lambda_perp)))
(Eq. 8). Free: v_int in [0,1], lambda in [0, lambda_free] (3.05 um^2/ms at
37 C); no isotropic compartment; no myelin compartment; orientation-free.

Factory: `MultiCompartmentSphericalMeanModel([C1Stick, G2Zeppelin])` with the
same links; free `C1Stick_1_lambda_par` and fractions.

| Difference | Effect |
|---|---|
| lambda bounds 0.1-3.0 versus 0-3.05 um^2/ms | Cosmetic for brain tissue; excludes lambda > 3.0. |
| Fractions in [0.01, 0.99] versus [0,1] | Cosmetic. |
| Authors in docstring | Cosmetic citation error. |

Tests: `test_c4_matches_kaden_eqs_6_to_8` (three (v, lambda), five shells,
rtol 1e-9) and `test_c4_free_parameters_are_v_int_and_lambda` pass.

## Open findings for the lead

| ID | Finding | Suggested handling |
|---|---|---|
| B1 | One shared d in the published model | Link the two diffusivities, or rename the model. |
| B3 | "Standard Model" name for a constrained NODDI-without-dispersion | Rename, or free Da/De_par/De_perp and add an ODF. |
| B4 | Tissue Zeppelin instead of tensor | Use a tensor compartment, or rename. |
| C1/C2 | Extracellular signal averaged as signals, not tensors | Make the extracellular compartment exponential of the dispersed tensor (closed form in Eqs. 5-8 for Watson). Changes every fitted NODDI value by a small, ODI-dependent amount. |
| C2 | kappa cap 31.8 vs 64 | Lower odi bound to about 0.01. |
| C3 | Not NODDIDA | Implement Eq. 2 (Watson dispersion, three free diffusivities, no ball), or rename. |
| C3b | `T1_tortuosity` mixes normalised and raw fractions; its unit test cannot fail | Fix the helper and test at the SI scale. |
| B2, B5 | Primary text unread | Needs the papers (institutional access) before a verdict. |

## Status after the fixes

Decision (lead, dmipy-fit PR 82): fix the obvious mistakes; documented design
choices stay.

| ID | Outcome |
|---|---|
| B1 | Fixed: stick and ball share one diffusivity d (`C1Stick_1_lambda_par`), as in Behrens 2003 Eqs. 12-13. |
| B3 | Model kept; docstring and table say it is the dispersion-free, tortuosity-constrained special case of the Standard Model. Asserted by a passing test. |
| B4 | Documented difference: the package has no full-tensor Gaussian compartment, so the tissue stays a Zeppelin; the docstring states it (secondary source Neto Henriques 2017). Asserted by a passing test (rms residual > 1e-2). |
| B5 | Citation volume corrected (Radiology 168); primary text still unread. |
| C1 | Formulation kept (dmipy's Watson bundle, signal-averaged extracellular, for nested volume fractions). Docstring states the 4.5e-3 to 7.3e-3 of S0 difference from Zhang Eq. 4; the test asserts it. |
| C2 | Same formulation kept and documented (7.3e-3, 7.4e-3 of S0 at the tested cases). kappa bound is the paper's 64. |
| C3 | Rebuilt as NODDIDA: Watson-dispersed Stick + Zeppelin with free D_a, D_e,par, D_e,perp in [0, 4] um^2/ms, no tortuosity, no Ball. The signal-averaged extracellular difference from Eq. 2 is documented (6.0e-3 to 9.8e-3 of S0 at the tested cases). The paper's bound on kappa was not read; the package's odi >= 0.02 stays. C3b was fixed in its own PR. |
| C4 | Authors corrected. |
| B2 | Unchanged (source unread). |

`noddida_mte()` keeps its Stick + tortuous Zeppelin + Ball structure (it is
not rebuilt here); only its NODDIDA citation is corrected.
