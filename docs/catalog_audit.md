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

`noddida_mte()` kept its Stick + tortuous Zeppelin + Ball structure in batch 1
(only its NODDIDA citation was corrected); batch 2 rebuilds it on `noddida()`.

# Reference catalogue audit, batch 2

Eight more catalogue entries against their papers: D1, E1-E3, F1-F3 and the
`noddida_mte()` factory. Checks are in `tests/test_catalog_audit_batch2.py`.
The lead's decision for this batch (2026-11-11): fix the obvious mistakes in the
same change; documented design choices stay. The tests' predictions were written
before any factory was changed; the tests that failed on the unchanged factories
are listed per model, and every one now passes. No finding is left open, so the
batch has no xfail.

**How each source was read.** As in batch 1. "Full text" is the author
manuscript or open-access page, equations read from its MathML. "Abstract only"
means nothing but the abstract was readable; no equation of such a paper is
claimed. "Secondary" is a later open paper restating the model.

## Summary

| ID | Model | Source read | Verdict | One line |
|---|---|---|---|---|
| D1 | `two_fascicle_noddi()` | Behrens 2007: full text (Eqs. 1-2) | no published source for this composition | The cited paper is a ball-and-sticks paper; the factory is two NODDI bundles. Docstring and table fixed; arithmetic verified. |
| E1 | `charmed()` | Assaf & Basser 2005: abstract only; Eqs. 1-3 as restated in Assaf 2008 (full text) | source not verifiable | Composite and Callaghan cylinder equal the restated Eqs. 1-3 to 1e-15; the tortuosity link and shared D_par are unverified. |
| E2 | `axcaliber()` | Assaf 2008: full text (Eqs. 1-5) | faithful | Gamma on the radius, area-weighted, equals Eqs. 4-5 to 2.1e-3 (quadrature). Docstrings said "diameter": fixed. |
| E3 | `active_ax()` | Alexander 2010: abstract only; Dyrby 2013 and Sepehrband 2016 (secondary, full text) | differs: changes recovery (fixed) | Cylinder and Zeppelin shared nothing (two orientations, free D_perp, two D_par). Now one orientation, one D_par, tortuosity. |
| F1 | `verdict()` | Panagiotaki 2014: abstract only | source not verifiable | Sphere + stick + ball arithmetic checks; fixed diffusivities and an isotropic vascular compartment unverified. |
| F2 | `sandi()` | Palombo 2020: full text (Eqs. 4-10) | differs: changes recovery (fixed) | Powder-averaged model; the factory was oriented, fixed D_in, and used D_soma 1.58 instead of 3 um^2/ms. |
| F3 | `impulsed()` | Xu 2019: main text (Eq. 1); supplement (OGSE equations) not read | faithful (PGSE) | Sphere + Ball with D_in = 1.58 fixed equals Eq. 1 with the Gaussian-phase sphere to 1.5e-12. |
| I5 | `noddida_mte()` | none: no paper defines it | no published source (kept, made consistent) | Was the old aligned Stick + Zeppelin + Ball; now NODDIDA with one T2 per compartment. |

## D1 two_fascicle_noddi: cited Behrens et al. 2007

Source: Behrens, Berg, Jbabdi, Rushworth, Woolrich, NeuroImage 34:144-155
(author manuscript on PMC, full text), Eq. 1: S_i = S0[(1 - sum f_j) exp(-b_i d)
+ sum_j f_j exp(-b_i d r_i^T R_j A R_j^T r_i)], A = diag(1, 0, 0). N sticks and
one isotropic compartment, **one diffusivity d**, orientations (theta_j, phi_j),
automatic relevance determination on the fractions. It has no Watson dispersion,
no tortuosity and no CSF compartment. The paper does not define the factory.

Factory: two Watson bundles (each the NODDI bundle: Stick D_a = 1.7 and a
Zeppelin with D_par = D_a, D_perp = D_a (1 - f), its own mu, ODI and f_intra) and
a CSF Ball at 3 um^2/ms, three signal fractions.

| Difference | Effect |
|---|---|
| Behrens 2007 cited as the source | Citation error: the composition has no published source. The docstring now says it is this package's composition of two NODDI bundles (Zhang 2012) and names Behrens 2007 only as the motivation. The table says the same. |
| Extracellular signal averaged as signals | Inherited from the documented design choice at `noddi()` (4.5e-3 to 7.3e-3 of S0); not re-audited. |

Tests: `test_d1_is_two_noddi_bundles_and_a_csf_ball` (3e-6, tolerance 1e-4),
`test_d1_free_parameters_are_per_bundle_orientation_dispersion_fraction`,
`test_d1_citation_does_not_claim_behrens_defines_this_model` (failed before the
docstring fix).

Verdict: no published source for this composition. Kept, reported.

Removed: the entry was removed for lack of a published source; `two_fascicle_noddi()` and its tests are gone.

## E1 charmed: Assaf & Basser 2005

Source: the 2005 paper is not open; **abstract only** (CHARMED, high-b
high-angular-resolution data, population fraction of the restricted
compartment). What I did read is Assaf 2008 (E2), which restates CHARMED as
Eq. 1, E = f_h E_h + f_r E_r, with E_h Gaussian (Eq. 2) and E_r Callaghan's
cylinder (Eq. 3, perpendicular gradients, short pulses). The printed bracket of
Eq. 3 is not squared; as printed the sum does not reach 1 at q -> 0, the
squared (Callaghan) form does, and that is what the test reference uses.

Factory: `C3CylinderCallaghanApproximation` + `G2Zeppelin`, shared orientation
and D_par, D_perp = D_par (1 - f_r), free D_par, diameter and f_r.

| Difference | Effect |
|---|---|
| Tortuosity link, shared D_par, one free diameter | Unverified: the 2005 paper was not readable. Not changed. |
| Restricted compartment equals Eq. 3; composite equals Eqs. 1-2 | Matches to 6e-16 (perpendicular gradients, delta/Delta = 1.7 %; the factory's diffusion time is Delta). |

Tests (all pass): `test_e1_restricted_part_equals_callaghan_eq3`,
`test_e1_composite_is_population_weighted_hindered_plus_restricted`,
`test_e1_eq3_reference_is_normalised_at_zero_q`.

Verdict: source not verifiable. Docstring states what was and was not checked.

## E2 axcaliber: Assaf et al. 2008

Source: Assaf, Blumenfeld-Katzir, Yovel, Basser, MRM 59:1347 (author manuscript
on PMC, full text), Eqs. 1-5. Hindered Gaussian Eq. 2: exp(-b D_h) for the
perpendicular gradients, D_h free. Restricted Eq. 4: Callaghan cylinders over a
distribution of radii a_i, Gamma weights w(a) of Eq. 5 with parameters
(alpha, beta). Eq. 4 as printed weights each radius by f_i / (pi a_i^2), which
is not normalised and whose meaning (volume or number fraction) the text does
not say; the area-weighted normalised ensemble is the usual reading and is what
the factory does.

Factory: `DD1GammaDistributed([C3...])` + `G2Zeppelin`, free alpha, beta,
D_par, D_perp, mu and fractions. The distribution draws the cylinder *radius*
from Gamma(alpha, beta) and sets the diameter to twice the draw (mean diameter
2 alpha beta), area-weighted: Eq. 5 as printed. The docstrings of `axcaliber()`
and `DD1Gamma` called it a Gamma of the diameter; fixed.

| Difference | Effect |
|---|---|
| Docstrings said "diameter" | Cosmetic; corrected. |
| Intra-axonal diffusivity fixed (1.7 um^2/ms, a default of the cylinder) | The paper does not list D_r among the fitted quantities; consistent. |
| Unlinked D_par of cylinder and Zeppelin | Irrelevant for the paper's perpendicular-only protocol. |

Tests (all pass): `test_e2_gamma_distributed_restricted_signal_matches_eq4_eq5`
(any gradient direction, measured 2.1e-3 against the 30-step quadrature),
`test_e2_hindered_part_is_stejskal_tanner_with_a_free_dh`.

Verdict: faithful.

## E3 active_ax: Alexander et al. 2010

Source: Alexander, Hubbard, Hall, Moore, Ptito, Parker, Dyrby, NeuroImage
52:1374. **Abstract only** ("a four-compartment tissue model"). The model
structure is from two open papers that restate it: Dyrby et al., MRM 2013
(full text; "the intra- and extra-axonal compartments, ... a single cylinder
diameter ..., a cerebrospinal fluid compartment and a fourth, isotropically
restricted, compartment") and Sepehrband et al., NMR Biomed 2016 (full text;
"a bundle of cylinders with equal radii, and ... hindered water displacement,
with a tortuosity hindrance in the direction perpendicular to the axons",
tortuosity = d_par v_ec / (v_ic + v_ec)). Values of the fixed diffusivities in
the original (intrinsic, CSF) are not in these secondaries and were not
checked; the factory's D_csf = 3 um^2/ms is kept.

Factory before the fix: Cylinder + Zeppelin + Ball(3), but with two orientation
parameters, two independent parallel diffusivities and a free D_perp: the
"extra-axonal" compartment had no relation to the axons.

| Difference | Effect |
|---|---|
| Two orientations, two D_par, free D_perp | Changes recovery. Fixed: one orientation, one D_par, D_perp by tortuosity (two free scalars and one orientation, two angles, fewer). At the harness's midpoint parameters the signal moves by up to 9.0e-2 of S0. |
| No dot compartment | Documented: the dot belongs to the ex-vivo variant; not added (a new compartment is out of scope). |

Tests: `test_e3_restricted_and_hindered_share_one_orientation` and
`test_e3_extra_axonal_perpendicular_diffusivity_is_tortuous` (failed before the
fix, pass now), `test_e3_csf_ball_is_fixed_at_free_water_diffusivity`.

Verdict: differed (fixed). Source for the details beyond the structure
unverified.

## F1 verdict: Panagiotaki et al. 2014

Source: **abstract only** (Cancer Res 74:1902; the full text was refused). It
names a vascular, an extracellular and a restricted intracellular compartment
and a vascular pseudo-diffusivity. Open abstracts of later work describe the
same three-compartment structure (restricted sphere, isotropic extracellular
tensor, vascular stick), and one a variant with an orientation-averaged
("astro") stick; none is the 2014 paper.

Factory: Gaussian-phase sphere (diameter free), oriented Stick (mu, D free),
Ball (D free), three signal fractions.

| Difference | Effect |
|---|---|
| Vascular stick has an orientation | Unverified: tumour tissue is isotropic and a variant uses an orientation-averaged stick; whether the 2014 model does is unknown. Not changed. |
| Fixed diffusivities of the original | Unverified. |

Test: `test_f1_is_sphere_plus_stick_plus_ball` (composition arithmetic with the
SANDI Eq. 8 sphere, 1e-12).

Verdict: source not verifiable.

## F2 sandi: Palombo et al. 2020

Source: Palombo, Ianus, Guerreri, Nunes, Alexander, Shemesh, Zhang, NeuroImage
215:116835 (open access, full text), Eqs. 4-10. The model is the
direction-averaged signal (Eq. 10): (1 - f_ec)(f_in A_in + (1 - f_in) A_is) +
f_ec A_ec, with A_ec = exp(-b D_ec), A_in the powder-averaged stick (Eq. 7), A_is
the Gaussian-phase sphere (Eq. 8; the prefactor is 2(gamma g)^2 / D_is, the
printed fraction bar is lost in text extraction, and the standard Murday-Cotts
form confirms it). Six parameters, of which D_is is fixed to the free-water
value, ~3 um^2/ms (Sec. 2.2.3): f_in, f_ec, D_in, D_ec, r_s free.

Factory before the fix: `MultiCompartmentModel([sphere(D 1.58), Stick, Ball])`
with an oriented stick (mu free) and D_in fixed at 1.7.

| Difference | Effect |
|---|---|
| Oriented stick for a direction-averaged model | Changes recovery: the neurite compartment of Eq. 7 has no orientation. Fixed: spherical-mean model. |
| D_soma = 1.58 | Changes recovery. Fixed: 3e-9. Soma signal at 8 um, b = 6 ms/um^2, delta/Delta 6/20 ms: 0.590 -> 0.708; at 12 um 0.197 -> 0.274. |
| D_in fixed at 1.7 | Changes recovery: the paper estimates D_in. Fixed: free (parameter `C1Stick_1_lambda_par`, default range 0.1-3 um^2/ms). |

The JAX spherical-mean forward gains the sphere (Gaussian-phase, per-shell PGSE);
before, `solver="jax"` would have refused the model.

Tests: `test_f2_has_the_published_free_parameters` and
`test_f2_matches_eq10_with_soma_diffusivity_3_um2_per_ms` (failed before the fix;
2e-12 after), `test_f2_jax_spherical_mean_forward_matches_the_numpy_model`.

Verdict: differed (fixed). End-to-end recovery of a fitted voxel was not tested
(see "Not verified").

## F3 impulsed: Xu et al. 2019

Source: Xu, Jiang, Devan, Arlinghaus, McKinley, Xie, Zu, Wang, Chakravarthy,
Wang, Gore, MRM 83:2002 (author manuscript, main text). Eq. 1: S = v_in S_in +
(1 - v_in) S_ex, S_ex = exp(-b D_ex) with D_ex a constant, cells impermeable
spheres, **D_in fixed at 1.58 um^2/ms**. The cosine-modulated OGSE sphere
equations are in the Supporting Information (Eqs. S5, S6), which I did not read.

Factory: Gaussian-phase sphere with D = 1.58, Ball with free D, two fractions.

| Difference | Effect |
|---|---|
| none for PGSE | Equals Eq. 1 with the Gaussian-phase sphere to 1.5e-12. |
| OGSE sphere equations | Unverified (supplement). |

Test: `test_f3_is_v_in_sphere_plus_ball_with_din_1_58`.

Verdict: faithful for PGSE; OGSE unverified.

## noddida_mte

No paper defines NODDIDA with a T2 per compartment; MTE-NODDI (Gong et al. 2020)
is NODDI with per-compartment T2, whose text I did not read. After batch 1
`noddida()` is the Watson-dispersed Stick + Zeppelin of Jelescu 2016 with no
Ball; `noddida_mte()` still held the former aligned Stick + Zeppelin + Ball and
called itself "unconstrained NODDIDA". Made consistent: it is `noddida()` with a
T2 on the stick and one on the zeppelin (the Ball, its diffusivity and T2 are
gone), bounds as `noddida()`, and the docstring says the combination is not a
published model.

Tests: `test_noddida_mte_is_noddida_with_a_t2_per_compartment` (T2-weighted
Eq. 2 reference to 7e-5; equals `noddida()` to 1.5e-4 with T2 off),
`test_noddida_mte_docstring_matches_the_model` (both failed before).

Removed: the entry was removed for lack of a published source; `noddida_mte()` and its tests are gone.

## Not verified

- E1, E3 and F1: primary papers unreadable (abstracts only); structure checked
  against restatements (E1, E3) or an abstract (F1).
- F3: the OGSE sphere equations (supplement).
- F2: fits of simulated voxels through `sandi().fit(...)` were tried once (JAX and
  brute2fine, one noise-free voxel) and did not recover the generating
  parameters, but the same data pipeline did not recover `mcsmt()` either, so
  the test data path, not the model, is suspected; not pursued.
- Eq. 4 of Assaf 2008 is typeset ambiguously (weighting) and Eq. 3 prints a
  missing square; the references follow the standard forms.

# Reference catalogue audit, batch 3

Audited 2026-11-12 against `origin/main` e58d230. Predictions live in
`tests/test_catalog_audit_batch3.py` and were written before factory changes.
The first run had 4 failures, 5 passes and 3 strict xfails. Reading the final
Shi manuscript subsequently exposed its 1.56 (not 1.58) soma diffusivity;
the source-based no-exchange prediction was changed **before** that fix and
failed by 3.88e-4 of S0. Tests are offline CPU checks with independent
NumPy/SciPy equations. SI units throughout; diameter = twice paper radius.
There is no claim of end-to-end parameter recovery from these forward tests.

## Batch 3 summary

| ID | Factory | Primary-source access/read | Verdict |
|---|---|---|---|
| G2 | `karger_two_compartment()` | Kärger 1985 abstract only; full text unavailable. Olesen 2022 full text read for equations | Conditional match to constant-q two-pool equations; finite PGSE pulses approximate |
| G4 | `sandix()` | Olesen et al. 2022 full text, especially Sec. 2.1 and Eqs. 1-6 | Composition mistakes fixed; finite-pulse exchange still differs (strict xfail) |
| G5 | `exchange_impulsed()` | Shi 2025 full author manuscript (PMC12805662), plus 2024 preprint (PMC11312621) | Fixed Din; still not EXCHANGE (strict xfail) |
| H1 | `temporal_zeppelin_model()` | Novikov 2019 full text, Secs. 2.4, 3.2; Burcaw 2015 full text, Eqs. 24-25 | Transverse asymptotic law; description corrected; no published source found for this composition |
| I1 | `mte_ball_stick()` | Gong 2020 full text | Relaxation algebra matches; no published source found for this two-pool composition; retained |
| I2 | `mte_noddi()` | Gong 2020 full text, Sec. 2, Eqs. 1-8 | Relaxation algebra matches; C1 signal-averaged extracellular design retained (strict xfail) |
| I3 | `mte_sandi()` | Complete ISMRM 2023 conference abstract #0766, including Methods and figure captions; not a journal full text. Palombo 2020 full text for base SANDI | Base SANDI mistakes and attribution fixed; joint forward form of the abstract's relaxation law |

### Source access and limits

Primary URLs actually read (no model is marked verified from memory):

- [Kärger 1985 abstract](https://www.sciencedirect.com/science/article/pii/000186868580018X).
  The full publisher page returns HTTP 403 through the web tool; the Elsevier
  content endpoint returns an input error. Searches found no accessible
  author copy. Its equations were **not** verified against the 1985 text.
- [Olesen et al. 2022 author PDF](https://shemeshlab.org/wp-content/uploads/2022/02/1-s2.0-S1053811922001057-main.pdf),
  *Diffusion time dependence, power-law scaling, and exchange in gray matter*,
  NeuroImage 251:118976, doi:10.1016/j.neuroimage.2022.118976. This is the
  paper defining SANDIX, not an unpublished combination of Palombo and Kärger.
- [Shi et al. 2025 full manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC12805662/),
  *Comprehensive characterization of tumor therapeutic response via simultaneous
  mapping of cell size, density, and transcytolemmal water exchange*,
  MRI 122:110433, doi:10.1016/j.mri.2025.110433. The initially consulted
  [2024 preprint XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11312621/fullTextXML)
  is not the final publication; the audit and tests use the final main-text
  equations. The final supplement's waveform discretisation was not read.
- [Novikov et al. 2019 full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC6481929/),
  doi:10.1002/nbm.3998, and [Burcaw et al. 2015 full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC4446209/),
  doi:10.1016/j.neuroimage.2015.03.061. The latter supplies the finite-pulse
  transverse law, of which the implemented log+3/2 is an asymptote.
- [Gong et al. 2020 full published PDF](https://discovery.ucl.ac.uk/id/eprint/10098354/1/1-s2.0-S105381192030392X-main.pdf),
  doi:10.1016/j.neuroimage.2020.116906. Supplementary simulations were not read.
- [ISMRM #0766 complete abstract](https://cds.ismrm.org/protected/23MProceedings/PDFfiles/0766_IoX8F2Pxv.html)
  (accessible from the [archive synopsis](https://archive.ismrm.org/2023/0766.html)),
  doi:10.58530/2023/0766. The archive landing page alone omits Methods.
- [Palombo et al. 2020 full PDF](https://discovery.ucl.ac.uk/10095891/1/1-s2.0-S1053811920303220-main.pdf),
  doi:10.1016/j.neuroimage.2020.116835, base equations as in batch 2 F2.

## G2 karger_two_compartment: Kärger 1985

Two isotropic Gaussian pools, fraction f and 1-f; the factory's `kappa` is
pool-1 efflux in s^-1. Detailed balance gives reverse rate kappa*f/(1-f).
In the constant-q approximation the signal is
`[1,1] exp([[ -b D1-k t, k_reverse t ],[ k t, -b D2-k_reverse t ]]) [f,1-f]`.
This equation is verified from **Olesen Eqs. 2, 4-6**, not the unreadable
Kärger full text. Both diffusivities and f/kappa remain free and unchanged.

The test `test_g2_matches_independent_karger_matrix` predicts 1e-12 agreement
including b0. PGSE substitutes t=Delta-delta/3; this is the existing
constant-q approximation, not a finite-pulse waveform solution. The docstring
now states the rate convention and access limit. Fitted values do not change.

Verdict: conditional match to the later primary restatement, not full
verification of the 1985 publication.

## G4 sandix: Olesen et al. 2022

Sec. 2.1.2 exchanges **neurites with extracellular water**, while soma
exchange is neglected. Eq. 3 is solved for each stick angle before averaging;
averaging the sticks before exchange is a different model. The independent
soma is a GPA sphere with D0 fixed at **2 um^2/ms** (Sec. 2.1.1). The models
include an immobile signal offset f_im. D_n and D_e are estimated (Table 1).

| Difference before fix | Action and effect |
|---|---|
| Soma exchanged with Ball; neurites impermeable | Fixed topology: non-exchanging sphere plus coupled powder-averaged Stick/Ball plus Dot offset |
| Oriented stick; D_n fixed at 1.7 | Fixed: direction-averaged model, free D_n |
| Soma D0=1.58 | Fixed: 2e-9 m^2/s |
| No immobile offset | Fixed: `S1Dot()` with its own total signal fraction |
| No defining SANDIX citation | Fixed: Olesen 2022 |
| Effective-time constant-q exchange rather than finite-pulse Eq. 3 | **Unfixed G4-finite-pulse**; strict xfail asserts the full finite-pulse ODE |

Implementation uses existing `X1NEXI` to average coupled kernels, without a
new compartment class. `partial_volume_0/1/2` now mean soma/pair/immobile;
`X1NEXI_1_f` is the neurite fraction *within* the pair, and
`t_ex=1/(r_n+r_e)`, r_n=(1-f)/t_ex. The old sphere/pair and oriented-stick
names are gone, with no aliases. Old kappa referred to soma efflux and has no
biophysical correspondence to the new neurite rate. The inherited t_ex
fitting range is 0.001-0.5 s instead of the old kappa range 0.1-200 s^-1;
these ranges are not equivalent. All parameter/meaning/bound changes are
flagged for the lead. They are not claims that the source prescribes bounds.

Tests: `test_g4_composition_and_narrow_pulse_kernel` pins the inventory and
independent sphere + angular matrix-exponential + offset (1e-8);
`test_g4_matches_published_finite_pulse_rate_equations` retains the published
Eq. 3 assertion at delta/Delta=12/20 ms, strict xfail at 1e-7.
The JAX spherical-mean path gains the Dot; its forward agrees with NumPy.

Verdict: composition fixed; finite-pulse SANDIX remains unimplemented.

## G5 exchange_impulsed: Shi et al. 2025

The final paper Sec. 2.2 fixes intrinsic Din (usually **1.56 um^2/ms**).
That corrects the factory's inherited 1.58, without changing `impulsed()`
(F3, out of scope). At zero exchange, d=14 um, f=0.55, delta/Delta=12/74 ms,
b=0.25/1/1.8 ms/um^2, the total signal changes by up to 3.88e-4 of S0.
Consequently a fitted diameter/fraction can shift even at zero exchange.
No fit recovery experiment was performed.

The remaining implementation is the effective-medium substitution
R_in=-log(S_sphere), R_ex=b D_ex in a two-site Kärger matrix. Shi explicitly
explains why this is insufficient (Sec. 2.1.1). The publication requires:

- Eq. 3 crossing probability p using cell radius, Din and kin;
- Eq. 4 two-mode intracellular signal, combining restricted diffusion and
  hindered diffusion, D_in,h = v_in ADC_r + v_ex D_ex;
- Eq. 5 edge-enhancement correction: magnetisation efflux differs from water
  molecular efflux, while k_ex^m is approximated by kin*v_in/v_ex;
- propagation for T=**Delta+delta** (Eq. 1), not Delta-delta/3;
- high-frequency extracellular diffusivity slope when f >= 100 Hz (Sec. 2.2).

The existing building blocks do not express those coupled, size-dependent
corrections. No new compartment class was added. `kappa` remains the generic
model's magnetisation efflux, **not** the published corrected molecular kin;
clinical validation claims were removed from the factory docstring.

Tests: `test_g5_no_exchange_limit` compares to the independent GPA sphere
mixture at the corrected Din (1e-8). Strict xfail
`test_g5_matches_shi_pgse_exchange` asserts final Eqs. 1-5 (PGSE constants
alpha=2.39, gamma1=0, gamma2=0.83, gamma3=2.88), 1e-7. It fails because the
published corrections remain absent, not because their expected signal was
changed to the current approximation. OGSE and arbitrary-waveform EXCHANGE
are unverified, including the supplement's numerical waveform treatment.

Verdict: fixed Din; still differs substantially from published EXCHANGE.

## H1 temporal_zeppelin_model: Novikov et al. 2019

The code changes **perpendicular**, not parallel diffusivity:
`Dperp=lambda_inf+A*(log(Delta/delta)+3/2)/(Delta-delta/3)`;
Dpar is constant. A has units m^2. Novikov Eq. 2.34 discusses transverse
packing disorder's log(t)/t tail; a longitudinal 1/sqrt(t) correction is a
different disorder regime. Burcaw Eqs. 24-25 give
`Dperp=Dinf+A*F(Delta/delta)/(Delta-delta/3)`, with
`F(x)=x^2/2*log(1-1/x^2)+log(x^2-1)/2+x*log((x+1)/(x-1))`.
The implemented `log(x)+3/2` is its large-x asymptote, retained as the
existing definition and now explicitly described as approximate.

The Ball is **fixed** at 3e-9, not free. The composition has no intra-neurite
stick and thus is not the Standard Model of Novikov Sec. 3.2. No published
source was found defining this particular temporal Zeppelin+fixed-CSF model;
it is reported and retained, not silently removed. The corrected description
changes no signal, fitted value, parameter or bound.

Test: `test_h1_perpendicular_log_time_dependence_and_csf` checks the actual
asymptotic transverse law and composition at 1e-12. It does not assert that
this is the full F(x) finite-pulse expression.

Verdict: description corrected; faithful asymptotic building block, catalogue
composition with no located defining source. Full finite-pulse F(x) is not implemented.

## I1 mte_ball_stick: Gong et al. 2020

Gong defines **MTE-NODDI**, not this Stick+Ball model. No defining publication
for this two-pool factory was found. It is retained and explicitly labelled a
generic composition for the lead to review. It has independent diffusivities
(unlike Behrens's single-diffusivity ball-and-stick), which are not changed.

The relaxation algebra follows the b0 compartment sum in Gong Eq. 8:
`S/S0=sum_i f_i^0 exp(-TE/T2_i) A_i`, with f_i^0 defined at **TE=0**.
At finite TE these are not the fractions of the TE-normalised diffusion
signal: divide each decayed weight by their sum to obtain those fractions.
`test_i1_generic_two_pool_relaxation` checks the independent oriented
stick/ball relaxation sum at 1e-12, including b0. It verifies algebra,
not the existence of a published two-pool model. No fitted values change.

Verdict: source does not define the composition; relaxation algebra matches.

## I2 mte_noddi: Gong et al. 2020

Sec. 2 and Eqs. 1-8 introduce separate intra-neurite, extra-neurite and CSF
T2. Eq. 4 predicts
`f_in(TE)=f_in^0 exp(-TE/T2in)/(f_in^0 exp(-TE/T2in)+(1-f_in^0) exp(-TE/T2en))`;
Eq. 5 applies the corresponding weighting to CSF. Eq. 8 supplies the total
b0 decay. The factory's inner fraction is f_in^0 within tissue; outer
fractions are tissue and CSF at TE=0. All three T2 parameters are exposed.
Fixed Di=1.7e-9 and Diso=3e-9 match Gong Sec. 3.1.2.

**As C1, the extracellular signal is the orientation average of microscopic
Zeppelin signals, rather than exp(-b g^T <D> g). This documented design choice
is retained.** Passing `test_i2_preserves_documented_signal_average_and_t2_weights`
uses independent normalised Watson quadrature (2e-4). Strict xfail
`test_i2_published_extracellular_gaussian` asserts the published
averaged-tensor Gaussian to the same tolerance (finding I2-EA).

Gong's estimation procedure first fits NODDI per TE, then Eq. 6/7 fraction
relations and Eq. 8 b0 relaxation; this factory is a joint forward fit with
one ODI, TE=0-fraction tortuosity, and three T2 values. These are different
estimators, not evidence of reproducing the staged algorithm's estimates.
No implementation or fitted-value change in this entry.

Verdict: relaxation algebra matches; retained C1 extracellular difference and
joint-estimation scope stated explicitly.

## I3 mte_sandi: Gong et al., ISMRM 2023 #0766

The complete conference abstract's authors are **Gong, Tax, Mancini, Jones,
Zhang and Palombo**, not Palombo, Gong and Shemesh. Methods estimates SANDI
per TE, then fits `S(b=0,TE)*f_soma(TE)=S0_soma*exp(-TE/T2_soma)`,
analogously for neurite and extracellular signals, and normalises the
extrapolated S0_i to get relaxation-independent fractions.

The factory uses the equivalent compartment relaxation sum as a **joint
forward model**, not the abstract's staged estimator. Its diffusion base
must remain SANDI (Palombo Eq. 10, as F2): direction average, soma D=3e-9,
free neurite and extracellular diffusivities. Fixed base errors: oriented
stick -> spherical-mean model; D_soma 1.58 -> 3; D_neurite fixed at 1.7 ->
free. `OccupancyGatedModel_2_mu` is removed; `..._2_lambda_par` is added;
three T2 values and all fraction names remain. These definition/parameter
changes are flagged for lead review. The abstract itself does not specify
new diffusion bounds or a new fixed soma diffusivity; the base follows F2.

Tests: `test_i3_direction_average_free_dn_and_compartment_t2` checks inventory
and independent direction-averaged sphere/stick/ball sum with separate T2
values, including b0, 2e-5. JAX gains the T2-gated spherical means (one
TransverseRelaxation factor only; unsupported combinations raise), including
compartment-weighted b0 decay instead of forcing it to one. Existing
relaxation-free spherical means keep their b0 convention. Fits now estimate
neurite diffusivity and cannot estimate an irrelevant fibre orientation;
fractions refer to TE=0 signals, not TE-weighted observations.

Verdict: base-model and attribution mistakes fixed; conference abstract's
relaxation law matched. Its staged estimation and parameter recovery were
not reproduced.

## Batch 3 validation and remaining limits

- Batch-3 tests: **12 passed, 3 strict xfailed** (15 cases). Both shell and
  measurement-broadcast JAX outputs are covered; a two-TE case checks both
  b0 relaxation sums independently.
- Regression runs: all eight files under `tests/` were exercised across two
  sequential runs, plus catalogue construction/DOI checks, core spherical-mean
  T2 parity, JAX spherical means and multi-compartment tests: **238 distinct
  passing tests, 3 strict xfails** in total. The full package suite, CI runners
  and GPU execution were not run. All local computation was CPU, cost $0.
- The 1985 full text remains unreadable; G5's waveform supplement and both
  multi-TE publications' staged estimation procedures were not reproduced.
  No end-to-end recovery claim is made for the changed factories.
- Parity was recorded exclusively with `harness.py rebaseline --only` for
  the SANDIX/EXCHANGE entries and the new SANDIX/MTE-SANDI spherical means:
  **8 added, 2 removed, 4 modified**; **605 other manifest entries and their
  arrays are byte-for-byte unchanged** against `origin/main`. The stored
  environment block was preserved.
- Full `harness.py check` emitted **617 cases** and exited 1 with **89
  mismatches in unselected cases**; none of the selected cases failed. This
  environment differs from the stored record (Python 3.11.18 vs 3.12.3,
  NumPy 2.4.6 vs 2.5.4, SciPy 1.17.1 vs 1.18.1, JAX/JAXlib 0.10.2 vs
  0.12.0.1, and dependency revisions). A same-environment base comparison
  was not performed, so the exact cause of those mismatches is unverified.
  No unrelated key was re-recorded to hide them.
