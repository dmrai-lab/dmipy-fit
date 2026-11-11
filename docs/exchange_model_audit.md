# NEXI and FEXI correctness audit

`nexi()` now fits the powder-averaged Stick + independent isotropic Ball
model defined by Jelescu et al. (2022), with four free parameters: f, Di, De
and t_ex in seconds. The former orientation-resolved, tortuous Stick +
Zeppelin variant retains its numerical behaviour as `stick_zeppelin_karger()`.
There is no compatibility alias for that variant under the NEXI name.
`fexi()` is a constrained two-pool forward model on a
user-supplied acquisition, not the phenomenological AXR analysis. The latter is
already available as `FEXIApparentExchange`.

## 1. Published NEXI and the factory

Primary source: [Jelescu et al., NeuroImage 256 (2022), 119277](https://orca.cardiff.ac.uk/id/eprint/150151/4/1-s2.0-S1053811922003986-main.pdf),
section 2.1.2, Eqs. (3)--(6). The equations and their numbering were checked
against the paper, rather than reconstructed from memory.

The two compartments are intra-neurite zero-radius sticks, axial diffusivity
Di, relative signal fraction f, and Gaussian extra-neurite water. The authors
first present the more general coherent Stick + Zeppelin exchange kernel in
Eqs. (3)--(5). Their specific implementation called NEXI then sets the
extra-neurite diffusivity isotropic, De_parallel = De_perpendicular = De, and
uses the powder average in Eq. (6). There is no tortuosity constraint linking
De to f or Di. There is no soma or CSF compartment.

For u = cos(theta), a = Di u², c = De, q² = b/t, and h = 1/(q² t_ex),
Eq. (4) gives

    d_minus/plus = (a + c + h ∓ sqrt((c-a+(2f-1)h)² + 4f(1-f)h²))/2.

Eq. (5) supplies w = [f a + (1-f)c - d_plus]/(d_minus - d_plus),
and Eq. (3) is

    K(q,t,u) = w exp(-b d_minus) + (1-w) exp(-b d_plus).

Eq. (6) is S_bar/S0 = integral_0^1 K(q,t,u) du. At q=0 or degenerate
eigenvalues one takes the continuous limit; the proposed reference predictor
uses the equivalent symmetric matrix exponential to avoid those divisions.

Detailed balance is f r_ie = (1-f) r_ei. Immediately following Eqs. (4)--(5),
the paper defines t_ex = 1/(r_ie + r_ei). Consequently

    r_ie = (1-f)/t_ex,  r_ei = f/t_ex,
    kappa_code = r_ie,  t_ex = (1-f)/kappa_code.

In particular, t_ex is not the intra-neurite residence time 1/r_ie. The four
free tissue parameters are [f, Di, De, t_ex]; S0 is an overall signal scale.
The fitted signal is powder-averaged and has no fitted orientation or ODF.
The kernel assumes Gaussian, time-independent compartment diffusivities,
barrier-limited exchange, and narrow diffusion pulses.

The following comparison describes the preserved directional variant;
`nexi()` now implements the published column (subject to the finite-pulse approximation of the published narrow-pulse
kernel; it has no compartment-signal floor).

| Property | Published NEXI | `stick_zeppelin_karger()` | Effect on recovered values |
|---|---|---|---|
| Intra-neurite compartment | Stick | Stick | Same compartment assumption. |
| Extra-neurite compartment | Independent isotropic Ball | Coaxial Zeppelin | Changes signal and meaning of De; can bias every tissue estimate under model mismatch. |
| Tortuosity | None | De_perp = (1-f) De_par | Couples fraction to diffusion anisotropy; can move f, diffusivities, and exchange. Powder averaging alone would not remove this difference. |
| Signal / orientation | Eq. (6) powder average; no mu | Directional kernel; shared mu with two angles | Fits a different signal and two extra scalars. Averaging measured data alone does not turn this factory into Eq. (6). |
| Exchange coordinate | t_ex | One-way kappa | Reparameterisation alone preserves predictions after t_ex=(1-f)/kappa. Calling kappa permeability or using 1/kappa as t_ex changes the reported quantity. |
| Free tissue parameters | Four scalars | Di, De_par, f, kappa, plus two angles: six scalars | Different inverse problem. The factory's bounded kappa interval also induces fraction-dependent t_ex bounds. |
| Pulse treatment | Eqs. (3)--(5) are narrow-pulse expressions | Plain PGSE uses b and t=Delta-delta/3 in the closed kernel | This is a finite-pulse approximation, not an exact finite-PGSE solution with exchange. Both require a pulse-width applicability check. |
| Double encoding | Outside Eqs. (3)--(6) | Can dispatch to a FEXI waveform | A useful extension, but not the original NEXI experiment or equation. |
| Very attenuated kernels | No imposed floor | No floor (the former 1e-10 floor is removed) | None; see EXCHANGE-AUDIT-3. |

In the absence of exchange,

    S_bar/S0 = f sqrt(pi/(4 b Di)) erf(sqrt(b Di)) + (1-f) exp(-b De).

For b Di >> 1, the stick factor tends to sqrt(pi/(4 b Di)). The zero-b
limit is one.

The brief's fast-exchange limit needs a scope qualification: a *local kernel*
becomes exp[-b(f Di u² + (1-f)De)]. Eq. (6) then still averages over u:

    S_bar_fast/S0 = exp[-b(1-f)De]
                   sqrt(pi/(4 b f Di)) erf(sqrt(b f Di)).

It does not become exp[-b(f Di/3 + (1-f)De)]. This is the residual orientation
heterogeneity discussed after Eq. (7); the paper distinguishes this analytic
local-exchange construction from exchange across differently oriented neurites
in a connected extracellular medium at very long times. The tests preserve
Eq. (6), including this distinction.

## 2. FEXI and AXR

Primary publication: [Lasič et al., MRM 66 (2011), 356--365](https://doi.org/10.1002/mrm.22782).
The accessible publisher page exposed its abstract but redirected full-text
and PDF requests to that abstract. **The 2011 equation numbering and exact
original fitting procedure were not verified.** The equations below are
reconstructed from two-site exchange and corroborated by the same research
group's accessible [Lampinen et al., MRM 77 (2017), Theory section](https://pmc.ncbi.nlm.nih.gov/articles/PMC5324642/).
Numbers from the later paper must not be presented as 2011 equation numbers.

FEXI applies a diffusion filter of b_f, stores magnetisation longitudinally
over a variable mixing time t_m, and measures a diffusion detection block of
b_d. Under the two-Gaussian-pool interpretation, the filter preferentially
attenuates the faster pool and changes the *surviving signal fractions*, not
the equilibrium physical volume fractions. Neglecting exchange during the
filter, the weights just afterwards are proportional to

    [f exp(-b_f Ds), (1-f) exp(-b_f Df)].

They relax towards equilibrium during mixing. With equilibrium generator

    K = [[-k12, k21], [k12, -k21]],
    f k12 = (1-f) k21,

the nonzero relaxation eigenvalue is -(k12+k21). Thus

    ADC_eq = f Ds + (1-f) Df,
    ADC'(t_m) = ADC_eq [1 - sigma exp(-AXR t_m)],
    sigma = 1 - ADC'(0)/ADC_eq,
    AXR = k12 + k21 = kappa_code/(1-f) = 1/t_ex.

The detection model is S = S0_series exp(-b_d ADC'(t_m)); without the filter,
ADC = ADC_eq. Equal relaxation rates prevent differential T1/T2 from changing
pool fractions. For ideal well-mixed two-site Gaussian exchange the ADC
recovery is exact at infinitesimal detection b. Finite encoding durations
can reduce filter efficiency without changing the mixing eigenvalue. At finite
detection b the logarithm of the biexponential signal contains higher
cumulants, so AXR fitted with a monoexponential detection model can be biased.
If the filter has no contrast (equal diffusivities, no filter, or infinitely
fast exchange), sigma vanishes and AXR cannot be identified from this recovery.

| Property | FEXI / AXR interpretation | Implementation | Effect on recovered values |
|---|---|---|---|
| Two pools | Slow/fast Gaussian pools motivate the analysis | `fexi()` wraps two Balls | Appropriate forward idealisation, not itself the model-free AXR estimator. |
| Fast diffusivity | AXR fit need not know either pool diffusivity | `fexi()` fixes Df = 2e-9 m²/s | Additional biophysical assumption; wrong Df can shift fitted Ds, f, and kappa. |
| Slow label | Ds < Df | No ordering constraint; free Ball range can exceed Df | Pool interpretation can reverse; kappa always means first-to-second rate. |
| Filter and mixing | Required diffusion-filter / mixing / detection experiment | FEXI acquisition invokes waveform propagation; plain PGSE is also accepted | A single PGSE acquisition is not a FEXI experiment. No filter is inferred merely from the factory name. |
| Fitted quantity | ADC_eq, sigma, AXR, and signal scale | `fexi()`: Ds, f, kappa; `FEXIApparentExchange`: ADC_eq, sigma, AXR plus S0 per series | Different inverse problems. Only the latter directly implements the phenomenological ADC equation. |
| Rate | AXR = k12+k21 | kappa = k12 in forward model | Convert by 1/(1-f); comparing the two raw rate values is wrong. |
| Statistical fitting | ADC recovery equation | AXR code fits signal jointly, projecting out per-series S0, using bounds and multiple starts | Mathematical recovery equation matches; noise weighting and active bounds can affect estimates. Original 2011 optimiser details are unverified. |
| Finite detection b | Low-b monoexponential approximation | AXR code applies it at all supplied b_d | Higher cumulants can bias recovered AXR; no automatic small-b validation. |
| Waveform weighting | Consistent diffusion weighting for prediction and analysis | Replay's b_from_gradient and the exact held-waveform integral differ at O(dt^2) | No/fast-exchange limit failures at the declared b; EXCHANGE-AUDIT-4. |
| Plain PGSE / attenuation floor | Same two-site limits | Shares the closed-path approximation; no floor | EXCHANGE-AUDIT-2 (documented) and -3 (resolved). |

## 3. Numerical checks and open findings

`tests/test_exchange_model_audit.py` uses NumPy/SciPy on CPU. Each test's
prediction and tolerance was specified in its docstring before execution.
Direct two-site ODE oracles construct K independently and use `scipy.linalg.expm`;
the published NEXI oracle separately implements Eqs. (4)--(5) and adaptive
quadrature. The AXR synthetic oracle propagates filter, mixing, and detection
matrices independently, including exchange during both encoding blocks.

Checks cover both existing factories' no-exchange and fast-exchange limits,
the published powder/no-exchange stick law through b Di=100, the published
fast-exchange powder limit, the Kärger closed form against ideal narrow-pulse
PGSE, finite rectangular PGSE by time-ordered exponentials, the waveform
propagator, AXR from independent two-site data, and AXR from `fexi()` itself.

The finite-PGSE oracle uses 512 midpoint matrix exponentials per gradient lobe
and one exact plateau exponential; its 256/512-step difference is checked to
be below 2e-7 absolute. Infinitely fast exchange is tested with large finite
rates and a stated tolerance, and also with the explicit limiting expression
in the new NEXI reference predictor.

| Audit ID (local finding, not a remote issue number) | Evidence | Disposition |
|---|---|---|
| EXCHANGE-AUDIT-1 | The former factory was directional Stick + tortuous Zeppelin. | Resolved: renamed `stick_zeppelin_karger()`; `nexi()` is now a spherical-mean Stick + Ball fitter with t_ex in seconds. The original equality assertion passes without xfail. |
| EXCHANGE-AUDIT-2 | At delta=18 ms, Delta=40 ms, b=2.4e9, f=0.35, kappa=17/s, closed-path NEXI variant gives 0.02358852056 versus ODE 0.02349958737 (0.378% high); two-Ball FEXI on PGSE gives 0.1339173042 versus 0.1307964269 (2.386% high). Narrow-pulse and waveform checks pass. | Not a defect: the published models are defined with the narrow-pulse kernel at t = Delta - delta/3. The xfail is now a passing test asserting the approximation at its measured size (0.378 % and 2.386 %); the exact finite-pulse route remains the waveform propagator, used for double encoding and arbitrary waveforms. |
| EXCHANGE-AUDIT-3 | At b=1e11, kappa=0, the E_i>=1e-10 floor replaces the true small compartment signals. | Resolved: no floor. Gaussian pools form their exponents directly; a restricted pool's -log(E) is guarded by `CompartmentSignalDomainError`. The xfail mark is removed (NumPy and JAX, float64 and float32). |
| EXCHANGE-AUDIT-4 | FEXI delta=4 ms, Delta=10 ms, dt=0.2 ms reports filtered b about 1.19999987e9, while exact integration of its piecewise-constant effective gradient gives about 1.19957310e9 (0.0356% less). No-exchange signal is 0.30323503 versus 0.30315341 at declared b; fast exchange also fails. | Not a defect: the package's b is dmipy-replay's `b_from_gradient` (trapezoid over a rectangular moment); the exact integral of the held waveform differs by O(dt^2) (0.036 % here). The two xfails are passing tests that state both quadratures and use the exact integral for the waveform route. |

The last finding is consistent with reported b using trapezoidal integration
of sampled squared moments, whereas `karger_waveform` integrates q(t)²
continuously within each constant-gradient interval. This attribution follows
from source inspection and an independent integral, not from a change to
either dependency.

The small-detection-b AXR check on the actual FEXI waveform gives about
7.9925/s for true AXR=8/s and k12=5.2/s, inside its predeclared 1% tolerance.
These checks establish ideal-model consistency, not biological validity or
parameter identifiability for an arbitrary demo protocol.

## 4. Proposed fixes and decisions for the lead

* **NEXI model definition (implemented):** `reference_models.nexi()` returns
  a `MultiCompartmentSphericalMeanModel` containing `X1NEXI`, which composes
  `X0GeneralizedKarger(C1Stick(), G1Ball())` and averages its coupled kernels
  with 64-point Gauss-Legendre quadrature. Its JAX twin uses the same nodes
  and the analytic Karger twin. All four tissue parameters are required:
  `X1NEXI_1_f`, `X1NEXI_1_D_i`, `X1NEXI_1_D_e`, `X1NEXI_1_t_ex`.
  Diffusivities use m²/s, exchange time uses seconds, with fitting bounds
  f=0.01--0.99, Di/De=0.1--3 µm²/ms and t_ex=1--500 ms. There is no mu or
  tortuosity. The independent `nexi_jelescu2022` reference is retained.
  The NumPy fitter and both JAX precisions are tested on 36 tissue settings
  spanning f=0.2/0.5/0.8, Di=1.2/2.8, De=0.6/1.8 µm²/ms,
  t_ex=5/30/200 ms and 15 b/Delta pairs (b=0/0.2/1/3/6 ms/µm²,
  Delta=15/40/80 ms, delta=4 ms). NumPy/float64 tolerance is rtol=2e-12,
  atol=2e-14; float32 tolerance is rtol=2e-5, atol=2e-7.
  The directional variant's callers and parity cases now use
  `stick_zeppelin_karger()`; its predictions and assertions are preserved.
* **Time definition:** the reference accepts b and t explicitly, with b=q²t;
  it does not silently substitute Delta-delta/3 for a finite-pulse experiment.
  t_ex=inf and t_ex=0 explicitly select the two mathematical limits.
* **FEXI analysis definition:** use `FEXIApparentExchange()` for the published
  ADC-recovery analysis. Retain `fexi()` only if the demo labels it as a
  constrained two-pool forward fit. If fitting both diffusivities is wanted,
  `karger_two_compartment()` already exposes them; removing the fixed Df
  assumption or imposing a pool ordering requires the lead's decision.
* **Finite pulses:** the published Karger-based models keep the narrow-pulse
  kernel at t = Delta - delta/3 and state the approximation (asserted at its
  size in the audit tests). The waveform propagator serves double encoding and
  arbitrary waveforms.
* **High attenuation:** resolved; no signal floor.
* **Waveform b:** dmipy-replay's `b_from_gradient` is the package's b; the exact
  held-waveform integral differs by O(dt^2) and is used only as the oracle of
  the waveform route in the tests.
* **Permeability and units:** kappa and AXR have units s^-1; permeability P has
  units length/time. In a well-mixed two-pool geometry with membrane area A
  and volumes Vi, Ve, k12=P A/Vi, k21=P A/Ve. Thus
  P=kappa_code Vi/A=AXR(1-f)Vi/A. The Stick signal does not estimate Vi/A
  (e.g. cylinder radius/2); the demo must supply the actual geometry and
  consistent units. Without that input a true-permeability comparison is
  undefined, even with the correct published model.

The initial audit corrected documentary inaccuracies in place: the directional
variant's identity, the FEXI forward/AXR distinction, the finite-PGSE exactness
claim, and the finite-b ADC/first-cumulant statement. The NEXI follow-up then
changed the factory definition as described above and preserved the renamed
variant's equations and parameter bounds. Finding 3 was then resolved; findings 2 and 4 are documented approximations
and quadratures asserted at their measured size, with no xfail marks.

## 5. Initial audit execution

Only the new test file is run with the requested interpreter, this worktree
and the supplied analytic source on PYTHONPATH, JAX_PLATFORMS=cpu, and pytest's
cache provider disabled. No required import was missing. The venv reports
that optional numba is absent; no installation or dependency checkout was
needed. No existing suite or GPU execution is claimed, and nothing is committed.

Initial run: 18 passed, 5 xfailed, 2 failed in 1.17 s. The two additional
failures became EXCHANGE-AUDIT-4 strict xfails without changing their
predictions, tolerances, or assertions. Final run: **18 passed, 7 xfailed in
1.36 s**, exit status zero. All seven xfails are strict and represent the
open findings above, rather than claims that those physical limits pass.

## 6. Published NEXI follow-up execution

Affected files were run individually with the supplied interpreter, worktree
and analytic source on PYTHONPATH, JAX_PLATFORMS=cpu, and
`python -m pytest -q <file> -p no:cacheprovider`:

| File | Result |
|---|---|
| `signal_models/tests/test_nexi.py` | 42 passed |
| `jax/tests/test_nexi_jax.py` | 77 passed, 1 skipped |
| `jax/tests/test_fexi_karger_jax.py` | 2 passed |
| `signal_models/tests/test_filter_exchange.py` | 14 passed |
| `custom_optimizers/tests/test_reference_models_catalog.py` | 96 passed |
| `tests/test_exchange_model_audit.py` | 19 passed, 6 strict xfailed |
| `jax/tests/test_spherical_mean_jax.py` | 6 passed, 6 failed at missing jaxopt import |
| `jax/tests/test_sm_zeppelin_jax.py` | 7 passed |

Paths in the first five rows and the last two rows are relative to
`dmipy_fit/`. No dedicated layout test was present; the NEXI twin and
existing spherical-mean dispatch tests were included.

Across the 36 tissue settings and 15 b/Delta pairs above, maximum absolute
errors versus `nexi_jelescu2022` were **5.8287e-16** (NumPy), **5.5511e-16**
(JAX float64), and **3.1364e-7** (JAX float32). Both JAX precisions compiled
and produced finite gradients for every free parameter. The NumPy fitter
recovered all four parameters within 0.2% from a 5% perturbed initial guess.

The environment lacks the declared JAX-extra dependency `jaxopt`, and its
site-packages directory is read-only, preventing installation. The new JAX
fit test skips explicitly when that optional dependency is absent; the six
existing spherical-mean fit tests retain their assertions and fail at its
import. End-to-end JAX fitting and GPU execution are therefore unverified.

All 614 parity archive arrays still match their recorded hashes; six case
IDs and archive keys were renamed to `ref.stick_zeppelin_karger`, with their
values and hashes preserved. The complete parity harness was not run.
Nothing was committed.
