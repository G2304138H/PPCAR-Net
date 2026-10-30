# FFR estimation along the reconstructed centreline

Branch `ffr-estimate`. Module `vessel_code/ffr/`, configs `configs/ffr/`, tests `tests/test_ffr.py`.

Input: one reconstructed artery as vessel code, `[M, N, 4]` = branch centreline xyz + lumen radius (mm), with
branch existence. Output: hyperemic FFR(s) = Pd(s)/Pa at every centreline sample of every branch, lesion
table (MLD, %DS, length, ΔFFR), figures, and optionally a 3D CFD case.

## 1. Method survey and choice

| Family | Representative work | Physics | Inputs | Notes |
|---|---|---|---|---|
| 3D CFD on CT | FFR-CT (HeartFlow; Taylor et al. 2013) | Navier–Stokes, lumped outlets | CTA lumen | Reference standard for virtual FFR; hours per case |
| 1D / reduced order | Blanco et al. 2018 (1D vs 3D, 52 pts); Boileau et al. 2018; Fossan et al. 2018 | 1D mass/momentum + stenosis loss | Centreline + area | 1D−3D FFR difference 0.00 ± 0.03 with a stenosis model; seconds |
| Analytic stenosis models | Young & Tsai 1973; Gould; Huo et al. 2012 | Lumped ΔP(Q) | Lesion geometry | Building block of the 1D model |
| Angiography-derived (fixed/contrast flow) | QFR (Tu et al., FAVOR), vFFR (CAAS, FAST), caFFR (FLASH FFR), AccuFFRangio | 3D-QCA + reduced-order ΔP | 2 views ≥ 25–30° apart, frame count, Pa | Per-vessel accuracy ~ 90–95% vs wire FFR |
| Network/resistance | FFRangio (CathWorks) | Lumped resistances on the whole tree | 2–3 views | Whole tree, ~minutes |
| Data-driven surrogates | Itu et al. 2016 (ML on reduced-order data); GNN/mesh surrogates | Learned | Geometry features | Need a physics model for training labels |

PPCAR-Net gives exactly what the 1D family needs — explicit branch centrelines, dense radii and branch
topology — with no graph extraction. The default is therefore a **steady hyperemic 1D network model**
(Blanco-type, with Young–Tsai stenosis losses), which the literature shows agrees with 3D CFD to within the
FFR measurement repeatability while running in seconds; that also makes Monte Carlo uncertainty affordable.

## 2. 1D model (`solver1d.py`)

Per branch, steady flow Q, area A(s) = πr(s)², profile order ζ (2 = Poiseuille):

```
dH/ds = −2(ζ+2) π μ Q / A²,     H = P + ρQ²/(2A²)      (viscous loss of total pressure; static P from H)
```

Inside a detected lesion (reference area A0, minimum area As, length Ls) the distributed model is replaced by
the Young–Tsai lumped loss (as parameterised in Blanco et al. 2018):

```
ΔP = Kv μ U0/D0 + Kt ρ/2 (A0/As − 1)² U0²,   U0 = Q/A0,   Kt = 1.52
Kv = 32 (0.83 Ls + 3.28 √As) / (2 √A0) · (0.75 A0/As + 0.25)²
```

Junctions: total-pressure continuity (`junction="total_pressure"`, default) or static continuity. Blood:
ρ = 1060 kg/m³, μ = 3.5 mPa·s, Newtonian, rigid wall.

**Topology** (`tree.py`): the parent of side branch j is the earlier branch whose centreline passes closest to
j's first sample; the effective start is where j leaves the parent lumen (samples before that inherit the
parent pressure). On the bundled LCA example this correctly attaches the OM-type branch to the LCX.

**Reference lumen and lesions** (`reference.py`): reference radius = non-increasing (isotonic) fit in which
interior narrowings (> 3% below the fit, widened to their shoulders) are replaced by interpolation between
healthy samples, so focal lesions do not pull the reference down while bifurcation steps and tapering are
kept. Lesion = run with diameter stenosis ≥ 30%, extended to the 10% shoulders. Synthetic check: a 50%
Gaussian lesion on a tapered tube → reference error 0.002 mm, detected DS 50 ± 4%.

## 3. Boundary conditions (`boundary.py`)

- **Total hyperemic flow** (`flow_model`):
  - `fixed_velocity` (default): v = 0.35 m/s × mean reference area of each major vessel (RCA; LM–LAD + LCX)
    — the QFR fixed-flow assumption (FAVOR Pilot fQFR).
  - `scaling_law`: Q_rest [ml/min] = 1.43 d[mm]^2.55 (van der Giessen et al. 2011) × `hyperemia_factor` (3).
  - `frame_count`: v = contrast path length / (frames / fps) × factor (TIMI frame count; the real-X-ray
    dataset has all frames per view, so this is available once the frame rate is known).
  - `explicit`: Q in ml/s.
- **Split**: at each junction Q_child/Q_continuation = (d_child/d_cont)^2.27 on reference diameters averaged
  over 10 mm (van der Giessen et al.; Murray's law is 3).
- **`bc_mode="resistance"`** (default): outlet microvascular resistances R_i = (P_i − Pv)/Q_i are calibrated
  on the *healthy reference tree* at Pa = 93 mmHg, Pv = 5 mmHg; the diseased tree is then solved with the
  same R_i (nonlinear system, `scipy.optimize.root` with a damped fixed-point fallback). A lesion reduces its
  own flow, as in vivo.
- **`bc_mode="fixed_flow"`**: imposes the target flows on the diseased tree (QFR-like; overestimates the drop
  of severe lesions — at 70% DS on the bundled LCA main branch: 0.17 fixed-flow vs 0.63 with flow adaptation).

Reported FFR uses the pressure after kinetic recovery to the reference lumen (what a wire reads distal to a
lesion; monotone along a branch). `ffr_static` keeps the local Bernoulli dip at a throat.

## 4. Is a 3D model applicable? Yes — as a reference path (`mesh3d.py`, `openfoam.py`)

The vessel code describes a closed tubular lumen, so a 3D domain can be built directly:

- **Surface**: zero level set of a signed distance field = union of spheres on a densely resampled centreline,
  truncated by flat caps at the ostium and every branch end; marching cubes (scikit-image). Checks on the
  bundled LCA: watertight (0 non-manifold edges), all 4 caps found, cap areas within ~10% of πr², volume
  3554 mm³ vs 3662 mm³ tube integral (branch overlaps counted twice in the latter).
- **Case**: OpenFOAM (ESI v2006+) `blockMesh` + `snappyHexMesh` (surface level 2, 3 boundary layers) +
  `simpleFoam`, laminar Newtonian. BCs consistent with the 1D model: inlet static pressure Pa, outlet
  volumetric flows from the converged 1D solution, no-slip wall. Centreline sampling and outlet patch-average
  pressure function objects are pre-configured.
- **Read-back**: `python -m vessel_code.ffr.openfoam results --case <out>/cfd3d` → FFR_3D along the same
  centreline samples + `ffr_1d_vs_3d.png`. `update-flows` performs one 1D–3D coupling step
  Q_i ← (P_i^3D − Pv)/R_i (relaxed) so the 3D run inherits the flow adaptation.

Why 3D is not the default: (i) the reconstructed lumen is circular by construction, so 3D adds little
geometric information beyond what the 1D model uses; (ii) FFR error from 2–3 view reconstruction is dominated
by the minimum lumen radius (losses ∝ r⁻⁴), not by the flow model; (iii) published 1D–3D differences
(0.00 ± 0.03) are below that uncertainty; (iv) 3D costs ~10–60 CPU-minutes per case vs ~0.1 s. Use the 3D path
to validate/calibrate the 1D model on a subset (e.g. fit the Kt/Kv multipliers, junction losses), or for
complex lesions (bifurcation lesions, ostial disease, tandem lesions closer than ~2 diameters).

Not verified here: the OpenFOAM case was generated and its read-back/coupling exercised on synthetic solver
output in the tests, but OpenFOAM itself was not run in this environment. First run on the cluster should
check `log.checkMesh` and that `sampled_fraction` in `ffr3d_results.json` is ~1.

## 5. Inference step (`infer.py`)

```bash
python -m vessel_code.ffr.infer --input <refined.npz | run dir> --output <dir> \
    [--config configs/ffr/default.json] [--set bc_mode=fixed_flow --set pa_mmhg=95] \
    [--mc 200] [--projections projections.npz --views 0,1] \
    [--virtual-lesion branch=0,t=0.3,sigma=0.03,severity=0.6] [--export-3d openfoam]
```

- Input: `refined.npz`/`coarse.npz` from `vessel_code.evaluate` (`vessel_code_mm`, `branch_exists`),
  prepared examples (`raw_vessel_code_mm`), or any NPZ with an `[M,N,4]` array (`--key` to name it; the
  real-X-ray prediction NPZ works the same way). A directory input runs every match of `--pattern` and writes
  `ffr_cases.csv`.
- `--mc N`: Monte Carlo over global radius scale (σ 5%), smooth local radius error (σ 8%, 5 mm correlation),
  flow (log-σ 0.25) and Pa (σ 5 mmHg); writes 5/50/95% bands and P(FFR ≤ 0.80) per branch.
- `--virtual-lesion`: inserts a Gaussian lesion with the same profile as the model's lesion head
  (`severity` = diameter stenosis at the throat) — for demos, sensitivity studies and synthetic validation.
- ~0.1 s per case for the 1D solve, ~4 s with 200 MC draws and figures; ~5 s more for the 3D export.

## 6. Visualisation (`visualize.py`)

Colour scale for FFR: diverging around the clinical threshold — red below 0.80, neutral grey at 0.80, blue
towards 1.0 (range 0.50–1.00), so the colour itself answers "ischaemic or not".

- `ffr_3d.png`: lumen surface coloured by FFR in two orthogonal views (AP + lateral), distal FFR at every
  branch end, lesion tags L1, L2… with a lesion table (DS, MLD, ΔFFR).
- `ffr_pullback.png`: one column per branch — FFR vs distance from the ostium (virtual pressure-wire pullback,
  0.75–0.80 grey zone, Monte Carlo 90% band, ΔFFR per lesion) above lumen vs reference diameter with lesions
  shaded. This is the view that maps to clinical QFR/vFFR reports.
- `ffr_projections.png`: FFR-coloured centrelines projected onto the input views (uses the model camera from
  `projections.npz` or a prepared example; for the real-X-ray acquisition camera, project with that branch's
  `set_model_camera` geometry instead).
- `ffr_interactive.html`: rotatable 3D surface (plotly from cdnjs) with hover read-out of FFR, distance,
  lumen/reference diameter and flow, plus a branch table; light/dark aware.
- `ffr_centreline.csv`, `ffr_result.npz`, `ffr_summary.json`: all values for downstream analysis.

## 7. Validation status and limitations

Verified (tests/test_ffr.py, 11 tests): Poiseuille drop in a straight tube (0.2%), inviscid Bernoulli in a
taper, Young–Tsai lesion drop reproduced by the network solver (2%), mass conservation and split at
junctions, outlet condition P−Pv = RQ after the resistance solve, FFR monotone in severity, flow adaptation
milder than fixed flow, reference/lesion detection, healthy bundled examples > 0.9, 3D surface watertight with
correct caps, OpenFOAM case files + read-back + coupling fixed point.

Not yet validated against measured FFR. Main limitations:
- The result is only as good as the reconstructed minimum lumen radius; at 0.46–0.54 mm/pixel (model
  isocentre) a 1-pixel radius error at a 1 mm MLD is a large FFR error — read the MC band, not just the line.
- Steady hyperemia, rigid walls, Newtonian blood, no collateral flow; side branches not reconstructed by the
  model are not perfused (their flow stays in the parent — FFR slightly pessimistic downstream).
- Flow is assumed (fixed velocity / scaling law) unless frame count is supplied; Pa defaults to 93 mmHg.

Suggested next steps: (1) run on reconstructions of cases with known FFR/QFR if any become available, else
compare against CT-FFR on ImageCAS-style data (reconstruct from rendered projections, compare FFR on the true
vs reconstructed artery — isolates reconstruction error); (2) contrast frame-count flow from the real-X-ray
`seg_frames` once the frame rate is known; (3) calibrate Kt/Kv with a handful of 3D runs.

## References

- Blanco PJ, Bulant CA, Müller LO, et al. Comparison of 1D and 3D models for the estimation of fractional flow
  reserve. Sci Rep 2018 (arXiv:1805.11472).
- Boileau E, Pant S, Roobottom C, et al. Estimating the accuracy of a reduced-order model for the calculation of
  fractional flow reserve. Int J Numer Method Biomed Eng 2018.
- Fossan FE, Sturdy J, Müller LO, et al. Uncertainty quantification and sensitivity analysis for computational
  FFR estimation in stable coronary artery disease. Cardiovasc Eng Technol 2018.
- Young DF, Tsai FY. Flow characteristics in models of arterial stenoses. J Biomech 1973.
- Huo Y, Svendsen M, Choy JS, Zhang ZD, Kassab GS. A validated predictive model of coronary fractional flow
  reserve. J R Soc Interface 2012.
- Tu S, Westra J, Yang J, et al. Diagnostic accuracy of fast computational approaches to derive FFR from
  diagnostic coronary angiography: the FAVOR Pilot study. JACC Cardiovasc Interv 2016.
- van der Giessen AG, Groen HC, Doriot PA, et al. The influence of boundary conditions on wall shear stress
  distribution in patients specific coronary trees. J Biomech 2011.
- Taylor CA, Fonte TA, Min JK. Computational fluid dynamics applied to cardiac computed tomography for
  noninvasive quantification of FFR. J Am Coll Cardiol 2013.
- Review of angiography-derived FFR methods: Front Cardiovasc Med 2023, doi:10.3389/fcvm.2023.1181803.
