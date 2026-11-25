# THEORY_cfcap: why the T0 oracle is almost never ambiguous, where it will be, and what a certificate pays

Role: theory (study 09 cfcap). Model: T0 quasi-static plant (`cfcap/t0/qs.py`, = capxfer THEORY.md Def. 1 plus
per-wheel motor spread and per-wheel commands). **Everything numerical is simulation and exploratory.**
Scripts: `scripts/theory_{coverage,nuisance,slack,merge}.py`. Outputs: `results/theory/*.csv`.

**Not re-claimed** (capxfer `docs/THEORY.md`, cited): L0 (QS = convex minimiser), L1 excitation lemma and Cor. 1
(danger set, "unexcited = invisible and harmless"), P2 masking 1/(n-1), P3 reveal-on-split / composition /
non-transfer, P4 blind spot and the 22.5° probe, T5 compositional + stop-aware certificate. Generic Fisher-rank / FDI
observability (Cristofaro & Johansen style) is also not claimed. Every result below is about **decisions on other
bodies** (children, merges), not about detecting a fault in the current body.

Notation. Wheel i, module k. Candidate (child or merge) B, decision leg l with commanded twist ξ_l, threshold τ.
Capability ρ_l(B) as capxfer Def. 2. Per-wheel motor: stall scale st_i, no-load scale nl_i, weak gain g_i,
battery ratio v_b = V/V0, nominal τ̄_s, ω̄_0 (θ-level). μ_k per module, normal loads N_i(B) from B's geometry.

---

## 1. Lemma 1 (two-constant reduction of a driven wheel)

**Statement.** In T0, a driven wheel (mode OK or WEAK, any heterogeneity) enters every QS equation of every
assembly it is part of only through the pair

  α_i = 1 / (g_i v_b st_i),  β_i = 1 / (g_i v_b nl_i)  ∈ ℝ²₊.

Its required (published) duty at an operating point (load torque τ, speed ω) is

  u = a α_i + b β_i,  a = τ/τ̄_s, b = ω/ω̄_0,

it tracks its reference iff |u| ≤ 1 there, and otherwise it runs on the motor line a α_i + b β_i = sgn u.

*Proof.* `qs.solve_flat`: tracking wheels satisfy ω = ω_ref and publish u_req = (τ_c/(τ̄_s st) + ω/(ω̄_0 nl))/(g v_b);
saturated or duty-commanded wheels satisfy τ̄_s st (U g v_b − ω/(ω̄_0 nl)) = τ_c, i.e. U = aα + bβ. Divide through;
g, st, nl appear nowhere else (the brake term belongs to passive wheels). ∎

**Consequences.**
- A weak fault (α, β) → (α, β)/g and motor spread (α, β) → (α/st, β/nl) are the *same kind of object*: a point in
  the per-wheel motor plane. For capability the fault/nuisance distinction is a prior, not a physical one. The
  decision-sufficient per-wheel latent of a driven wheel is (α_i, β_i) (plus μ_k and passive modes).
- Healthy tracking has load torque ≈ rolling resistance, so **a ≈ 0 on every v2 leg** of a healthy wheel in a healthy
  body (θ_plated crr = 5e-4: τ ≈ 8e-4 N·m against τ̄_s = 3.3 N·m). Measured torque share of the v2 duty:
  see §3 (`nuisance_wheel_ident.csv`). v2 therefore identifies β_i (the speed constant, which carries the weak gain)
  to the duty noise, and is blind to α_i (the torque constant) and to μ_k.

## 2. Proposition 2 (operating-point coverage: two identification regimes)

Let P be the parent's probe set. For wheel i let O_P(i) = {(a_p, b_p)} be its observed operating points (load torque,
speed, both normalised), and O_B,l(i) the operating point the candidate B needs from that wheel on leg l.

**(a) Nonparametric (monotone envelope) regime: identified iff dominated.** Suppose each wheel's torque envelope
E_i(ω) (max load torque at speed ω) is only known to be nonincreasing in |ω| and to lie below the healthy envelope.
Observing tracking at (a_p, b_p) reveals E_i(b_p) ≥ a_p and nothing else. The sharp identified set of E_i is
{E : E(b) ≥ a_p for b ≤ b_p, p ∈ P}; its lower end is the step E_i^low(b) = max{a_p : b_p ≥ b}. Hence
  ρ_l(B) ∈ [ρ_l(B; E^low), ρ_l(B; E^healthy)] (sharp), and
  the decision is identified for every member of the class iff for every wheel the needed point is dominated,
  O_B,l(i) ≤ (a_p, b_p) componentwise for some p (or the step envelope already gives ρ ≥ τ).
*Proof.* Monotone-class partial identification: each envelope between E^low and E^healthy is observationally
equivalent on P (it agrees at the probed points up to the observed inequality); ρ is monotone in each envelope
(more available torque never lowers QS capability in T0 — the WEAK-gain monotonicity (M) of capxfer T5, measured
δ_M(0.005) = 0.17 %), so the extremes are attained at the two ends. Dominated ⇒ every member gives the same
tracking ⇒ the same ρ. Undominated and consequential ⇒ the two ends straddle τ. ∎
*Two-point corollary (Le Cam; textbook, stated for completeness):* any rule that commits on an undominated decision
whose ends straddle τ has unsafe-go ≥ the prior mass of the low end there.

**(b) Closed form for X-mecanum polyominoes (speed coordinate).** Contact roller axes satisfy |ŵ·r̂| = 1/√2, so
  ω_i = √2 (v_i · r̂_i)/R, v_i = (v_x − ω_z p_{i,y}, v_y + ω_z p_{i,x}) (exact; max error vs T0 IK 0).
- Translations at speed v and heading φ: |ω_i| = √2 v |cos(φ − ψ_i)|/R with ψ_i ∈ {±45°, ±135°}, independent of
  position. Children keep their modules' yaws, so **surge/sway legs of every child and merge need exactly the
  parent's v2 speeds** (dominated in speed, ratio 1.000 in 100 % of child legs).
- Diagonals (d, d), d = 0.21: |ω_i| ∈ {0, 2d/R}; ratio to v2's v/R is 2d/v = 1.40 on half the wheels. **Undominated by
  v2 on 99.9 % of child diagonal legs; dominated by the clamp legs (0.45 m/s) on 100 %** (2d = 0.42 ≤ 0.45).
- Yaw: |ω_i| = √2 w_B |r̂_i · p_i^⊥(B)|/R with w_B = min(W, cap/L_c(B)), p measured from B's own centroid. A lone
  module is dominated (ratio 0.67). Pairs: undominated by v2 in 100 % (ratio 1.2), dominated after clamp. Children of
  3–7 modules: undominated in 99–100 % (median ratio 1.2–1.7) and still 96–98 % after clamp. **Yaw legs of children
  with ≥ 3 modules are the one class no translation probe covers**; only a yaw probe at the child's own (higher)
  yaw rate does. (`coverage_dominance.csv`, `coverage_dominance_yaw_by_size.csv`; 200 random 4–8-module parents,
  every connected child, 56,480 child legs.)
- Torque coordinate: v2 has a ≈ 0 (Lemma 1), so in regime (a) every child point that needs load torque (any wheel
  compensating a faulted neighbour, any saturated weak wheel) is undominated.

**(c) Parametric (two-constant) regime: identified iff the needed functional is in the probe span.** Under Lemma 1
the envelope is the line aα_i + bβ_i ≤ 1. With noise-free duties on legs p ∈ P the identified set for (α_i, β_i) is
{(α, β) : a_p α + b_p β = u_p ∀p} ∩ prior. If two observed points are linearly independent (e.g. a v2 point (≈0, b)
and a saturated point on the line from clamp) the wheel is point-identified, whatever the child needs. If all
points are (≈0, b_p) only β is identified, and the child's tracking condition a_c α + b_c β ≤ 1 is identified iff
  a_c α_max ≤ 1 − b_c β (sure track) or a_c α_min > 1 − b_c β (sure saturate),
where [α_min, α_max] is the prior range of α given β (the nuisance box on nl/st). Undominated speed is *not* a
problem in this regime: β alone extrapolates the speed ceiling to any speed.

**Numerical check (`theory_coverage.py`, `theory_nuisance.py`).**
- Closed form = T0 IK to 0.0 rad/s on every wheel of 200 parents and all their children; |ŵ·r̂| = 1/√2 on every wheel
  (`coverage_closedform.csv`).
- v2 torque coordinate: median torque share of the v2 duty 0.5 % on healthy wheels, 0.6 % on weak wheels (p90 of the
  per-wheel max over legs: 20 % / 66 %, from the yaw legs and from saturation). v2 saturates 57 % of weak wheels
  (g ~ U(0.3, 1)), v2+clamp 91 % (`nuisance_wheel_ident.csv`; 200 parents, 3,972 wheels).
- **Speed-only (nonparametric ceiling) class** (`coverage_speed_only.csv`, 200 parents of 4–6 modules, 1 weak fault in
  3/4 of them, every child + 4 merges, 39,680 (candidate, leg) pairs): with v2 the ambiguous share is 0.11 % / 0.71 % /
  **20.7 %** at τ = 0.6 / 0.7 / 0.8 (certified fraction of capable 79 % at τ = 0.8); with v2+clamp it is 0 / 0.005 % /
  0.17 % (certified 99.9 %). Clamp removes 99 % of the nonparametric ambiguity, exactly as (b) predicts (it dominates
  every diagonal operating point).
- **Parametric class (A2 oracle):** 0.02 % (A2). Attribution of A2's 24 ambiguous decisions (re-run of the unchanged
  A2 engine with a read-only scorer hook, `coverage_attrib.csv`, `coverage_ambiguous.csv`): **100 % are diagonal legs
  of a lone faulted module** (coast/brake confusion or weak g = 0.3/0.5), 83 % undominated in the faulted wheel's
  speed (8.38 vs 5.99 rad/s). Ambiguity rate on undominated faulted-child decisions 0.16 % vs 0.011 % on dominated
  ones (14×). With the nuisance prior (A2 `prior`): 1.46 % vs 0.67 %, 98 % lone modules. A3's finding that v2+clamp is
  the best single add-on (−67 %) is the dominance prediction.

**What A2 measured.** A2's oracle uses the true two-constant law (rho_model `known`), so it lives in regime (c);
that is why 0.02 % of decisions are ambiguous although the majority of diagonal and child-yaw decisions are
undominated in the sense of (a). The A2 identification is extrapolation along a known parametric family, not
coverage. The decisions that rely on it are exactly the undominated ones in (b); they are where a misspecified
motor law (current limit, thermal derating, nonlinear torque–speed curve, T2) will produce confident errors.
Numbers for the nonparametric speed-ceiling class are in §3.

## 3. Proposition 3 (nuisance-induced ambiguity band)

**Statement.** Let the parent's evidence identify β_i up to |log β_i − log β̂_i| ≤ ε_i (duty noise: ε_i = kσ_u/|u_i|)
and leave the unidentified coordinates x_i = log α_i and y_k = log μ_k free in a box of half-width w(δ) =
log((1+δ)/(1−δ)) ≈ 2δ (motor spread ±δ on st and nl ⇒ the ratio nl/st ranges over that box). Then, for every
candidate B and leg l, the identified interval of ρ_l(B) is

  [ρ − h_l, ρ + h_l] + O(δ²),  h_l(δ) = w(δ) (S^α_l + S^μ_l) + Σ_i ε_i |∂ρ_l/∂log β_i|,
  S^α_l = Σ_{i∈B} |∂ρ_l/∂log α_i|,  S^μ_l = Σ_{k∈B} |∂ρ_l/∂log μ_k| · (w_μ / w),

and the ambiguous share is A(δ) = P(|ρ_l(B) − τ| < h_l(δ)). If the margin m = ρ − τ has density f near 0 and S is
roughly independent of m there, A(δ) ≈ 2 f(0) E[h | m≈0] = 2 f(0) (2δ E[S^α + S^μ] + E[Σ ε|∂ρ/∂log β|]): linear in δ.

*Proof sketch.* Lemma 1 reduces the nuisance to (x, y, z). ρ_l is piecewise smooth in them (one QS active set per
piece, Lemma 0); on a piece, Taylor at the truth gives the box image ρ ± (‖∇_x ρ‖₁ w + ‖∇_y ρ‖₁ w_μ + Σ ε_i|∂_{z_i} ρ|)
+ O(w²) (the image of a box under an affine map is the interval of half-width Σ|grad|·halfwidth). Ambiguous iff τ
lies inside. ∎

**Key structural fact.** S^α_l = 0 whenever every wheel of B tracks on leg l (Lemma 1: α multiplies a ≈ 0), and
S^μ_l = 0 whenever no wheel of B slips. So the band is non-zero only on legs where B saturates or slips — the same
decisions where ρ < 1. Nuisance in the *identified* coordinate β costs only the duty noise. This is why A2's
`known` cell (σ inflation, discrete g grid) has 0.05 % ambiguity with heterogeneity on and exactly 0 with it off,
and why ignoring the parent's evidence on β (`prior` cell: st, nl, μ all free) raises it to 0.37 %.

**Numerical check (`theory_nuisance.py`, `nuisance_curve.csv`, `nuisance_sensitivity.csv`).** 200 parents of 4–6
modules, motor spread ±8 % and μ ±10 % in the truth, one weak wheel g ~ U(0.3, 1.0) (continuous) in 3/4 of parents,
every child + 4 merges with a 1–2-module assembly: 39,680 (candidate, leg) × 3 τ = 119,040 decisions. Variants:
`ident` (Prop. 3: α, μ free, β to duty noise k = 3), `none` (st, nl, μ all free: the A2 `prior` analogue).
- **Linearisation is accurate:** the exact box image at the gradient-sign corners and the linear prediction give the
  same ambiguous share to within 10 % at every δ (e.g. 0.047 % vs 0.047 % at δ = 0.08; 0.064 % vs 0.069 % at 0.12).
- **S^α = 0 on 78 % of (candidate, leg) pairs** (every wheel tracks), as the structural fact predicts. S^μ is small but
  rarely exactly 0 (tanh slip depends on μ at any force).
- **Ambiguous share vs δ (all decisions):**

| δ | ident, sensor | ident, floor | none (prior-like) | p99 band half-width (ident, sensor) |
|---|---|---|---|---|
| 0.02 | 0.013 % | 0.076 % | 0.030 % | 0.0029 |
| 0.04 | 0.032 % | 0.084 % | 0.045 % | 0.0054 |
| 0.08 | 0.047 % | 0.111 % | 0.091 % | 0.0104 |
| 0.12 | 0.064 % | 0.124 % | 0.150 % | 0.0154 |
| 0.16 | 0.089 % | 0.156 % | 0.207 % | 0.0206 |

  Fit: A_ident,sensor(δ) ≈ 0.0056 δ (slope 0.0053–0.0080 across δ); A_none(δ) ≈ 0.012 δ (2.2× ident: using the
  parent's evidence on β halves the nuisance ambiguity); the floor preset adds a δ-independent ≈ 0.07 % from duty
  noise on β. The p99 band half-width is ≈ 0.13 δ (ident) and 0.45 δ (none).
- Against A2: A2 `known` with heterogeneity on gives 0.047 %, A2 `prior` 0.37 % (1–2 faults incl. coast/brake,
  4–8 modules, 6 prior draws per hypothesis). Same order as `ident`/`none` at δ = 0.08 (0.047 % / 0.091 %); the A2
  `prior` excess (4×) comes from passive-fault hypotheses and larger parents, not from the motor box.
- Conclusion: in the parametric model, nuisance of realistic size (±8 %) cannot by itself produce the ≥ 5 % ambiguity
  that would make probing a headline; it would need δ ≈ 4–9 (meaningless). Ambiguity at the percent level needs
  either a wrong envelope class (Prop. 2(a): 7 % average, 21 % at τ = 0.8 with v2) or unidentified
  child-specific quantities (normal-load split, μ at higher slip).

## 4. Proposition 4 (certificate slack decomposition)

**Setting.** Any score-based certificate: estimator ρ̂_j, scale s_j, normalised residual r_j = (ρ̂_j − ρ_j)/s_j,
commit iff ρ̂_j − q s_j ≥ τ_j. Simultaneous version: q̂ = conformal (1−α) quantile of the per-episode max_j r_j over
the candidate family M(m) (`SimultaneousLower`), marginal version: of the pooled r_j.

**Statement.** A capable decision (m_j = ρ_j − τ_j ≥ 0) is lost iff m_j < s_j(q − r_j). Write
G(t) = P(0 ≤ m < t | m ≥ 0) for the capable-margin cdf and f̄ = sup_{0<t≤t_max} G(t)/t. With constant scale s,
  1 − CF_cert ≤ A_oracle + f̄ s [ E(q_marg − r)₊  +  (q_sim − q_marg)  +  (q̂ − q_sim)₊ ],
        estimator + marginal quantile ↑   simultaneity price ↑   finite-n slack ↑
where q_marg, q_sim are the population (1−α) quantiles of r and of max_j r_j, and A_oracle is the oracle's own loss
(§2–3). First-order equality: 1 − CF_cert ≈ E_r[G(s(q̂ − r))].
For exchangeable sub-Gaussian(σ_r) residuals with M_eff effectively independent entries per episode,
q_sim − q_marg ≤ σ_r(√(2 log(M_eff/α)) − z_{1−α}) (Bonferroni), and q̂ − q_sim = O_P(n^{-1/2}) with the
⌈(n+1)(1−α)⌉ order statistic.

*Proof.* Lost ⇔ ρ + s r − q s < τ ⇔ m < s(q − r). Split q − r = (q_marg − r) + (q_sim − q_marg) + (q̂ − q_sim),
use (u + v + w)₊ ≤ u₊ + v₊ + w₊, and G(t) ≤ f̄ t. Bonferroni: P(max r > t) ≤ M P(r > t). ∎

This is elementary; its value is that every term is *measurable* per method, which turns "58–75 % certified" into
an attribution.

**The cliff rule (what actually dominates on T0 margins).** T0 capable margins are bimodal: healthy candidates sit at
ρ ≈ 1, i.e. m = 1 − τ ∈ {0.2, 0.3, 0.4}, and faulted ones spread below. G(0.1) = 0.56 %, G(0.2) = 33 %
(`slack_margin_density.csv`). So the linear bound is informative only while s q̂ < 1 − τ_max = 0.2; beyond that the
certified fraction falls off a cliff: CF ≈ mean_τ P_r(s(q̂ − r) ≤ 1 − τ). Design rule: **a certificate keeps ≥ 90 %
of capable decisions at τ = 0.8 iff s·q̂ ≲ 0.2, i.e. σ_r ≲ 0.2/q̂.**

**Numerical check (`theory_slack.py`, `slack_decomposition.csv`).** T0 margins of the 100 A2 episodes (1,168
decisions per episode on average = the candidate family M(m) × 10 legs × 3 τ), synthetic residuals r = √c e_episode +
√(1−c) e_entry, Gaussian or Student-t₃, n_cal = 50 calibration episodes, α = 0.05, 40 splits.
- Population quantiles: q_marg = 1.65 (Gaussian); q_sim = 3.92 (c = 0), 3.57 (c = 0.5), 2.62 (c = 0.9).
  **The Bonferroni value z_{1−α/M} = 3.93 matches q_sim for independent residuals to 0.01.** Price of simultaneity
  q_sim/q_marg = 2.4× / 2.2× / 1.6×. Student-t₃: q_marg ≈ 1.36 but q_sim = 17 / 12 / 5.6 (12× / 8.5× / 4×).
  Finite-n slack q̂ − q_sim at n = 50: +0.1–0.25 (Gaussian), +1.4–4.7 (t₃).
- Certified fraction of capable (oracle 99.98 %):

| residual | σ | marginal | simultaneous (pop. q) | simultaneous (q̂) | lone faulted: marginal / simultaneous | unsafe per episode: marginal / simultaneous |
|---|---|---|---|---|---|---|
| Gauss, c=0 | 0.02 | 99.9 % | 99.7 % | 99.7 % | 97.8 / 89.4 % | 5.6 / 0.0 % |
| Gauss, c=0 | 0.05 | 99.2 % | 81.3 % | 79.3 % | 87.6 / 55.2 % | 10.4 / 0.0 % |
| Gauss, c=0.5 | 0.05 | 99.2 % | 86.6 % | 84.2 % | 88.0 / 61.0 % | 7.4 / 0.0 % |
| Gauss, c=0.9 | 0.05 | 99.1 % | 95.8 % | 93.6 % | 88.2 / 73.9 % | 6.4 / 0.4 % |
| Gauss, c=0 | 0.10 | 83.3 % | 22.9 % | 21.1 % | 65.2 / 12.4 % | 13.9 / 0.0 % |
| t₃, c=0.5 | 0.02 | 99.9 % | 65.9 % | 54.2 % | 97.8 / 34.3 % | 5.0 / 0.0 % |
| t₃, c=0.9 | 0.05 | 99.2 % | 52.7 % | 37.9 % | 91.5 / 23.7 % | 6.2 / 0.05 % |

- Attribution of the σ = 0.05, c = 0 Gaussian case: oracle → marginal loses 0.8 pt (estimator + marginal quantile),
  marginal → simultaneous (population) 17.9 pt (simultaneity price), → q̂ 2.0 pt (finite n). The bound of the
  statement is tight at σ ≤ 0.02 (predicted loss ≤ 0.19 % / 0.53 % vs realised 0.08 % / 0.34 %) and vacuous beyond
  the cliff.
- **capxfer's 58–75 % on lone faulted modules corresponds to σ_r ≈ 0.10 marginal, or ≈ 0.05 with the simultaneous
  quantile.** The marginal certificate's per-decision unsafe rate is tiny (≤ 0.02 %) only because the unsafe base rate
  is tiny; per episode it commits at least one unsafe candidate 3–14 % of the time.

## 5. Theorem 5′ (compositional certificate for merges of never-assembled modules)

**Assumptions.** Assemblies G_1, …, G_J probed separately (possibly never docked together). For each G_j, capxfer
T5's stage 1 at level α₁^{(j)} with budget K_j (assumption (U) per wheel class, (K) per assembly). Stage 2: a band
b(h) for any assembly B built from wheels of ∪_j G_j, computed from the physics F_B, where F_B depends on the
wheels' latents and on B's geometry only (normal loads, origin, frame — Prop. 3(a) of capxfer, which holds for any
QS assembly, so also for a merge). Let δ_comp = P(z_B < F_B(h_B, θ) − ε_c) be the composition defect of the true
plant against the QS composition (dock compliance, load transfer; 0 in T0; measured by A4 in T1 and T2).

**Statement.** Let S_j be G_j's projected set and ℓ(B) = min_{h ∈ ×_j S_j|B_j} b(h) − ε − ε_c. Then
(i) for every merge or child B, P(z_B < ℓ(B)) ≤ Σ_j α₁^{(j)} + α₂ + δ_M(ε) + δ_comp;
(ii) the stage-1 term is **family-free**: the same event {every wheel of ∪_j G_j covered} makes ℓ valid for *all*
candidates B ∈ M simultaneously, so the simultaneous bound over M is Σ_j α₁^{(j)} + P(∃B∈M: band or composition
fails). With the deterministic in-model band (θ known, α₂ = 0) and δ_comp = 0, simultaneity over M costs nothing;
(iii) if the probe noises of the G_j are independent, Σ_j α₁^{(j)} can be replaced by 1 − Π_j(1 − α₁^{(j)}).

*Proof.* (i) As capxfer T5(a) with the union over every wheel of every G_j: if all are covered, the true latent of
each G_j lies in S_j with its own budget, hence the true h_B ∈ ×_j S_j|B_j (the budgets are per assembly, so a
miscovered wheel of G_1 cannot exhaust G_2's budget — the corrected T5 argument applies per assembly). Band and
composition events as in T5 plus δ_comp. (ii) The coverage event does not depend on B. (iii) Independence of the
per-assembly coverage events. ∎

**What is new against capxfer T5.** T5 certifies sub-assemblies of *one* probed parent. 5′ adds (1) cross-parent
merges with per-assembly budgets (needed: a joint budget would couple G_1's miscoverage to G_2's set), (2) the
explicit composition defect δ_comp as the only merge-specific term, and (3) the family-free stage 1, which is the
formal reason the compositional route avoids the simultaneity price of Prop. 4.

**Numerical check (`theory_merge.py`, `merge_coverage.csv`).** 600 pairs (G_1, G_2) of 1–2-module assemblies with
random yaws, ≤ 1 fault each (weak 0.4/0.6/0.8 or coast, 70 %), motor spread and μ heterogeneity in the truth; stage 1
= split-conformal sets on a noisy 2-D class feature (noise 0.18), α₁ = 0.1 per assembly, levels α₁/(2n_G) and
α₁/(2K); stage 2 exact T0 with the true continuous plant; every merge layout (3,590 merges, 6 per pair) certified from
the same sets.
- Stage-1 miscoverage per assembly 9.8 % / 7.3 % (≤ 0.10 each, nearly tight); either 16.5 % ≤ 1 − 0.9² = 19 % ≤ 20 %.
- Realised miscoverage of the lower bound: 0.38 % per (merge, leg), 1.1 % per merge, **1.7 % simultaneously over all
  merges and legs of a pair** ≤ 19 %. Violations given full coverage: 0 (composition is exact in T0, δ_comp = 0).
- Certified fraction of capable (τ = 0.6) 88.5 %. The bound is loose because most miscoverages are inconsequential
  (Cor. 1 of capxfer: a missed class that does not change ρ cannot cause a violation).
- Not tested here: δ_comp in T1/T2 (A4), and stage 1 with the real self-model.

## 6. Which results are paper-grade

**Strongest (ICML-style):**
1. **Lemma 1 + Prop. 2 (identification regimes for counterfactual decisions).** The decision-sufficient latent of a
   driven wheel is a point in a 2-D motor plane; the parent identifies the speed constant but not the torque
   constant or μ; and the gap between "identified by dominance" (nonparametric) and "identified by extrapolation"
   (parametric) is measured: 7 % (21 % at τ = 0.8) vs 0.02 % ambiguous after v2, collapsing to 0.06 % with one clamp
   probe that dominates the diagonal operating points; the closed form tells which legs (diagonals, yaw of ≥ 3-module
   children) are undominated by any translation probe. It turns "probing is secondary" into a sharper claim: probing
   is secondary *given the envelope class*; it is what makes a certificate robust to envelope misspecification, and the
   closed form gives the minimal probe (clamp ≥ 2d for diagonals; child-rate yaw for yaw). This is partial
   identification with a decision target, not FDI rank.
2. **Thm 5′ + Prop. 4 (family-free compositional simultaneity vs the simultaneity price).** A direct certificate over
   the candidate family pays q_sim/q_marg = 2.4× (Gaussian) to 12× (t₃) in margin, which on T0 margins costs up to
   18 pt of certified fraction at σ = 0.05 (cliff rule s q̂ < 1 − τ); the compositional route certifies all children
   and merges of never-assembled modules from one per-wheel coverage event at no extra α. This gives the paper its
   "why compose" theorem with a measurable price on the other side.

**Textbook / supporting:** Prop. 4's decomposition algebra and Bonferroni bound (standard conformal facts; its value is
the measured attribution), the Le Cam two-point corollary, Prop. 3's box-image linearisation (the structural fact
S^α = 0 on tracking legs is model-specific and worth one sentence). Anything about detecting the fault in the current
body (excitation, masking, blind spots) is capxfer THEORY.md and is cited, not claimed.

**Caveats.** All checks are T0 (QS); the two-constant law is T0's and T1's motor model, so the "parametric regime"
results are in-model by construction; the speed-only class is a stand-in for envelope misspecification, not a
measured misspecification. Prop. 4's synthetic residuals are not a trained model's.
