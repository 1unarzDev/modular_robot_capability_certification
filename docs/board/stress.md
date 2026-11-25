# stress (T2 independent plant, A5 misspecification)

Status: T2 built and tested; A5 targets written (below, before the run).

## A5 targets (written 2026-11-25 before the run; exploratory)

Design. For each episode, the latents are drawn once: theta from `params.theta_dr(theta_plated)`, `plant.HET_FULL`
heterogeneity, and 0, 1 or 2 faulted wheels (k uniform on {0,1,2}, mode uniform on weak/coast/brake, weak
g ~ U(0.3, 0.9)). The same latents then go to T1 (`cfcap.t1.capability`) and to T2 (`cfcap.t2.capability`), and
each plant gives rho_rel on the 10 decision legs, each normalised by its own healthy twin. Children are pair, L3,
square and 2x3 (single is reported separately as a control). Arm P0 has no payload. Arm P1 adds a 0-10 kg payload
at a random deck point, to stress load transfer. Seeds 905000+ (A5 pilot namespace, not the T2 evaluation seeds).

The T2 conditions are the ablation: `base` (crane kernel + robot firmware, no added effect), `compliance` (dock
compliance + 1.1 mm hook play), `load_transfer`, `aniso` (roller-drag anisotropy), `aniso_mu` (I4 per-wheel roller
friction scatter), `all-aniso_mu`, and `all`. All conditions use the same effect draws.

A decision is (episode, leg, tau), tau in {0.7, 0.8}. Its truth is rho_rel >= tau.
- **Flip share** F = share of decisions whose truth differs between T1 and T2 `all`, pooled over the four children
  and both taus. Reported for all decisions and for faulted children only, with an episode-cluster bootstrap 95 % CI.
- **Shift** d = rho_rel(T2) - rho_rel(T1): quantiles (1, 5, 50, 95, 99 %), and the share with |d| > 0.05 (a fixed
  stand-in for the certificate margin), per condition and child.

Decision thresholds (PLAN §5, A5):
- **F > 5 % (point estimate, `all`, P0) -> misspecification robustness becomes the lead ML claim.** Secondary
  signal: share(|d| > 0.05) > 5 %.
- F <= 1 % and share(|d| > 0.05) <= 1 % -> T2 stays an evaluation-only check and is not a headline.
- Attribution. The effect whose single-effect condition has the largest F (or |d| > 0.05 share) causes the shift.
  `base` alone measures the crane-kernel change (numerics, Coulomb roller drag, slip ratio, massless wheels).

Pilot budget: each run <= 15 min through `scripts/heavy.sh`, numba with 4 threads.
