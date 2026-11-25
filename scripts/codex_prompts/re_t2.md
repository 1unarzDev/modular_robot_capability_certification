TASK re_t2: second-implementation re-analysis of the stress results (exploratory). Inputs: docs/board/stress.md and
results/stress/* (A5m_rows.csv, the fixed-reference relabel, t2-motor / t2-knee outputs pulled under
~/.cache/vast-out/*/crane*/cfcap-t2*). Never open `confirmatory` paths or conf_* data.
Recompute with the FIXED T1 reference (ρ_dec = ρ_abs^T2 / ρ_ref^T1 via cfcap.gen.rho_ref):
- decision flips and |d|>0.05 by shape × leg type × dominance class × condition (motor, all+motor, knee severity
  1–3), with episode-bootstrap CIs;
- the clamp / v2 / yaw misfit AUCs on fixed-reference positives.
Compare with the role's numbers and flag any disagreement. Write results/codex/re_t2.csv and scripts/codex/re_t2.py.
