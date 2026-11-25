TASK n3_fixedref: an independent re-analysis that is a second implementation; the stress role does its own.
Confound: in the T2 misspecification tests each plant's decision label ρ_dec divides by THAT plant's own nominal
healthy reference, so under a misspecified motor law the label absorbs the misspecification. The robot only has T1's
reference table.

Re-score with ρ_dec^fixed = ρ_abs^T2 / ρ_ref^T1(shape, leg).
- Inputs: results/stress/A5m_rows.csv (motor-law arm), the A5 and A4 merge row files in results/stress/, T1's
  reference table (cfcap/t1/reference.py rho_ref_many, or cfcap/gen/rho_ref.py; use the T1 one), and T2's
  reference (cfcap/t2, rho_ref) only to recover ρ_abs if the rows store only ρ_dec. If ρ_abs and enough identifiers
  are missing, say exactly what's missing and stop.
- Report decision flips (T1 truth vs T2 truth at τ ∈ {0.7, 0.8}) and |d| > 0.05 shares by shape × leg type ×
  dominance class (dominated = surge/sway; undominated = diagonals and yaw of ≥3-module children), with
  episode-bootstrap 95 % CIs, under both labels side by side.
- Pre-stated bar (H12, all+motor): undominated flips > 1 % (upper CI) AND dominated flips ≤ 0.5 %.
- Write results/codex/n3_fixedref.csv and scripts/codex/n3_fixedref.py. CPU only, via scripts/heavy.sh if > 2 min.
