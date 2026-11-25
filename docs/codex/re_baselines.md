Exploratory re-analysis is **incomplete** because the required calibration and comparison data are absent.  
Available pilot-run raw-bound metrics are in [re_baselines.csv](/home/lunarz/cfcap/results/codex/re_baselines.csv).  
No numerical disagreement with the role can be established: its [board report](/home/lunarz/cfcap/docs/board/baselines.md) does not report scored E3 results or an SMC accuracy curve.

The matched operating point cannot be fitted without calibration rows. The E3 bars cannot be evaluated without the learned method and model-free comparator, and the particle selection cannot be made from the available single-setting canary. The CSV marks these results unavailable; its pilot metrics use raw run bounds and are not calibrated simultaneous certificates.

The decision-relevant stratum uses faulted candidates **or** truth near the decision threshold, as specified in [baselines.md](/home/lunarz/cfcap/docs/board/baselines.md). For example, the exploratory raw copy-parent result for pair-shaped episodes at τ = 0.8 is a certified fraction of 38.1% and an unsafe-commit rate of 13.0%, with 86 episodes ([re_baselines.csv](/home/lunarz/cfcap/results/codex/re_baselines.csv)).

Files written:

- [re_baselines.py](/home/lunarz/cfcap/scripts/codex/re_baselines.py)
- [re_baselines.csv](/home/lunarz/cfcap/results/codex/re_baselines.csv)