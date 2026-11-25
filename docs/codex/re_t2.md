Exploratory pilot re-analysis reproduces the A5m fixed-reference flip counts in [N3_flips.csv](/home/lunarz/cfcap/results/stress/N3_flips.csv); no count disagreements were found.  
For the A5y yaw target, v2, clamp and yaw misfit AUCs are 0.400, 0.389 and 0.438, respectively, agreeing with [A5y_auc.csv](/home/lunarz/cfcap/results/stress/A5y_auc.csv).  
The campaign and knee analysis remains incomplete: the available motor units lack matched T1 labels, and no knee units were found under the specified cache path.

The output includes episode-bootstrap confidence intervals for pilot flip and shift shares by shape, leg type and dominance class. A5m AUCs use fixed-reference flips against faulted non-flipped episodes; A5y AUCs also include the board’s yaw flip-or-shift target.

Files written:

- [re_t2.csv](/home/lunarz/cfcap/results/codex/re_t2.csv)
- [re_t2.py](/home/lunarz/cfcap/scripts/codex/re_t2.py)