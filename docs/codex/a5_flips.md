Exploratory: single flips occur only in faulted cases—6.75% of faulted decisions ([CSV](results/codex/a5_flips_analysis.csv)).  
Friction scatter raises single flips from 2.90% without it to 4.37% with all effects ([CSV](results/codex/a5_flips_analysis.csv)).  
A single-module clamp-plus-margin certificate gate cannot be validated from the available CSVs.

Singles lack recorded leg type, fault type/gain, margins, friction draws, and clamp observations. Payload-arm singles flip at 5.10% versus 4.37% without payload; different draws prevent causal attribution ([CSV](results/codex/a5_flips_analysis.csv)).

Nested cross-validation of logistic models therefore covers multi-module episodes only:

| Model | AUC | Abstention precision | Recall | Abstained |
|---|---:|---:|---:|---:|
| Clamp | 0.795 | 3.4% | 94.7% | 88.0% |
| Shape + clamp | 0.881 | 6.6% | 100% | 48.2% |

Source: [analysis CSV](results/codex/a5_flips_analysis.csv), high-recall rules. Misfit assumes known latents; these exploratory episode-level results do not establish certificate safety.

CPU execution and output checks passed.

Files wrote:

- [results/codex/a5_flips_analysis.csv](results/codex/a5_flips_analysis.csv)
- [scripts/codex/a5_flips_analysis.py](scripts/codex/a5_flips_analysis.py)