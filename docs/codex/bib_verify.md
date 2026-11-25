Exploratory audit completed for every cited bibliography entry and citing sentence.  
Corrected copy written; original bibliography and sections unchanged.  
Claim mismatches remain in the prose and are documented in bibliography comments.

Statuses concern substantive metadata and claims; adding source URLs alone does not change `ok` to `fixed`. Uncited entries were retained without re-auditing.

| key | status | source URL | fix |
|---|---|---|---|
| `cristofaro2014fault` | fixed | [Publisher](https://www.sciencedirect.com/science/article/pii/S0005109814001824) | Added DOI; metadata and observer/allocation claim confirmed. |
| `bickfordsmith2023epig` | fixed | [PMLR](https://proceedings.mlr.press/v206/bickfordsmith23a.html) | Added proceedings details and arXiv identifier; prediction-oriented claim correct. |
| `foster2021dad` | fixed | [PMLR](https://proceedings.mlr.press/v139/foster21a.html) | Added proceedings details and arXiv identifier; sequential amortization claim correct. |
| `huang2024tndp` | fixed | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c59f05d7ab3638b138cc61f32e1a7cd1-Abstract-Conference.html) | Added volume, pages, DOI and arXiv identifier; decision-aware design claim correct. |
| `wagenmaker2021task` | fixed | [PMLR](https://proceedings.mlr.press/v139/wagenmaker21a.html) | Added author middle initial, proceedings details and arXiv identifier; claim correct. |
| `xiong2023modumorph` | fixed | [PMLR](https://proceedings.mlr.press/v202/xiong23a.html) | Replaced “others” with Jacob Beck and Shimon Whiteson; added proceedings details and arXiv identifier. |
| `hu2024reconfigurable` | ok | [arXiv](https://arxiv.org/abs/2403.10496) | Configuration inference from motion/proprioception described correctly. |
| `gupta2022metamorph` | ok | [arXiv](https://arxiv.org/abs/2203.11931) | Morphology-conditioned control transfer described correctly. |
| `huang2020onepolicy` | ok | [PMLR](https://proceedings.mlr.press/v119/huang20d.html) | Shared modular control claim correct. |
| `pathak2019selfassembling` | ok | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2019/hash/c26820b8a4c1b3c2aa868d6d57e14a79-Abstract.html) | Self-assembly and control generalization claim correct. |
| `bongard2006resilient` | ok | [Publisher](https://doi.org/10.1126/science.1133687) | Self-modeling and damage recovery claim correct. |
| `kwiatkowski2019taskagnostic` | ok | [Publisher](https://doi.org/10.1126/scirobotics.aau9354) | Self-modeling and damage detection claim correct. |
| `cully2015robots` | wrong-claim | [Publisher](https://doi.org/10.1038/nature14422) | Separate adaptation from diagnosis: this method adapts without requiring self-diagnosis. |
| `blanke2016diagnosis` | wrong-claim | [Springer](https://link.springer.com/book/10.1007/978-3-662-47943-8) | “One plant/fixed actuators” overstates the restriction; book covers distributed diagnosis and reconfigurability. Normalized edition field. |
| `campbell2004auxiliary` | ok | [Princeton](https://press.princeton.edu/books/ebook/9781400880041/auxiliary-signal-design-for-failure-detection-pdf) | “Decision-agnostic” is defensible specifically for downstream task utility; clarify that meaning. |
| `lekeufack2024conformal` | wrong-claim | [Project/paper](https://conformal-decision.github.io/) | Replace “single decisions”: CDT controls sequential decision risk. |
| `angelopoulos2024conformal` | wrong-claim | [Paper](https://arxiv.org/html/2208.02814v3) | Includes multiple/adversarial risks. Method’s formula matches, but requires exchangeability and bounded monotone loss; candidate reselection needs a monotonicity argument. |
| `schubert2024conformal` | ok | [arXiv](https://arxiv.org/abs/2404.16500) | Vehicle performance prediction under actuator degradation described correctly. |
| `chenni2026domainaware` | ok | [arXiv](https://arxiv.org/abs/2611.26065) | Silent coverage failure under synthetic-to-real shift described correctly. |
| `yang2023object` | ok | [arXiv](https://arxiv.org/abs/2303.12246) | Geometric uncertainty propagation supports the analogy; does not itself establish robot-module composition. |
| `cranmer2020frontier` | ok | [Publisher](https://doi.org/10.1073/pnas.1912789117) | SBI overview attribution appropriate. |
| `ramos2019bayessim` | ok | [RSS](https://roboticsproceedings.org/rss15/p29.html) | Simulator-parameter posterior inference attribution correct. |
| `wehenkel2025addressing` | fixed | [PMLR](https://proceedings.mlr.press/v267/wehenkel25a.html) | Matched publisher’s “Joern-Henrik” spelling; misspecification claim correct. |
| `capxfer2026` | not-found | No public URL located | Retained anonymous submission without invented identifiers; local manuscript exposes claim mismatches below. |

The capxfer hardware-certification claim contradicts its [abstract](/home/lunarz/worktrees/capxfer-cluster/capxfer/paper/sections/abstract.tex:2), which states that evidence is simulated. Its [theory](/home/lunarz/worktrees/capxfer-cluster/capxfer/docs/THEORY.md:24) also does not support the fault-free-twin normalization or convex traction-allocation attribution used in the citing sections.

Validation passed: bibliography keys/order preserved, braces balanced, cited keys present, and public sources recorded. No corpus investigation or GPU work.

Files written:

- [paper/references_verified.bib](/home/lunarz/cfcap/paper/references_verified.bib)