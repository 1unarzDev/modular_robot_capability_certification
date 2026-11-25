"""Run every paper analysis script and write paper/numbers.tex, paper/tables/*.tex, paper/figs/*.pdf.

    source env.sh; $PY scripts/analysis/make_all.py                  # real results/ -> paper/
    $PY scripts/analysis/mock.py /tmp/cfmock && \
    $PY scripts/analysis/make_all.py --results /tmp/cfmock --paper /tmp/cfmock-paper   # dry run on mock data

Missing inputs become red placeholders; schema violations raise. Refuses to write mock numbers into the repo paper/.
"""
from __future__ import annotations

import sys

import common as C
import f1_ambiguity
import f2_diversity
import f3_throughput
import f4_gate
import merge_table
import stageA_numbers
import t1_certificate
import t2_heldout
import t3_grid
import t4_probing

SCRIPTS = (stageA_numbers, t1_certificate, t2_heldout, t3_grid, t4_probing, merge_table, f1_ambiguity, f2_diversity,
           f3_throughput, f4_gate)


def main(argv=None):
    res, paper = C.env_paths(argv)
    if (res / "MOCK.txt").exists() and paper.resolve() == C.PAPER.resolve():
        sys.exit("refusing: mock results would overwrite paper/numbers.tex; pass --paper elsewhere")
    paper.mkdir(parents=True, exist_ok=True)
    nums = [s.run(res, paper) for s in SCRIPTS]
    seen = C.write_numbers(nums, paper / "numbers.tex")
    n_ph = sum(1 for N in nums for v in N.items.values() if v[2] == "placeholder")
    print(f"{len(seen)} macros ({n_ph} placeholders) -> {paper / 'numbers.tex'}")
    return nums


if __name__ == "__main__":
    main()
