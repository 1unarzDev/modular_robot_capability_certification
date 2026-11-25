"""Emit the generator's campaign plans (scripts/campaigns/{scaling16,hetero,confirm}.json, lineage cfcap-gen) from one
rate table, so unit_s follows the latest pilot. Rates are 3090 steady episodes/s (schema v3, rho_ref rows included).

  python scripts/gen_campaigns.py            # writes the three plans; then: python scripts/make_jobs.py emit <plan>
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
# 3090 steady episodes/s by split. Basis: local 4070 pilot (results/generator/pilot.csv, pilot2.csv) x 14/17.9 (3090 vs
# 4070, coordinator's 3090 rate for train-like episodes), /1.3 for nominal rho_ref rows where not measured.
RATE = dict(train=11.0, het=11.0, H4=4.0, H4b=1.8)
PILOT = ("local 4070 Ti S: scripts/gen_pilot.py (train 17.9 eps/s steady, results/generator/pilot.csv) and "
         "scripts/gen_pilot2.py (het/H1/H2/H4b, results/generator/pilot2.csv), scaled to a 3090 by 14/17.9")
OVERHEAD = 180


def group(name, split, G, n, rate, prio, purpose, plate="plated", jobs=None):
    jobs = jobs or G
    ups = G // jobs
    return {"name": name, "task": "gen_shard", "stage": 1, "lane": "rented", "profile": "torch", "shards": jobs,
            "units_per_shard": ups, "unit_s": round(n / rate), "overhead_s": OVERHEAD, "vram_gb": 8,
            "priority": prio, "lineage": "cfcap-gen",
            "task_args": ["--split", split, "--plate", plate, "--gen-shards", str(G), "--episodes", str(n),
                          "--probes", "v2", "--engine", "compile", "--block", "250"],
            "pilot": PILOT, "role": "generator", "purpose": purpose}


def plan(name, stop, doc, groups):
    return {"campaign": name, "lineage": "cfcap-gen", "submitter": "09-cfcap", "owner": "09-cfcap", "project": "crane",
            "stop": stop, "_doc": doc, "groups": groups}


def main():
    out = {}
    out["scaling16"] = plan(
        "cfcap-gen-scaling16", "Stop when the H4b (11-16 modules) exploratory shards are complete (sha256 sidecars).",
        "Plan 2b. Exploratory H4b test split: parents of 11-16 modules (random-growth shapes, family FAM_H4), children "
        "and merges up to 16 modules, plated. 3,000 episodes = 6 shards x 500 (bins 11-13 / 14-16 ~1,650 / 1,350). "
        "Confirmatory H4b episodes are in confirm.json (conf_H4b), not here.",
        [group("gen-H4b", "H4b", 6, 500, RATE["H4b"], 40,
               "Exploratory 11-16-module test episodes for the size-generalisation curve (models eval2-ext, theory).")])
    out["hetero"] = plan(
        "cfcap-gen-hetero", "Stop when het_cal, het_test, H1 and H2 (plated) shards are complete.",
        "Plan 2c. Module types (cfcap.t1.plant wtype: mecanum, conventional/skid = roller-locked channels, omni-like = "
        "roller-axis mu x0.5). het_* = one type per assembly (mecanum or conventional); H1 = all-omni parent (unseen "
        "type); H2 = mecanum+conventional mixed in one body (unseen combination). Training is on the fly from "
        "het_train; persisted here: het_cal 10k (conformal for the type-aware models), het_test, H1, H2 5k each.",
        [group("gen-het-cal", "het_cal", 1, 10000, RATE["het"], 41, "Calibration split for models trained on het_train."),
         group("gen-het-test", "het_test", 1, 5000, RATE["het"], 42, "In-distribution test for module types."),
         group("gen-H1", "H1", 1, 5000, RATE["het"], 43, "Unseen module type (omni-like parent)."),
         group("gen-H2", "H2", 1, 5000, RATE["het"], 44, "Unseen type combination (mecanum+conventional in one body).")])
    out["confirm"] = plan(
        "cfcap-gen-confirm", "Stop when every conf_* shard below is complete; never score before 01:45Z "
        "(docs/CONFIRMATORY.md).",
        "Plan 2d, docs/CONFIRMATORY.md (8ffc33a). Fresh seed root SEED_CONF=92599, split code +64. Plated. Scored "
        "ranges: conf_test 0-4,999 (C-MAIN 0-2,999, C-GATE 3,000-4,999); conf_H4 0-1,499 (600 per bin 7-8 / 9-10 at "
        "the H4 size weights); conf_H4b 0-1,399 (>=600 per bin 11-13 / 14-16); conf_het_test, conf_H1, conf_H2 0-999. "
        "Each split has ONE SPARE last shard beyond the scored range (conf_test shard 2, conf_H4/H4b shard 2, het "
        "shard 1): if ops wants a canary, run the spare first; it is never scored. Labels must not be read before "
        "the scoring window.",
        [group("conf-test", "conf_test", 3, 2500, RATE["train"], 50, "C-MAIN + C-GATE episodes (5,000 scored)."),
         group("conf-H4", "conf_H4", 3, 750, RATE["H4"], 51, "C-SCALE bins 7-8, 9-10 (1,500 scored)."),
         group("conf-H4b", "conf_H4b", 3, 700, RATE["H4b"], 52, "C-SCALE bins 11-13, 14-16 (1,400 scored)."),
         group("conf-het-test", "conf_het_test", 2, 1000, RATE["het"], 53, "C-HET in-distribution (1,000 scored).",
               jobs=1),
         group("conf-H1", "conf_H1", 2, 1000, RATE["het"], 54, "C-HET unseen omni type (1,000 scored).", jobs=1),
         group("conf-H2", "conf_H2", 2, 1000, RATE["het"], 55, "C-HET mixed types (1,000 scored).", jobs=1)])
    for k, v in out.items():
        (ROOT / "scripts/campaigns" / f"{k}.json").write_text(json.dumps(v, indent=1))
        print("wrote", k)


if __name__ == "__main__":
    main()
