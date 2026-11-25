"""F3, throughput and compute: (a) T1 simulator throughput vs assembly size (Stage A0), (b) wall time per decision
per method vs wheels (amortized vs SMC vs enumeration), the latency half of the ML claim (T3).

Inputs:
  results/stageA/A0_throughput.csv        (existing; 4070 Ti SUPER; kind legs = 3-s decision legs)
  results/stageA/A6_summary.csv           (existing; exact enumeration s/episode, 2x3)
  contract 'lat' (results/*/paper/latency*.csv*), else results/models/latency.csv (pilot ms_per_episode)
Outputs: paper/figs/f3_throughput.pdf, macros cfFThree*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as C
import figstyle as S

WHEELS = {"single": 4, "pair": 8, "L3": 12, "square": 16, "2x3": 24}


def run(res, paper):
    N = C.Numbers("f3_throughput.py")
    out = paper / "figs/f3_throughput.pdf"
    a0f, a6f = res / "stageA/A0_throughput.csv", res / "stageA/A6_summary.csv"
    L, lsrc = C.load("lat", res)
    if L.empty and (res / "models/latency.csv").exists():
        m = pd.read_csv(res / "models/latency.csv")
        L = pd.DataFrame(dict(method=m.method, shape=m["shape"], n_wheels=m["shape"].map(WHEELS),
                              sec_per_decision=m.ms_per_episode / 1000, hardware="RTX 4070 Ti SUPER",
                              status="exploratory"))
        lsrc = [C.rel(res / "models/latency.csv")]
    if not a0f.exists() and L.empty:
        S.placeholder(out, "F3: awaiting A0_throughput.csv / latency.csv")
        for k in ("f3.legs.persec.sixmod", "f3.speedup"):
            N.ph(k)
        return N
    fig, axes = S.plt.subplots(1, 2, figsize=(S.W2, 2.2))
    if a0f.exists():
        a0 = pd.read_csv(a0f)
        a0 = a0[a0.kind == "legs"]
        best = a0.groupby(["engine", "modules"]).trials_per_s.max().reset_index()
        for i, (eng, d) in enumerate(best.groupby("engine")):
            axes[0].plot(d.modules, d.trials_per_s, "o-", color=S.PAL[i], ms=3, label=f"T1, {eng}")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("modules per assembly")
        axes[0].set_ylabel("3-s decision legs per second")
        axes[0].legend(frameon=False)
        six = best[best.modules == 6].trials_per_s.max()
        N.num("f3.legs.persec.sixmod", six, [C.rel(a0f)], "exploratory")
        N.set("f3.gpu", str(pd.read_csv(a0f).gpu.iloc[0]).replace("NVIDIA GeForce ", ""), [C.rel(a0f)])
    else:
        N.ph("f3.legs.persec.sixmod")
    srcs = list(lsrc)
    if a6f.exists():
        a6 = pd.read_csv(a6f)
        L = pd.concat([L, pd.DataFrame(dict(method=["enum_known"], shape=["2x3"], n_wheels=[24],
                                            sec_per_decision=[a6.sec_per_episode_median.iloc[0]],
                                            hardware=["RTX 4070 Ti SUPER"], status=["exploratory"]))])
        srcs.append(C.rel(a6f))
    if not L.empty:
        for i, (m, d) in enumerate(L.groupby("method")):
            d = d.groupby("n_wheels").sec_per_decision.median().reset_index()
            axes[1].plot(d.n_wheels, d.sec_per_decision, "o-", color=S.PAL[i % len(S.PAL)], ms=3,
                         label=m.replace("_", " "))
        axes[1].set_yscale("log")
        axes[1].set_xlabel("wheels in the parent")
        axes[1].set_ylabel("wall time per decision (s)")
        axes[1].legend(frameon=False, fontsize=6)
        am = L[L.method.str.startswith("amortized")].sec_per_decision.median()
        sm = L[L.method.isin(["nuisance_smc", "enum_known"])].sec_per_decision.median()
        N.num("f3.speedup", sm / am if am and np.isfinite(am) and np.isfinite(sm) else np.nan, srcs,
              C.status_of(L), fmt="{:,.0f}")
        N.num("f3.sec.amortized", am, srcs, C.status_of(L), fmt="{:.2g}")
        N.num("f3.sec.physbayes", sm, srcs, C.status_of(L), fmt="{:.2g}")
    else:
        N.ph("f3.speedup")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    S.plt.close(fig)
    return N


if __name__ == "__main__":
    res, paper = C.env_paths()
    for k, v in run(res, paper).items.items():
        print(k, v[0])
