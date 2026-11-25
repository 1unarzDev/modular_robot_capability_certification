"""Shared loaders, scorers and the numbers/table writers for the paper analysis scripts.

Every paper number is a LaTeX macro written to paper/numbers.tex with a comment naming its source file(s) and its
label (exploratory / confirmatory). A number whose input does not exist yet becomes a red placeholder, so the paper
always compiles. The input contracts (file globs and columns) are SCHEMAS below and in docs/board/paper.md; a file
that breaks its schema raises, it is never silently skipped.

Scorer definitions (identical to cfcap/baselines/base.py:metrics, cross-checked in tests/test_paper_analysis.py):
  decision        = (episode, candidate, leg, tau), tau in TAUS
  capable         = truth >= tau
  go              = finite lower >= tau          (NaN / -inf lower = refuse; capxfer scorer-bug lesson)
  cert_frac       = #(go & capable) / #capable
  unsafe_commit   = #(go & ~capable) / #incapable
  ep_any_unsafe   = share of episodes with at least one unsafe go (what the simultaneous guarantee bounds)
  entry_cov       = #(lower <= truth) / #entries   (per (candidate, leg); tau-free)
  ep_cov          = share of episodes whose every entry is covered (simultaneous coverage)
Uncertainty: percentile bootstrap over episodes (never over decisions), B = 1000, seed 0.
"""
from __future__ import annotations

import glob
import math
import os
import pathlib
import re

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "results"
PAPER = ROOT / "paper"

TAUS = (0.6, 0.7, 0.8)          # cfcap/baselines/base.py TAUS
ALPHA = 0.10                    # nominal level used by the roles' conformal layers (scripts/baselines_pilot.py)
U_STAR = 0.01                   # matched unsafe-commit rate for "certified fraction at matched unsafe-go" (fixed
                                # 2026-11-25 before any Stage C result; capxfer's sim unsafe-go was <= 1.3 %)
B_BOOT = 1000
LEGS = ("surge+", "surge-", "sway+", "sway-", "yaw+", "yaw-", "diag++", "diag+-", "diag-+", "diag--")

# ---------------------------------------------------------------------------------------------------------------
# Input contracts. glob is relative to results/. "req" columns must exist; "opt" may be absent (filled with NaN).
SCHEMAS = {
    "dec": dict(
        glob="*/paper/dec_*.csv*",
        req=("method", "route", "posterior", "train", "test_plant", "split", "status", "eid", "shape", "cand",
             "kind", "cand_size", "has_fault", "leg", "truth", "point", "scale", "lower"),
        opt=("family", "alpha", "seconds"),
    ),
    "probe": dict(
        glob="*/paper/probe_*.csv*",
        req=("policy", "split", "status", "eid", "shape", "n_probes", "probe_time_s", "action", "n_dec",
             "n_capable", "n_incapable", "n_certified", "n_unsafe", "n_amb_before", "n_amb_after"),
        opt=("utility", "wall_s", "test_plant"),
    ),
    "amb": dict(
        glob="*/paper/amb_het*.csv*",
        req=("source", "delta", "subset", "n_dec", "amb_share", "status"),
        opt=("lo", "hi", "plant"),
    ),
    "div": dict(
        glob="*/paper/diversity*.csv*",
        req=("arch", "diversity", "n_train_families", "max_train_n", "seed", "test_morph", "test_n", "heldout",
             "metric", "value", "status"),
        opt=("split",),
    ),
    "lat": dict(
        glob="*/paper/latency*.csv*",
        req=("method", "shape", "n_wheels", "sec_per_decision", "hardware", "status"),
        opt=("n",),
    ),
}


class SchemaError(ValueError):
    pass


def load(kind: str, res: pathlib.Path = RES) -> tuple[pd.DataFrame, list[str]]:
    """Concatenate every file matching the contract glob. Returns (frame, relative source paths)."""
    sc = SCHEMAS[kind]
    files = sorted(glob.glob(str(pathlib.Path(res) / sc["glob"])))
    frames, srcs = [], []
    for f in files:
        d = pd.read_csv(f)
        miss = [c for c in sc["req"] if c not in d.columns]
        if miss:
            raise SchemaError(f"{f}: missing required columns {miss} (contract '{kind}', docs/board/paper.md)")
        for c in sc["opt"]:
            if c not in d.columns:
                d[c] = np.nan
        if "status" in d.columns:
            bad = set(d["status"].dropna().unique()) - {"exploratory", "confirmatory"}
            if bad:
                raise SchemaError(f"{f}: status must be exploratory|confirmatory, got {sorted(bad)}")
        d["_src"] = rel(f)
        frames.append(d)
        srcs.append(rel(f))
    if not frames:
        return pd.DataFrame(columns=list(sc["req"]) + list(sc["opt"]) + ["_src"]), []
    return pd.concat(frames, ignore_index=True), srcs


def rel(p) -> str:
    p = pathlib.Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def status_of(d: pd.DataFrame) -> str:
    """A number is confirmatory only if every row behind it is."""
    s = set(d["status"].dropna().unique()) if "status" in d else set()
    return "confirmatory" if s == {"confirmatory"} else "exploratory"


# ---------------------------------------------------------------------------------------------------------------
# Scorers (pure numpy; unit-tested)

def go_mask(lower, tau):
    lo = np.asarray(lower, float)
    return np.isfinite(lo) & (lo >= tau)


def decision_counts(d: pd.DataFrame, lower_col: str = "lower", taus=TAUS) -> dict:
    """Counts over decisions (rows x taus). d needs truth, eid and the lower column."""
    tr = d["truth"].to_numpy(float)
    lo = d[lower_col].to_numpy(float)
    ok = np.isfinite(tr)
    tr, lo = tr[ok], lo[ok]
    eid = d["eid"].to_numpy()[ok]
    n_cap = n_inc = n_cert = n_uns = 0
    uns_ep = set()
    for tau in taus:
        g = go_mask(lo, tau)
        cap = tr >= tau
        n_cap += int(cap.sum())
        n_inc += int((~cap).sum())
        n_cert += int((g & cap).sum())
        u = g & ~cap
        n_uns += int(u.sum())
        uns_ep.update(eid[u].tolist())
    n_ep = len(np.unique(eid))
    return dict(n_ep=n_ep, n_capable=n_cap, n_incapable=n_inc, n_certified=n_cert, n_unsafe=n_uns,
                cert_frac=_ratio(n_cert, n_cap), unsafe_commit=_ratio(n_uns, n_inc),
                ep_any_unsafe=_ratio(len(uns_ep), n_ep))


def coverage(d: pd.DataFrame, lower_col: str = "lower") -> dict:
    """Entry and episode-simultaneous coverage of truth by the lower bound (refusal = -inf covers)."""
    tr = d["truth"].to_numpy(float)
    lo = np.nan_to_num(d[lower_col].to_numpy(float), nan=-np.inf)
    ok = np.isfinite(tr)
    cov = lo[ok] <= tr[ok]
    eid = d["eid"].to_numpy()[ok]
    if not len(cov):
        return dict(entry_cov=np.nan, ep_cov=np.nan, n_entries=0, n_ep=0)
    ep = pd.Series(cov).groupby(eid).all()
    return dict(entry_cov=float(cov.mean()), ep_cov=float(ep.mean()), n_entries=int(len(cov)), n_ep=int(len(ep)))


def matched_margin(z_incapable: np.ndarray, u_star: float = U_STAR) -> float:
    """Smallest k such that the share of incapable decisions with z >= k is <= u_star, where
    z = (point - tau) / scale and the shifted rule is go iff point - k * scale >= tau  (iff z >= k)."""
    z = np.sort(np.asarray(z_incapable, float)[np.isfinite(z_incapable)])[::-1]
    n = len(z)
    if n == 0:
        return -math.inf
    m = int(math.floor(u_star * n + 1e-12))      # unsafe decisions allowed
    if m >= n:
        return -math.inf
    return float(np.nextafter(z[m], math.inf))  # exclude z[m] and everything below it


def _point_scale(d: pd.DataFrame):
    p = d["point"].to_numpy(float)
    s = d["scale"].to_numpy(float)
    lo = d["lower"].to_numpy(float)
    nop = ~np.isfinite(p)
    p = np.where(nop, lo, p)                    # lower-only methods: shift the lower bound itself
    s = np.where(nop | ~np.isfinite(s) | (s <= 0), 1.0, s)
    return p, s


def z_scores(d: pd.DataFrame, taus=TAUS):
    """Stack (z, capable, eid, row) over taus."""
    p, s = _point_scale(d)
    tr = d["truth"].to_numpy(float)
    eid = d["eid"].to_numpy()
    zs, caps, eids, rows = [], [], [], []
    for tau in taus:
        ok = np.isfinite(tr)
        zs.append(((p - tau) / s)[ok])
        caps.append((tr >= tau)[ok])
        eids.append(eid[ok])
        rows.append(np.flatnonzero(ok))
    return np.concatenate(zs), np.concatenate(caps), np.concatenate(eids), np.concatenate(rows)


def matched_lower(d_fit: pd.DataFrame, d_eval: pd.DataFrame, u_star: float = U_STAR, taus=TAUS):
    """Pick k on d_fit at unsafe-commit u_star (pooled over taus), return lower = point - k*scale on d_eval and k."""
    z, cap, _, _ = z_scores(d_fit, taus)
    z = np.where(np.isfinite(z), z, -np.inf)
    k = matched_margin(z[~cap], u_star)
    p, s = _point_scale(d_eval)
    with np.errstate(invalid="ignore"):
        lo = p - k * s
    return np.where(np.isfinite(lo), lo, -np.inf), k


def ep_counts(d: pd.DataFrame, lower, taus=TAUS) -> pd.DataFrame:
    """Per-episode decision counts (capable, certified, incapable, unsafe) for a lower-bound vector."""
    tr = d["truth"].to_numpy(float)
    lo = np.asarray(lower, float)
    ok = np.isfinite(tr)
    eid = d["eid"].to_numpy()
    cols = {k: np.zeros(len(d)) for k in ("cap", "cert", "inc", "uns")}
    for tau in taus:
        g = go_mask(lo, tau) & ok
        cap = ok & (tr >= tau)
        inc = ok & (tr < tau)
        cols["cap"] += cap
        cols["cert"] += g & cap
        cols["inc"] += inc
        cols["uns"] += g & inc
    return pd.DataFrame(cols).groupby(eid).sum()


def boot_ratio(num: np.ndarray, den: np.ndarray, B: int = B_BOOT, seed: int = 0, level: float = 0.95):
    """Percentile CI of sum(num)/sum(den) under an episode (cluster) bootstrap; num, den are per-episode."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    E = len(num)
    if E < 2 or den.sum() == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(E, np.full(E, 1 / E), size=B).astype(float)
    dn = w @ den
    r = np.where(dn > 0, (w @ num) / np.where(dn > 0, dn, 1), np.nan)
    r = r[np.isfinite(r)]
    a = (1 - level) / 2
    return float(np.quantile(r, a)), float(np.quantile(r, 1 - a))


def _ratio(a, b):
    return float(a / b) if b else float("nan")


# ---------------------------------------------------------------------------------------------------------------
# Trivial baselines computed here from the roles' truth columns (COMMON.md §6: constant answer, metadata only)

def trivial_methods(d: pd.DataFrame) -> pd.DataFrame:
    """For each (test_plant, split) block of the *first* method's rows (truth is shared across methods):
      always_go     lower = +inf  (constant 'yes')
      never_go      lower = -inf  (constant 'no'; cert_frac 0 by construction)
      metadata      point/scale = mean/sd of truth per (kind, cand_size, has_fault, leg), 2-fold cross-fit over
                    episodes (each fold predicted from the other), then the same simultaneous-free shift rule.
    has_fault is metadata only in the sense that the robot would need a detector for it; we also give metadata_nf
    (grouped without has_fault), the honest 'shape only' baseline."""
    if d.empty:
        return d
    first = d["method"].iloc[0]
    base = d[d["method"] == first].copy()
    out = []
    for name in ("always_go", "never_go"):
        b = base.copy()
        b["method"], b["route"], b["posterior"] = name, "none", "none"
        b["point"] = np.nan
        b["scale"] = np.nan
        b["lower"] = np.inf if name == "always_go" else -np.inf
        b["seconds"] = 0.0
        out.append(b)
    for name, keys in (("metadata", ["kind", "cand_size", "has_fault", "leg"]),
                       ("metadata_nf", ["kind", "cand_size", "leg"])):
        parts = []
        for _, blk in base.groupby(["test_plant", "split"]):
            blk = blk.copy()
            ue = np.sort(blk["eid"].unique())
            fold = dict(zip(ue, np.arange(len(ue)) % 2))
            f = blk["eid"].map(fold).to_numpy()
            pt = np.full(len(blk), np.nan)
            sc = np.full(len(blk), np.nan)
            for k in (0, 1):
                tr_, te_ = blk[f != k], f == k
                if tr_.empty:
                    continue
                st = tr_.groupby(keys)["truth"].agg(["mean", "std"]).reset_index()
                m = blk[te_][keys].merge(st, on=keys, how="left")
                pt[te_] = m["mean"].to_numpy()
                sc[te_] = m["std"].fillna(0.0).to_numpy() + 1e-3
            blk["point"], blk["scale"] = pt, sc
            blk["lower"] = np.nan   # no guarantee of its own; only the matched-u* column is reported
            parts.append(blk)
        b = pd.concat(parts)
        b["method"], b["route"], b["posterior"], b["seconds"] = name, "none", "none", 0.0
        out.append(b)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------------------------------------------
# Numbers -> paper/numbers.tex

_DIG = dict(zip("0123456789", ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")))


def macro_name(key: str) -> str:
    """'t1.cert.amortized_sim.T1' -> 'cfTOneCertAmortizedSimTOne' (LaTeX macro names are letters only)."""
    parts = re.split(r"[^A-Za-z0-9]+", key)
    out = []
    for p in parts:
        if not p:
            continue
        p = "".join(_DIG.get(c, c) for c in p)
        out.append(p[0].upper() + p[1:])
    name = "cf" + "".join(out)
    assert re.fullmatch(r"[A-Za-z]+", name), name
    return name


class Numbers:
    def __init__(self, script: str):
        self.script = script
        self.items: dict[str, tuple[str, str, str]] = {}

    def set(self, key, value, sources, status="exploratory"):
        src = ", ".join(sorted(set(sources))) if not isinstance(sources, str) else sources
        self.items[macro_name(key)] = (value, src, status)

    def ph(self, key, why="input missing"):
        n = macro_name(key)
        self.items[n] = (rf"\cfPH{{{n}}}", why, "placeholder")

    def pct(self, key, x, sources, status="exploratory", d=1):
        if x is None or not np.isfinite(x):
            return self.ph(key, f"no finite value from {sources}")
        self.set(key, f"{100 * x:.{d}f}\\,\\%", sources, status)

    def num(self, key, x, sources, status="exploratory", fmt="{:,.0f}"):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return self.ph(key, f"no finite value from {sources}")
        self.set(key, fmt.format(x), sources, status)


def write_numbers(all_numbers: list[Numbers], path: pathlib.Path):
    lines = ["% GENERATED by scripts/analysis/make_all.py -- do not edit. Each macro: value % source [label].",
             r"\providecommand{\cfPH}[1]{\textcolor{red}{\textbf{[#1]}}}",
             r"\providecommand{\cfExp}{\textsuperscript{e}}  % exploratory marker",
             ""]
    seen = {}
    for N in all_numbers:
        lines.append(f"% ---- {N.script}")
        for name, (v, src, st) in sorted(N.items.items()):
            if name in seen:
                raise ValueError(f"macro {name} defined by {seen[name]} and {N.script}")
            seen[name] = N.script
            lines.append(f"\\newcommand{{\\{name}}}{{{v}}}  % {src} [{st}]")
        lines.append("")
    path.write_text("\n".join(lines))
    return seen


def write_table(path: pathlib.Path, body: str, sources, note: str = ""):
    src = ", ".join(sorted(set(sources))) if sources else "none yet (placeholders)"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"% GENERATED by scripts/analysis -- sources: {src}\n% {note}\n{body}\n")


def fmt_pct(x, d=1, mark=""):
    if x is None or not np.isfinite(x):
        return r"\textemdash"
    return f"{100 * x:.{d}f}{mark}"


def placeholder_table(path, caption_key, ncols, why):
    why_tex = why.replace("_", r"\_").replace("*", r"$\ast$")
    body = (f"\\begin{{tabular}}{{{'l' * ncols}}}\\toprule\n"
            f"\\multicolumn{{{ncols}}}{{c}}{{\\cfPH{{{caption_key}}}: {why_tex}}}\\\\\\bottomrule\n\\end{{tabular}}")
    write_table(path, body, [], why)


def env_paths(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.environ.get("CFCAP_PAPER_RESULTS", str(RES)))
    ap.add_argument("--paper", default=os.environ.get("CFCAP_PAPER_DIR", str(PAPER)))
    a = ap.parse_args(argv)
    return pathlib.Path(a.results), pathlib.Path(a.paper)
