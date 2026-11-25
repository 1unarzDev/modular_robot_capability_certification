"""Exploratory A5 analysis, CSV inputs only; no simulation, torch, or GPU.

Run from anywhere with the environment's $PY. Only writes the named result CSV.
Single-module features cannot be reconstructed from aggregate tables. The CV
exercise is explicitly a MULTI-MODULE EPISODE surrogate, not a single-decision
certificate. Misfit uses known latents (oracle), not an operational residual.
"""
from pathlib import Path
import csv
import hashlib
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/codex/a5_flips_analysis.csv"
SOURCE = "results/stress/"
ROWS = []


def emit(section, metric, value="", source="", **kw):
    ROWS.append(dict(status="exploratory", section=section, metric=metric,
                     value=value, source=source, **kw))


def read(name):
    path = ROOT / SOURCE / name
    emit("provenance", "sha256", hashlib.sha256(path.read_bytes()).hexdigest(),
         SOURCE + name)
    with path.open() as f:
        return list(csv.DictReader(f))


def auc(y, p):
    n = y.sum()
    return (rankdata(p)[y == 1].sum() - n * (n + 1) / 2) / (n * (len(y) - n))


def folds(y, seed):
    """Stratified, shuffled episode folds; every episode occurs in one test fold."""
    rng = np.random.default_rng(seed)
    f = np.empty(len(y), int)
    for label in (0, 1):
        ix = np.flatnonzero(y == label)
        rng.shuffle(ix)
        f[ix] = np.arange(len(ix)) % 5
    return f


def fit(x, y):
    """Ridge logistic; training-only scaling; sum loss + 0.5 ||beta||^2."""
    mu, sd = x.mean(0), x.std(0)
    sd[sd < 1e-12] = 1
    z = np.column_stack([np.ones(len(x)), (x - mu) / sd])

    def objective(b):
        t = z @ b
        loss = np.logaddexp(0, t).sum() - y @ t + .5 * (b[1:] @ b[1:])
        grad = z.T @ (expit(t) - y) + np.r_[0., b[1:]]
        return loss, grad

    opt = minimize(objective, np.zeros(z.shape[1]), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=500, gtol=1e-7))
    if not opt.success:
        raise RuntimeError(opt.message)
    return opt.x, mu, sd


def predict(model, x):
    b, mu, sd = model
    return expit(b[0] + ((x - mu) / sd) @ b[1:])


def oof(x, y, seed):
    f = folds(y, seed)
    p = np.zeros(len(y))
    for k in range(5):
        p[f == k] = predict(fit(x[f != k], y[f != k]), x[f == k])
    return p, f


def stats(y, abstain):
    tp = int((y.astype(bool) & abstain).sum())
    fp = int((~y.astype(bool) & abstain).sum())
    fn = int((y.astype(bool) & ~abstain).sum())
    tn = len(y) - tp - fp - fn
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp / (tp + fp) if tp + fp else np.nan,
                recall=tp / (tp + fn), abstain_share=float(abstain.mean()),
                retained_flip_rate=fn / (fn + tn) if fn + tn else np.nan)


def main():
    agg = read("A5_flips.csv")
    legtype = read("A5_by_legtype.csv")
    read("A5_by_leg.csv")
    clamp = read("A5_clamp.csv")
    src = SOURCE + "A5_flips.csv"
    for r in agg:
        if r["label"] != "rho_dec":
            continue
        if r["child"] == "single" or (r["cond"] == "all" and r["sel"] == "all"):
            for metric in ("episodes", "decisions", "flip", "flip_lo", "flip_hi",
                           "share_d05", "d_q01", "d_q99"):
                emit("aggregate", metric, r[metric], src, arm=r["arm"],
                     condition=r["cond"], child=r["child"], selection=r["sel"])

    def one(arm, cond, sel="all"):
        return next(r for r in agg if r["arm"] == arm and r["label"] == "rho_dec"
                    and r["cond"] == cond and r["child"] == "single" and r["sel"] == sel)

    for arm in ("P0", "P1"):
        for cond in sorted({r["cond"] for r in agg}):
            a, b = one(arm, cond), one(arm, cond, "faulted")
            # Recover integer counts only where unambiguous under five-significant-digit CSV rounding.
            total, fault = [round(float(r["flip"]) * int(r["decisions"])) for r in (a, b)]
            for r, count in ((a, total), (b, fault)):
                assert abs(count / int(r["decisions"]) - float(r["flip"])) < 5e-7
            n = int(a["decisions"]) - int(b["decisions"])
            emit("healthy_single", "flip", (total - fault) / n, src, arm=arm,
                 condition=cond, child="single", selection="no_fault",
                 note="Counts recovered from rounded aggregate rates; not episode-level data.")
            emit("healthy_single", "flips", total - fault, src, arm=arm, condition=cond)
            emit("healthy_single", "decisions", n, src, arm=arm, condition=cond)
        for a, b in (("all", "base"), ("all", "all-aniso_mu"), ("aniso_mu", "base"),
                     ("compliance", "base"), ("load_transfer", "base"), ("aniso", "base")):
            emit("effect_contrast", "flip_rate_difference", float(one(arm, a)["flip"]) -
                 float(one(arm, b)["flip"]), src, arm=arm, condition=a + " minus " + b,
                 note="Aggregate rate difference; cannot identify overlap or attribute individual flips.")
    emit("payload_contrast", "P1_minus_P0_single_flip", float(one("P1", "all")["flip"]) -
         float(one("P0", "all")["flip"]), src,
         note="Different arm draws; not a paired payload effect or dose-response estimate.")

    for feature, availability in {
        "single leg type": "Absent: leg and leg-type summaries exclude singles.",
        "single fault type and gain": "Absent; only aggregate all/faulted selections for singles.",
        "single fault count": "Absent; clamp n_fault is multi-module only.",
        "absolute T1 rho_dec minus tau margin": "Absent at episode/decision level; near summaries are multi-module only.",
        "payload": "Arm summaries only; no per-episode mass or location in CSVs.",
        "friction scatter": "Condition ablation only; no per-wheel draws in CSVs.",
        "single clamp misfit": "Absent: A5_clamp.csv contains no singles.",
        "flip direction and T1 go": "Absent: cannot estimate unsafe-go reduction or go-only gate performance.",
        "operational residual": "Absent: recorded misfit compares plants with known latents; oracle proxy.",
    }.items():
        emit("availability", feature, "unavailable_for_requested_single_model", SOURCE + "A5_*.csv",
             note=availability)

    for r in legtype:
        if r["label"] == "rho_dec" and r["cond"] == "all":
            for metric in ("decisions", "flips", "flip", "near", "flip_given_near"):
                emit("multimodule_margin_context", metric, r[metric], SOURCE + "A5_by_legtype.csv",
                     arm=r["arm"], child=r["size"], feature=r["leg_type"],
                     note="Near means |rho_dec(T1)-tau| < 0.05; not a single-module estimate.")

    keys = [(r["shard_seed"], r["shape"], r["ep"]) for r in clamp]
    assert len(set(keys)) == len(keys)
    assert "single" not in {r["shape"] for r in clamp}
    y = np.array([int(r["flip"]) for r in clamp])
    shape = np.array([r["shape"] for r in clamp])
    nf = np.array([int(r["n_fault"]) for r in clamp])
    src = SOURCE + "A5_clamp.csv"
    emit("cv_protocol", "method", "ridge_logistic", src,
         note="Five outer stratified episode folds, seed=905; thresholds use five inner folds, seed=1905+k. "
              "L2 penalty=1 on standardized coefficients; unpenalized intercept; no class weights. "
              "Endpoint is any decision flip in an episode, not a per-decision flip. Independent episode keys; "
              "single shard only; no held-out simulator, morphology or seed validation.")
    emit("cv_protocol", "scope", "multi-module episodes only; cannot validate single certificate gate", src)
    for group in sorted(set(shape)):
        m = shape == group
        emit("clamp_composition", "episodes", int(m.sum()), src, child=group)
        emit("clamp_composition", "positive_episodes", int(y[m].sum()), src, child=group)
    for k in sorted(set(nf)):
        m = nf == k
        emit("clamp_fault_count", "episode_flip_rate", float(y[m].mean()), src,
             selection=f"n_fault={k}", note=f"episodes={m.sum()}; positives={y[m].sum()}")

    logclamp = np.log1p([float(r["D_imu_clamp"]) for r in clamp])
    logv2 = np.log1p([float(r["D_imu_v2"]) for r in clamp])
    for name, v in (("clamp", logclamp), ("v2", logv2)):
        for selection, mask in (("all", np.ones(len(y), bool)), ("faulted", nf > 0)):
            emit("univariate_ranking", "auc", auc(y[mask], v[mask]), src,
                 model=name, selection=selection, note="In-sample raw-score ranking; not CV.")
    shape_x = np.column_stack([shape == s for s in ("pair", "L3", "square")]).astype(float)
    models = {
        "clamp_only": (logclamp[:, None], ["log1p_clamp"]),
        "v2_only": (logv2[:, None], ["log1p_v2"]),
        "shape_only": (shape_x, ["pair", "L3", "square"]),
        "shape_clamp": (np.column_stack([shape_x, logclamp]), ["pair", "L3", "square", "log1p_clamp"]),
        "shape_clamp_fault_oracle": (np.column_stack([shape_x, logclamp, nf]),
                                      ["pair", "L3", "square", "log1p_clamp", "n_fault"]),
    }
    for name, (x, names) in models.items():
        p, fold = oof(x, y, 905)
        for metric, value in (("auc", auc(y, p)), ("brier", float(((p - y) ** 2).mean()))):
            emit("cv_score", metric, value, src, model=name)
        gates = {"probability_ge_0.5": p >= .5}
        for target in (.8, .95):
            ab = np.zeros(len(y), bool)
            for k in range(5):
                train, test = fold != k, fold == k
                ip, _ = oof(x[train], y[train], 1905 + k)
                pos = np.sort(ip[y[train] == 1])[::-1]
                threshold = pos[int(np.ceil(target * len(pos))) - 1]
                ab[test] = p[test] >= threshold
                emit("gate_threshold", "threshold", threshold, src, model=name,
                     rule=f"inner_target_recall_{target}", fold=k)
            gates[f"inner_target_recall_{target}"] = ab
        gates["abstain_all"] = np.ones(len(y), bool)
        for rule, ab in gates.items():
            for metric, value in stats(y, ab).items():
                emit("cv_gate", metric, value, src, model=name, rule=rule)
            # Conditional bootstrap of held-out predictions, not a refit/model-selection CI.
            rng = np.random.default_rng(2905)
            bs = []
            for _ in range(1000):
                ii = rng.integers(len(y), size=len(y))
                if y[ii].sum():
                    s = stats(y[ii], ab[ii])
                    bs.append([s["precision"], s["recall"]])
            for j, metric in enumerate(("precision", "recall")):
                lo, hi = np.nanpercentile(np.array(bs)[:, j], [2.5, 97.5]) if np.isfinite(np.array(bs)[:, j]).any() else (np.nan, np.nan)
                emit("cv_gate_uncertainty", metric + "_lo", lo, src, model=name, rule=rule)
                emit("cv_gate_uncertainty", metric + "_hi", hi, src, model=name, rule=rule,
                     note="95% episode bootstrap conditional on OOF predictions; excludes fitting uncertainty.")
        b, _, _ = fit(x, y)
        for feature, coefficient in zip(["intercept"] + names, b):
            emit("full_fit_coefficient", "log_odds_per_training_sd", coefficient, src,
                 model=name, feature=feature,
                 note="Full-data descriptive fit; 2x3 reference for shape indicators; intercept at feature means.")
        for i, r in enumerate(clamp):
            emit("oof_prediction", "flip_probability", p[i], src, model=name,
                 child=r["shape"], fold=int(fold[i]), episode=f'{r["shard_seed"]}/{r["shape"]}/{r["ep"]}',
                 observed_flip=int(y[i]))
    emit("conclusion", "single_gate", "not_estimable", SOURCE + "A5_*.csv",
         note="No single-level feature rows, no margins, no single clamp. Aggregate rates cannot be treated as "
              "independent observations for logistic regression. Multi-module oracle proxy is exploratory only; "
              "does not establish a calibrated certificate or an unsafe-go risk bound.")
    fields = list(dict.fromkeys(k for r in ROWS for k in r))
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ROWS)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
