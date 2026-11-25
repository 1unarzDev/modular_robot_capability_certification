"""Generic shard wrapper for cfcap cluster jobs: ``python -m cfcap.runner <task> --shard K --n-shards N [...]``.

One job = one shard of one task. A task (see `Task`) splits its shard into *units* (e.g. 250 episodes, one SMC
episode batch, one training run). The runner gives every task the same contract:

  outputs     OUT/units/<unit>.npz (columnar: one named array per column, equal length) + <unit>.json (rows,
              sha256, wall). OUT/manifest.json lists every finished unit, the task args, the code version and the
              status (running / complete / interrupted). OUT is --out, else $CRANE_OUT_ROOT/$CRANE_JOB_ID (a rented
              box sets CRANE_OUT_ROOT=/workspace/out; crane-vast pulls it back), else
              $CRANE_RUN_DIR/artifacts/out (a free worker: uploaded as raw artifacts), else
              $CFCAP_OUT/runs/<task>/shard<K>.
  resume      a unit whose .json exists (and whose .npz hash matches) is skipped; every file is written to a temp
              name and renamed, so a kill -9 leaves no half file. A unit may also checkpoint its own state
              (ctx.save_state / ctx.load_state, atomic) and continue from it after a restart. A re-run with
              different task args into the same OUT is refused (exit 2) unless --fresh.
  progress    $CRANE_RUN_DIR/progress.json (and OUT/progress.json) {done, total, failed, rate_per_s, eta_s, phase,
              unit, unit_frac, updated}, rewritten at least every HEARTBEAT_S (30 s) by a heartbeat thread, plus a
              ``RC-PROGRESS {"done": n, "total": m}`` line on stdout per change and per minute.
  metrics     $CRANE_RUN_DIR/metrics.json (else OUT/metrics.json): flat, at most 1000 finite numbers, names
              matching the coordinator's ^[A-Za-z0-9._:/-]{1,128}$ (runner keys first, then the task's summary).
  signals     SIGTERM, SIGUSR1 (yield) and SIGINT set a stop flag: the task sees ctx.should_stop() at its next safe
              point, checkpoints and raises `Stop`; the runner writes the manifest (interrupted) and exits 75
              (research-cluster: a release, not a failure). A unit that ignores the flag is interrupted after a
              grace (SIGTERM 20 s, SIGUSR1 60 s), never inside an atomic rename.
  exit codes  0 complete, 75 interrupted (resumable), 2 usage / args mismatch / unknown task, 3 missing input or
              module, 1 a task error (traceback in the log).

Tasks are registered by name in TASKS as "module:attr" (lazy, so a role's heavy imports load only for its task) or
given ad hoc with --task-spec module:attr. The attr is a `Task`. Built-in: `selftest`.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import math
import os
import pickle
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

SCHEMA = "cfcap.runner/1"
HEARTBEAT_S = 30.0
RC_LINE_S = 60.0
GRACE_S = {"SIGTERM": 20.0, "SIGINT": 20.0, "SIGUSR1": 60.0}
EXIT_OK, EXIT_TASK, EXIT_USAGE, EXIT_INPUT, EXIT_YIELD = 0, 1, 2, 3, 75
METRIC_NAME = re.compile(r"^[A-Za-z0-9._:/-]{1,128}$")
UNIT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
MAX_METRICS = 1000
REPO = Path(__file__).resolve().parents[1]

# Task name -> "module:attr" (a Task). Roles add their module; the names are what make_jobs.py emits.
TASKS: dict[str, str] = {
    "selftest": "cfcap.runner:SELFTEST",
    "gen_shard": "cfcap.gen.shard:TASK",          # role generator: T1 episodes (train/cal/test/H1-H7 splits)
    "train": "cfcap.models.train:TASK",           # role models: one training run (arch x diversity x seed)
    "smc": "cfcap.baselines.smc:TASK",            # role baselines: GPU-SMC posterior over test episodes
    "t2_eval": "cfcap.t2.eval:TASK",              # role stress: T2 (JAX closed loop) evaluation shard
}
OWNER = {"gen_shard": "generator", "train": "models", "smc": "baselines", "t2_eval": "stress"}


def grace_s(sig_name: str) -> float:
    """Seconds a unit gets to reach its safe point after a signal (CFCAP_GRACE_S overrides, for tests)."""
    v = os.environ.get("CFCAP_GRACE_S")
    return float(v) if v else GRACE_S.get(sig_name, 20.0)


class Stop(Exception):
    """Raised (by the task at a safe point, or by the runner after the grace) to end the job resumably."""


class InputMissing(Exception):
    """A task's input file or module is absent on this host (exit 3)."""


@dataclass
class Task:
    """A pluggable task.

    units(args, shard, n_shards) -> unit ids of this shard (stable across restarts; [A-Za-z0-9._-]).
    run_unit(unit, ctx) -> {column: 1-D array-like of equal length} (columnar rows), or None for no rows.
        Long units call ctx.progress(frac) and check ctx.should_stop(); on True they ctx.save_state(...) and
        raise Stop. A restarted unit reads ctx.load_state() (None the first time).
    add_args(parser): task flags (after the runner's own).
    summarize(records, args) -> {name: number} for metrics.json; records are the per-unit .json dicts (rows,
        wall_s, sha256, columns, and whatever run_unit put in ctx.meta).
    requires: modules that must import (exit 3 otherwise), e.g. ("numpy", "torch").
    """
    name: str
    units: Callable[[argparse.Namespace, int, int], Sequence[str]]
    run_unit: Callable[[str, "UnitCtx"], Any]
    add_args: Callable[[argparse.ArgumentParser], None] | None = None
    summarize: Callable[[list[dict], argparse.Namespace], dict] | None = None
    requires: tuple[str, ...] = ("numpy",)
    doc: str = ""


# ------------------------------------------------------------------ atomic files

class _Guard:
    """Signals defer their hard interrupt while a rename is in flight."""

    def __init__(self):
        self.depth = 0
        self.pending = False


_GUARD = _Guard()


@contextlib.contextmanager
def critical():
    _GUARD.depth += 1
    try:
        yield
    finally:
        _GUARD.depth -= 1
        if _GUARD.depth == 0 and _GUARD.pending:
            _GUARD.pending = False
            raise Stop("grace expired")


def atomic_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with critical():
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)


def atomic_json(path: Path, obj: Any) -> None:
    atomic_bytes(path, (json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n").encode())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------ helpers for tasks

def shard_range(n_items: int, shard: int, n_shards: int) -> range:
    """Contiguous, balanced slice [lo, hi) of n_items for this shard."""
    lo = n_items * shard // n_shards
    return range(lo, n_items * (shard + 1) // n_shards)


def chunk_units(prefix: str, items: range, chunk: int) -> list[str]:
    """Unit ids '<prefix>-<lo>-<hi>' covering `items` in chunks (parse back with parse_chunk)."""
    return [f"{prefix}-{a:07d}-{min(a + chunk, items.stop):07d}" for a in range(items.start, items.stop, chunk)]


def parse_chunk(unit: str) -> range:
    lo, hi = unit.rsplit("-", 2)[-2:]
    return range(int(lo), int(hi))


def unit_seed(*parts: Any) -> int:
    """A stable 63-bit seed from the task name, unit id and base seed (same on every host and restart)."""
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") >> 1


# ------------------------------------------------------------------ the unit context

@dataclass
class UnitCtx:
    runner: "Runner"
    unit: str
    seed: int
    args: argparse.Namespace
    meta: dict = field(default_factory=dict)

    @property
    def out_dir(self) -> Path:
        return self.runner.out

    @property
    def inputs_dir(self) -> Path:
        return self.runner.inputs

    @property
    def state_path(self) -> Path:
        """Where this unit may keep its own checkpoint (e.g. torch.save via atomic_bytes); removed when it ends."""
        return self.runner.out / "state" / f"{self.unit}.state"

    def should_stop(self) -> bool:
        return self.runner.stop_signal is not None

    def progress(self, frac: float, phase: str | None = None) -> None:
        self.runner.unit_frac = max(0.0, min(1.0, float(frac)))
        if phase:
            self.runner.phase = phase
        self.runner.maybe_progress()

    def save_state(self, obj: Any) -> None:
        buf = io.BytesIO()
        pickle.dump(obj, buf, protocol=pickle.HIGHEST_PROTOCOL)
        atomic_bytes(self.state_path, buf.getvalue())

    def load_state(self) -> Any:
        try:
            return pickle.loads(self.state_path.read_bytes())
        except (OSError, pickle.UnpicklingError, EOFError):
            return None

    def input(self, rel: str) -> Path:
        """A pushed input (`$CRANE_INPUTS_DIR/rel`); InputMissing (exit 3) when it is not on this host."""
        p = self.runner.inputs / rel
        if not p.exists():
            raise InputMissing(f"input {rel} not under {self.runner.inputs}")
        return p


# ------------------------------------------------------------------ the runner

def _git_version(root: Path) -> dict:
    stamp = root / "CFCAP_COMMIT"          # a `git archive` export (scripts/make_jobs.py stage) carries its commit
    if stamp.is_file():
        return {"commit": stamp.read_text().strip(), "dirty": False, "export": True}
    try:
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return {"commit": sha or None, "dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def resolve_out(explicit: str | None, task: str, shard: int, env=os.environ) -> Path:
    if explicit:
        return Path(explicit)
    job = env.get("CRANE_JOB_ID")
    root = env.get("CRANE_OUT_ROOT") or ("/workspace/out" if Path("/workspace/out").is_dir() else None)
    if job and root:
        rel = Path(*[p for p in Path(job).parts if p not in ("", ".", "..", "/")])
        return Path(root) / rel
    if env.get("CRANE_RUN_DIR"):
        return Path(env["CRANE_RUN_DIR"]) / "artifacts" / "out"
    base = Path(env.get("CFCAP_OUT") or Path.home() / ".cache/modular-runs/2026-11/cfcap")
    return base / "runs" / task / f"shard{shard:04d}"


def clean_metrics(d: dict, limit: int = MAX_METRICS) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        if len(out) >= limit:
            break
        if isinstance(v, bool) or type(v).__name__ == "bool_":
            continue
        if not isinstance(v, (int, float)):
            try:
                v = float(v)          # numpy scalars
            except (TypeError, ValueError):
                continue
        if not METRIC_NAME.match(str(k)) or not math.isfinite(float(v)):
            continue
        out[str(k)] = float(v)
    return out


class Runner:
    def __init__(self, task: Task, args: argparse.Namespace, task_argv: list[str], env=os.environ):
        self.task, self.args, self.task_argv = task, args, task_argv
        self.out = resolve_out(args.out, task.name, args.shard, env)
        self.run_dir = Path(env["CRANE_RUN_DIR"]) if env.get("CRANE_RUN_DIR") else None
        self.inputs = Path(env.get("CRANE_INPUTS_DIR") or "/workspace/in")
        self.job_id = env.get("CRANE_JOB_ID")
        self.stop_signal: str | None = None
        self.units: list[str] = []
        self.done: dict[str, dict] = {}
        self.resumed = 0
        self.failed = 0
        self.phase = "start"
        self.unit: str | None = None
        self.unit_frac = 0.0
        self.t0 = time.time()
        self.done_at_start = 0
        self._lock = threading.Lock()
        self._last_written = 0.0
        self._last_rc = 0.0
        self._last_rc_done = -1
        self._hb_stop = threading.Event()

    # -- identity of the work, so a resume never mixes outputs of different args
    def fingerprint(self) -> str:
        key = {"task": self.task.name, "shard": self.args.shard, "n_shards": self.args.n_shards,
               "seed": self.args.seed, "task_argv": self.task_argv}
        return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]

    # -- progress
    def progress_doc(self) -> dict:
        n, total = len(self.done), len(self.units)
        el = max(time.time() - self.t0, 1e-9)
        new = n - self.done_at_start + self.unit_frac
        rate = new / el if new > 0 else 0.0
        eta = (total - n - self.unit_frac) / rate if rate > 0 else None
        return {"done": n, "total": total, "failed": self.failed, "rate_per_s": round(rate, 6),
                "eta_s": None if eta is None else round(eta, 1), "phase": self.phase, "unit": self.unit,
                "unit_frac": round(self.unit_frac, 4), "task": self.task.name, "shard": self.args.shard,
                "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def write_progress(self) -> None:
        with self._lock:
            doc = self.progress_doc()
            for d in filter(None, (self.run_dir, self.out)):
                try:
                    atomic_json(d / "progress.json", doc)
                except OSError:
                    pass
            self._last_written = time.time()
            if doc["done"] != self._last_rc_done or time.time() - self._last_rc >= RC_LINE_S:
                print("RC-PROGRESS " + json.dumps({"done": doc["done"], "total": doc["total"]}), flush=True)
                self._last_rc, self._last_rc_done = time.time(), doc["done"]

    def maybe_progress(self) -> None:
        if time.time() - self._last_written >= 5.0:
            self.write_progress()

    def _heartbeat(self) -> None:
        while not self._hb_stop.wait(HEARTBEAT_S / 3):
            if time.time() - self._last_written >= HEARTBEAT_S:
                try:
                    self.write_progress()
                except Exception:      # never kill the job from the heartbeat
                    pass

    # -- signals
    def _on_signal(self, signum, _frame):
        name = signal.Signals(signum).name
        if self.stop_signal is None:
            self.stop_signal = name
            self.phase = f"stopping ({name})"
            print(f"cfcap.runner: {name}: checkpoint at the next safe point (grace {grace_s(name):.0f} s)",
                  flush=True)
            if hasattr(signal, "setitimer"):
                signal.setitimer(signal.ITIMER_REAL, grace_s(name))
        else:
            self._hard_stop()

    def _on_alarm(self, _signum, _frame):
        self._hard_stop()

    def _hard_stop(self):
        if _GUARD.depth:
            _GUARD.pending = True
            return
        raise Stop("grace expired")

    def install_signals(self):
        self._prev = {}
        for s in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGUSR1", None), getattr(signal, "SIGALRM", None)):
            if s is not None:
                self._prev[s] = signal.getsignal(s)
        for s in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGUSR1", None)):
            if s is not None:
                signal.signal(s, self._on_signal)
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, self._on_alarm)

    # -- manifest
    def manifest(self, status: str) -> dict:
        files = []
        for u in self.units:
            r = self.done.get(u)
            if r:
                files.append({"unit": u, "path": f"units/{u}.npz" if r.get("rows") is not None else None,
                              "rows": r.get("rows"), "sha256": r.get("sha256"), "columns": r.get("columns"),
                              "bytes": r.get("bytes")})
        return {"schema": SCHEMA, "task": self.task.name, "shard": self.args.shard, "n_shards": self.args.n_shards,
                "seed": self.args.seed, "task_argv": self.task_argv, "fingerprint": self.fingerprint(),
                "status": status, "units_total": len(self.units), "units_done": len(self.done),
                "rows": sum(int(r.get("rows") or 0) for r in self.done.values()), "files": files,
                "job_id": self.job_id, "host": socket.gethostname(), "code": {"cfcap": _git_version(REPO)},
                "stop_signal": self.stop_signal, "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def write_manifest(self, status: str) -> None:
        m = self.manifest(status)
        atomic_json(self.out / "manifest.json", m)
        if self.run_dir and self.run_dir.resolve() not in self.out.resolve().parents:
            atomic_json(self.run_dir / "artifacts" / "manifest.json", m)   # a small record on the coordinator

    # -- resume
    def scan_done(self) -> None:
        for u in self.units:
            rec = _read_json(self.out / "units" / f"{u}.json")
            if not rec:
                continue
            npz = self.out / "units" / f"{u}.npz"
            if rec.get("rows") is not None and (not npz.is_file() or sha256_file(npz) != rec.get("sha256")):
                continue                # a torn or replaced file: redo the unit
            self.done[u] = rec
        self.resumed = self.done_at_start = len(self.done)

    # -- one unit
    def run_unit(self, u: str) -> None:
        import numpy as np
        self.unit, self.unit_frac, self.phase = u, 0.0, "unit"
        self.write_progress()
        ctx = UnitCtx(self, u, unit_seed(self.task.name, u, self.args.seed), self.args)
        t = time.time()
        cols = self.task.run_unit(u, ctx)
        rec: dict[str, Any] = {"unit": u, "wall_s": round(time.time() - t, 3), "seed": ctx.seed,
                               "meta": clean_meta(ctx.meta)}
        if cols:
            arrs = {str(k): np.asarray(v) for k, v in dict(cols).items()}
            lens = {a.shape[0] if a.ndim else 1 for a in arrs.values()}
            if len(lens) != 1:
                raise ValueError(f"unit {u}: columns of unequal length {sorted(lens)}")
            buf = io.BytesIO()
            (np.savez_compressed if self.args.compress else np.savez)(buf, **arrs)
            data = buf.getvalue()
            npz = self.out / "units" / f"{u}.npz"
            atomic_bytes(npz, data)
            rec.update(rows=int(lens.pop()), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                       columns={k: [str(a.dtype), list(a.shape[1:])] for k, a in arrs.items()})
        else:
            rec.update(rows=None, sha256=None, bytes=0, columns={})
        atomic_json(self.out / "units" / f"{u}.json", rec)     # the commit point of the unit
        with contextlib.suppress(OSError):
            ctx.state_path.unlink()
        self.done[u] = rec
        self.unit, self.unit_frac = None, 0.0
        self.write_manifest("running")
        self.write_progress()

    def metrics(self, status: str) -> dict[str, float]:
        recs = [self.done[u] for u in self.units if u in self.done]
        wall = time.time() - self.t0
        new = len(self.done) - self.resumed
        base = {"units_total": len(self.units), "units_done": len(self.done), "units_resumed": self.resumed,
                "units_new": new, "rows": sum(int(r.get("rows") or 0) for r in recs),
                "bytes": sum(int(r.get("bytes") or 0) for r in recs), "wall_s": round(wall, 3),
                "unit_wall_s_mean": (sum(r.get("wall_s", 0) for r in recs) / len(recs)) if recs else 0.0,
                "units_per_h": new / wall * 3600 if wall > 0 else 0.0, "interrupted": float(status != "complete"),
                "shard": self.args.shard, "n_shards": self.args.n_shards}
        extra = {}
        if self.task.summarize and recs:
            try:
                extra = self.task.summarize(recs, self.args) or {}
            except Exception as e:    # a summary bug must not lose the outputs
                print(f"cfcap.runner: summarize failed: {e!r}", file=sys.stderr)
                base["summarize_failed"] = 1.0
        return clean_metrics({**base, **{k: v for k, v in extra.items() if k not in base}})

    def write_metrics(self, status: str) -> None:
        d = self.run_dir or self.out
        atomic_json(d / "metrics.json", self.metrics(status))

    # -- the whole shard
    def run(self) -> int:
        self.out.mkdir(parents=True, exist_ok=True)
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        prev = _read_json(self.out / "manifest.json")
        if prev and prev.get("fingerprint") != self.fingerprint() and not self.args.fresh:
            print(f"cfcap.runner: {self.out} holds a different run ({prev.get('task')} shard {prev.get('shard')}, "
                  f"args {prev.get('task_argv')}); use another --out or --fresh", file=sys.stderr)
            return EXIT_USAGE
        if self.args.fresh:
            for sub in ("units", "state"):
                for p in (self.out / sub).glob("*") if (self.out / sub).is_dir() else []:
                    p.unlink()
        self.units = [str(u) for u in self.task.units(self.args, self.args.shard, self.args.n_shards)]
        bad = [u for u in self.units if not UNIT_ID.match(u)]
        if bad or len(set(self.units)) != len(self.units):
            print(f"cfcap.runner: bad or duplicate unit ids ({bad[:3]})", file=sys.stderr)
            return EXIT_USAGE
        self.scan_done()
        self.install_signals()
        hb = threading.Thread(target=self._heartbeat, daemon=True)
        hb.start()
        status, code = "running", EXIT_OK
        try:
            self.write_manifest(status)
            self.write_progress()
            for u in self.units:
                if u in self.done:
                    continue
                if self.stop_signal:
                    raise Stop(self.stop_signal)
                self.run_unit(u)
                if self.args.max_units and len(self.done) - self.resumed >= self.args.max_units:
                    raise Stop("max-units")
            status, self.phase = "complete", "done"
        except Stop as e:
            status, code, self.phase = "interrupted", EXIT_YIELD, f"interrupted ({e})"
            print(f"cfcap.runner: interrupted ({e}): {len(self.done)}/{len(self.units)} units kept; exit 75",
                  flush=True)
        except InputMissing as e:
            status, code, self.phase = "failed", EXIT_INPUT, "missing input"
            print(f"cfcap.runner: {e}", file=sys.stderr)
        except Exception:
            self.failed += 1
            status, code, self.phase = "failed", EXIT_TASK, "failed"
            traceback.print_exc()
        finally:
            if hasattr(signal, "setitimer"):
                signal.setitimer(signal.ITIMER_REAL, 0)
            self._hb_stop.set()
            self.unit, self.unit_frac = None, 0.0
            # The epilogue must finish even if a late signal arrives.
            for s in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGUSR1", None)):
                if s is not None:
                    signal.signal(s, signal.SIG_IGN)
            self.write_manifest(status)
            self.write_progress()
            self.write_metrics(status)
            for s, h in getattr(self, "_prev", {}).items():
                signal.signal(s, h)
        return code


def clean_meta(meta: dict) -> dict:
    try:
        json.dumps(meta)
        return meta
    except TypeError:
        return {k: (v if isinstance(v, (int, float, str, bool, type(None))) else repr(v)) for k, v in meta.items()}


# ------------------------------------------------------------------ the selftest task

def _st_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--units", type=int, default=6, help="units per shard")
    p.add_argument("--rows", type=int, default=1000, help="rows per unit")
    p.add_argument("--unit-s", type=float, default=2.0, help="wall seconds per unit (sleep, in steps)")
    p.add_argument("--steps", type=int, default=10, help="checkpointed steps per unit")
    p.add_argument("--fail-unit", default=None, help="raise in this unit (tests the failure path)")


def _st_units(args, shard, n_shards):
    return [f"st-s{shard:03d}-u{i:03d}" for i in range(args.units)]


def _st_run(unit: str, ctx: UnitCtx):
    import numpy as np
    if ctx.args.fail_unit == unit:
        raise RuntimeError(f"selftest: requested failure in {unit}")
    st = ctx.load_state() or {"step": 0, "acc": 0.0}
    ctx.meta["resumed_at_step"] = st["step"]
    rng = np.random.default_rng(ctx.seed)
    x = rng.standard_normal(ctx.args.rows)
    for k in range(st["step"], ctx.args.steps):
        if ctx.should_stop():
            ctx.save_state(st)
            raise Stop("selftest checkpoint")
        time.sleep(ctx.args.unit_s / ctx.args.steps)
        st = {"step": k + 1, "acc": st["acc"] + float(np.sum(np.sin(x * (k + 1))))}
        ctx.save_state(st)
        ctx.progress((k + 1) / ctx.args.steps)
    ctx.meta["acc"] = st["acc"]
    return {"x": x.astype(np.float32), "y": (x ** 2).astype(np.float32),
            "i": np.arange(ctx.args.rows, dtype=np.int32)}


def _st_summary(recs, args):
    return {"selftest.acc_sum": sum(r["meta"].get("acc", 0.0) for r in recs),
            "selftest.resumed_mid_unit": sum(1 for r in recs if r["meta"].get("resumed_at_step", 0) > 0)}


SELFTEST = Task("selftest", _st_units, _st_run, _st_args, _st_summary, ("numpy",),
                "deterministic rows with per-step unit checkpoints; exercises resume, signals and outputs")


# ------------------------------------------------------------------ entry

def load_task(name: str, spec: str | None = None) -> Task:
    ref = spec or TASKS.get(name)
    if not ref:
        raise KeyError(f"unknown task {name!r}; registered: {', '.join(sorted(TASKS))} (or --task-spec mod:attr)")
    mod, _, attr = ref.partition(":")
    try:
        m = importlib.import_module(mod)
    except ModuleNotFoundError as e:
        if e.name and (mod == e.name or mod.startswith(e.name + ".")):
            who = OWNER.get(name)
            raise KeyError(f"task {name!r}: {mod} does not exist yet" + (f" (role {who} provides {ref})" if who else ""))
        raise
    t = getattr(m, attr or "TASK")
    if not isinstance(t, Task):
        raise KeyError(f"{ref} is not a cfcap.runner.Task")
    return t


def check_requires(mods: Iterable[str]) -> list[str]:
    missing = []
    for mod in mods:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    return missing


def build_parser(task: Task | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m cfcap.runner", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--seed", type=int, default=0, help="base seed (unit seeds derive from task, unit and this)")
    p.add_argument("--out", default=None, help="output dir (default: see module doc)")
    p.add_argument("--task-spec", default=None, help="module:attr of a Task not in the registry")
    p.add_argument("--compress", action="store_true", help="np.savez_compressed for the unit files")
    p.add_argument("--fresh", action="store_true", help="discard outputs of another run in --out")
    p.add_argument("--max-units", type=int, default=0, help="stop (exit 75) after this many new units (tests)")
    p.add_argument("--list", action="store_true", help="print the shard's unit ids and exit")
    if task and task.add_args:
        g = p.add_argument_group(f"task {task.name}")
        task.add_args(g)
    return p


def _task_argv(argv: list[str], task: Task) -> list[str]:
    """The task's own flags as given (the resume fingerprint), in order."""
    runner_flags = {"--shard", "--n-shards", "--seed", "--out", "--task-spec", "--max-units"}
    runner_bools = {"--compress", "--fresh", "--list"}
    out, it = [], iter(argv[1:])
    for a in it:
        k = a.split("=", 1)[0]
        if k in runner_flags:
            if "=" not in a:
                next(it, None)
        elif k in runner_bools:
            continue
        else:
            out.append(a)
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("task", nargs="?")
    pre.add_argument("--task-spec", default=None)
    known, _ = pre.parse_known_args(argv)
    if not known.task or known.task.startswith("-"):
        build_parser().print_help()
        print("\nregistered tasks: " + ", ".join(sorted(TASKS)))
        return EXIT_USAGE
    try:
        task = load_task(known.task, known.task_spec)
    except KeyError as e:
        print(f"cfcap.runner: {e.args[0]}", file=sys.stderr)
        return EXIT_USAGE
    args = build_parser(task).parse_args(argv)
    if not (0 <= args.shard < args.n_shards):
        print(f"cfcap.runner: --shard {args.shard} not in [0, {args.n_shards})", file=sys.stderr)
        return EXIT_USAGE
    missing = check_requires(task.requires)
    if missing:
        print(f"cfcap.runner: task {task.name} needs {', '.join(missing)} (python {sys.executable})", file=sys.stderr)
        return EXIT_INPUT
    r = Runner(task, args, _task_argv(argv, task))
    if args.list:
        print("\n".join(task.units(args, args.shard, args.n_shards)))
        return EXIT_OK
    print(f"cfcap.runner: {task.name} shard {args.shard}/{args.n_shards} -> {r.out}  ({shlex.join(argv)})", flush=True)
    return r.run()


if __name__ == "__main__":
    # Run through the importable module, so tasks registered against cfcap.runner.Task (and SELFTEST) are the same
    # class objects as the ones main() checks.
    from cfcap import runner as _canonical
    sys.exit(_canonical.main())
