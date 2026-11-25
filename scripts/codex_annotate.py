"""Batch annotation through codex (codex-lb): chunked, parallel, schema-validated, resumable.

    $PY scripts/codex_annotate.py run --name NAME --cases cases.jsonl --prompt tmpl.md --schema schema.json \
        [--chunk 25] [--parallel 6] [--max-parallel 12] [--retries 3] [--var KEY=VALUE | KEY=@file] ...

Input cases: a JSONL or CSV file; every case has a unique id (`--id-field`, default `case_id`).
Prompt template: plain text with placeholders {{CASES}} (the chunk, one JSON object per line), {{N}} (chunk size),
{{IDS}} (the chunk's ids) and {{KEY}} for every --var. The rules file scripts/codex_prompts/_RULES.md is NOT
prepended: annotation calls run read-only in an empty scratch directory and only return JSON.
Schema: a JSON Schema for codex's final message (`codex exec --output-schema`). Its top level must be
{"items": [...]} and every item must carry the id field. Each answer is validated here against the same schema
(a strict-mode subset: type, properties, required, additionalProperties, items, enum, minimum, maximum,
minItems, maxItems, anyOf); the item ids must be exactly the chunk's ids (missing ids are re-queued, extra or
duplicate ids are dropped).

Outputs:
  results/annotations/<name>.jsonl      one validated record per case: the item plus _chunk, _attempt, _t
  results/annotations/<name>_calls.csv  one row per codex call: chunk, attempt, parallel limit, seconds, status,
                                        tokens (input, cached, output, reasoning), cases, valid items
  $CFCAP_OUT/annotations/<name>/        raw prompts, answers and codex event logs per call
Resumable: cases whose id is already in <name>.jsonl are skipped. Re-running is safe.

Concurrency is adaptive (AIMD): start at --parallel, add one slot after every --raise-after consecutive successful
calls (up to --max-parallel), halve it and sleep with exponential backoff after a failed call (non-zero exit,
timeout, empty or invalid JSON). A chunk is retried up to --retries times; after that its cases are logged as
failed in the calls CSV and left for the next run. A call that fails because codex-lb is out of quota ("usage
limit", "account is unavailable", 429) does not use up a retry: dispatch pauses until the advertised reset time
(else 15 min) and resumes, for at most --max-quota-wait-h hours in total. Calls run under nice 15 and use no GPU.

The codex binary is `$CODEX_BIN` if set (the tests use a stub), else `codex`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results" / "annotations"
OUT = Path(os.environ.get("CFCAP_OUT", Path.home() / ".cache/modular-runs/2026-11/cfcap")) / "annotations"
CALL_FIELDS = ["name", "chunk", "attempt", "limit", "t_start", "seconds", "status", "n_cases", "n_valid",
               "input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens", "error"]


# ----------------------------------------------------------------------------- schema validation (strict subset)
def schema_errors(x, s, path="$"):
    """Errors of value x against JSON Schema s (subset used by codex strict structured outputs)."""
    errs = []
    if "anyOf" in s:
        subs = [schema_errors(x, ss, path) for ss in s["anyOf"]]
        if all(subs):
            errs.append(f"{path}: matches no anyOf branch ({subs[0][0]})")
        return errs
    t = s.get("type")
    types = t if isinstance(t, list) else [t] if t else []
    ok_type = {"object": lambda v: isinstance(v, dict), "array": lambda v: isinstance(v, list),
               "string": lambda v: isinstance(v, str), "boolean": lambda v: isinstance(v, bool),
               "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
               "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
               "null": lambda v: v is None}
    if types and not any(ok_type[tt](x) for tt in types):
        return [f"{path}: expected {types}, got {type(x).__name__}"]
    if "enum" in s and x not in s["enum"]:
        errs.append(f"{path}: {x!r} not in enum")
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        if "minimum" in s and x < s["minimum"]:
            errs.append(f"{path}: {x} < minimum {s['minimum']}")
        if "maximum" in s and x > s["maximum"]:
            errs.append(f"{path}: {x} > maximum {s['maximum']}")
    if isinstance(x, dict):
        props = s.get("properties", {})
        for k in s.get("required", []):
            if k not in x:
                errs.append(f"{path}: missing {k}")
        if s.get("additionalProperties") is False:
            for k in x:
                if k not in props:
                    errs.append(f"{path}: unexpected {k}")
        for k, v in x.items():
            if k in props:
                errs += schema_errors(v, props[k], f"{path}.{k}")
    if isinstance(x, list):
        if "minItems" in s and len(x) < s["minItems"]:
            errs.append(f"{path}: {len(x)} < minItems {s['minItems']}")
        if "maxItems" in s and len(x) > s["maxItems"]:
            errs.append(f"{path}: {len(x)} > maxItems {s['maxItems']}")
        if "items" in s:
            for i, v in enumerate(x):
                errs += schema_errors(v, s["items"], f"{path}[{i}]")
    return errs


def item_schema(schema):
    return schema["properties"]["items"]["items"]


# ----------------------------------------------------------------------------- io
def load_cases(path, id_field="case_id"):
    path = Path(path)
    if path.suffix == ".csv":
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
    else:
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    ids = [str(r[id_field]) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate {id_field} in {path}")
    return rows


def done_ids(out_jsonl, id_field="case_id"):
    if not Path(out_jsonl).exists():
        return set()
    out = set()
    for l in Path(out_jsonl).read_text().splitlines():
        if l.strip():
            try:
                out.add(str(json.loads(l)[id_field]))
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def render(template, chunk, id_field, vars_):
    txt = template
    for k, v in vars_.items():
        txt = txt.replace("{{" + k + "}}", v)
    txt = txt.replace("{{N}}", str(len(chunk)))
    txt = txt.replace("{{IDS}}", json.dumps([str(c[id_field]) for c in chunk]))
    return txt.replace("{{CASES}}", "\n".join(json.dumps(c, separators=(",", ":"), default=str) for c in chunk))


def parse_usage(events_text):
    u = dict(input_tokens=0, cached_input_tokens=0, output_tokens=0, reasoning_output_tokens=0)
    for l in events_text.splitlines():
        try:
            e = json.loads(l)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "turn.completed":
            for k in u:
                u[k] += int(e.get("usage", {}).get(k, 0) or 0)
    return u


# ----------------------------------------------------------------------------- one call
def call_codex(prompt, schema_path, answer_path, events_path, workdir, timeout, model=None):
    """Run one `codex exec`; returns (rc, seconds, events text, error)."""
    exe = shlex.split(os.environ.get("CODEX_BIN", "codex"))
    cmd = exe + ["exec", "-s", "read-only", "--skip-git-repo-check", "--ephemeral", "-C", str(workdir),
                 "--output-schema", str(schema_path), "-o", str(answer_path), "--json", "--color", "never"]
    if model:
        cmd += ["-m", model]
    cmd.append("-")
    t0 = time.time()
    try:
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout,
                           preexec_fn=lambda: os.nice(15))
        Path(events_path).write_text(p.stdout + "\n--- stderr ---\n" + p.stderr[-20000:])
        err = "" if p.returncode == 0 else (p.stderr.strip().splitlines() or ["rc!=0"])[-1][:300]
        return p.returncode, time.time() - t0, p.stdout, err
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        Path(events_path).write_text(out)
        return -9, time.time() - t0, out, f"timeout {timeout}s"


QUOTA_PAT = re.compile(r"usage limit|account is unavailable|rate.?limit|\b429\b|quota", re.I)
RESET_PAT = re.compile(r"try again at (\d{1,2}):(\d{2})\s*([AP]M)?", re.I)


def quota_reset(text, now=None):
    """If the codex events say the account is out of quota, return the epoch time to resume (the advertised reset,
    else now + 15 min); else None."""
    if not text or not QUOTA_PAT.search(text):
        return None
    now = now or datetime.now()
    m = RESET_PAT.search(text)
    if not m:
        return (now + timedelta(minutes=15)).timestamp()
    h, mi, ap = int(m.group(1)), int(m.group(2)), (m.group(3) or "").upper()
    if ap == "PM" and h < 12:
        h += 12
    if ap == "AM" and h == 12:
        h = 0
    t = now.replace(hour=h, minute=mi, second=0, microsecond=0)
    if t <= now:
        t += timedelta(days=1)
    return (t + timedelta(minutes=2)).timestamp()


def check_answer(answer_path, schema, ids, id_field):
    """-> (valid items by id, error string). Items failing the item schema are dropped (and re-queued)."""
    try:
        ans = json.loads(Path(answer_path).read_text())
    except (OSError, json.JSONDecodeError) as e:
        return {}, f"bad json: {e}"[:300]
    top = schema_errors(ans, {k: v for k, v in schema.items() if k != "properties"} | {
        "properties": {"items": {"type": "array"}}})
    if top or not isinstance(ans.get("items"), list):
        return {}, f"bad top level: {top[:2]}"
    isch = item_schema(schema)
    want = set(ids)
    good, bad = {}, []
    for it in ans["items"]:
        e = schema_errors(it, isch)
        cid = str(it.get(id_field)) if isinstance(it, dict) else None
        if e or cid not in want or cid in good:
            bad.append(e[:1] if e else [f"id {cid}"])
            continue
        good[cid] = it
    err = "" if not bad and len(good) == len(want) else f"{len(want) - len(good)} missing/invalid; {bad[:2]}"
    return good, err[:300]


# ----------------------------------------------------------------------------- scheduler
class Limiter:
    """AIMD concurrency limit."""

    def __init__(self, start, hi, raise_after=4, backoff0=5.0, backoff_max=120.0):
        self.limit, self.hi, self.raise_after = start, hi, raise_after
        self.streak, self.backoff0, self.backoff_max, self.fails = 0, backoff0, backoff_max, 0
        self.lock = threading.Lock()

    def ok(self):
        with self.lock:
            self.fails = 0
            self.streak += 1
            if self.streak >= self.raise_after and self.limit < self.hi:
                self.limit += 1
                self.streak = 0

    def fail(self):
        with self.lock:
            self.streak = 0
            self.fails += 1
            self.limit = max(1, self.limit // 2)
            return min(self.backoff_max, self.backoff0 * 2 ** (self.fails - 1))


def _print(*a):
    print(*a, flush=True)


def run(name, cases, template, schema_path, chunk=25, parallel=6, max_parallel=12, retries=3, id_field="case_id",
        vars_=None, timeout=1200, model=None, raise_after=4, res_dir=RES, out_dir=OUT, backoff0=5.0, log=_print,
        max_quota_wait_h=8.0):
    res_dir, out_dir = Path(res_dir), Path(out_dir) / name
    res_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "work"                                   # empty scratch dir: codex sees nothing of the repo
    work.mkdir(exist_ok=True)
    out_jsonl, calls_csv = res_dir / f"{name}.jsonl", res_dir / f"{name}_calls.csv"
    schema = json.loads(Path(schema_path).read_text())
    have = done_ids(out_jsonl, id_field)
    todo = [c for c in cases if str(c[id_field]) not in have]
    log(f"[{name}] {len(cases)} cases, {len(have)} done, {len(todo)} to annotate")
    if not todo:
        return dict(n_cases=len(cases), n_done=len(have), n_failed=0, calls=0)
    by_id = {str(c[id_field]): c for c in todo}
    start = 0
    if calls_csv.exists():
        with open(calls_csv) as fh:
            start = 1 + max((int(r["chunk"]) for r in csv.DictReader(fh)), default=-1)
    queue = [(start + i, [str(c[id_field]) for c in todo[j:j + chunk]], 1)
             for i, j in enumerate(range(0, len(todo), chunk))]
    next_chunk = start + len(queue)
    lim = Limiter(parallel, max(parallel, max_parallel), raise_after, backoff0)
    wlock = threading.Lock()
    new_calls = not calls_csv.exists() or calls_csv.stat().st_size == 0
    fcalls = open(calls_csv, "a", newline="")
    wcalls = csv.DictWriter(fcalls, fieldnames=CALL_FIELDS)
    if new_calls:
        wcalls.writeheader()
    fout = open(out_jsonl, "a")
    failed, n_calls, pause_until, quota_waited = [], 0, [0.0], [0.0]

    def job(cid, ids, attempt, limit):
        tag = f"chunk{cid:05d}_a{attempt}"
        prompt = render(template, [by_id[i] for i in ids], id_field, vars_ or {})
        (out_dir / f"{tag}.prompt.txt").write_text(prompt)
        ans = out_dir / f"{tag}.json"
        t_start = time.time()
        rc, secs, ev, err = call_codex(prompt, schema_path, ans, out_dir / f"{tag}.events.jsonl", work, timeout,
                                       model)
        good, verr = check_answer(ans, schema, ids, id_field) if rc == 0 else ({}, "")
        u = parse_usage(ev)
        status = "ok" if rc == 0 and not verr else ("partial" if good else "fail")
        qr = quota_reset(ev) if not good else None
        if qr is not None:
            status = "quota"
            err = f"quota; resume after {datetime.fromtimestamp(qr):%H:%M}"
        with wlock:
            for i in ids:
                if i in good:
                    rec = dict(good[i])
                    rec.update(_chunk=cid, _attempt=attempt, _t=round(t_start))
                    fout.write(json.dumps(rec) + "\n")
            fout.flush()
            wcalls.writerow(dict(name=name, chunk=cid, attempt=attempt, limit=limit, t_start=round(t_start, 1),
                                 seconds=round(secs, 1), status=status, n_cases=len(ids), n_valid=len(good),
                                 error=(err or verr)[:300], **u))
            fcalls.flush()
        return [i for i in ids if i not in good], status, rc, qr

    try:
        with ThreadPoolExecutor(max_workers=max(parallel, max_parallel)) as ex:
            running = {}
            while queue or running:
                while queue and len(running) < lim.limit and time.time() >= pause_until[0]:
                    cid, ids, att = queue.pop(0)
                    running[ex.submit(job, cid, ids, att, lim.limit)] = (cid, ids, att)
                    n_calls += 1
                if not running:
                    time.sleep(max(0.05, min(1.0, pause_until[0] - time.time())))
                    continue
                fin, _ = wait(list(running), timeout=1.0, return_when=FIRST_COMPLETED)
                for f in fin:
                    cid, ids, att = running.pop(f)
                    try:
                        miss, status, rc, qr = f.result()
                    except Exception as e:                          # noqa: BLE001  (logged, chunk retried)
                        miss, status, rc, qr = ids, f"error {e!r}"[:80], -1, None
                    if status == "quota":                           # not the chunk's fault: same attempt, wait
                        quota_waited[0] += max(0.0, qr - max(time.time(), pause_until[0]))
                        pause_until[0] = max(pause_until[0], qr)
                        if quota_waited[0] > max_quota_wait_h * 3600:
                            failed += miss
                        else:
                            queue.insert(0, (next_chunk, miss, att))
                            next_chunk += 1
                    elif status == "ok":
                        lim.ok()
                    else:
                        if rc != 0 or not miss or len(miss) == len(ids):
                            pause_until[0] = time.time() + lim.fail()
                        if miss:
                            if att < retries:
                                queue.append((next_chunk, miss, att + 1))
                                next_chunk += 1
                            else:
                                failed += miss
                    log(f"[{name}] chunk {cid} a{att}: {status}, {len(ids) - len(miss)}/{len(ids)} valid, "
                        f"limit {lim.limit}, queue {len(queue)}, running {len(running)}")
    finally:
        fout.close()
        fcalls.close()
    n_done = len(done_ids(out_jsonl, id_field))
    log(f"[{name}] done: {n_done}/{len(cases)} annotated, {len(failed)} failed after {retries} attempts, "
        f"{n_calls} calls")
    return dict(n_cases=len(cases), n_done=n_done, n_failed=len(failed), calls=n_calls)


def summarize_calls(calls_csv):
    """Throughput / token totals of a calls CSV -> dict."""
    with open(calls_csv) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    t0 = min(float(r["t_start"]) for r in rows)
    t1 = max(float(r["t_start"]) + float(r["seconds"]) for r in rows)
    tok = {k: sum(int(r[k] or 0) for r in rows) for k in CALL_FIELDS[9:13]}
    nv = sum(int(r["n_valid"]) for r in rows)
    return dict(calls=len(rows), ok=sum(r["status"] == "ok" for r in rows),
                failed=sum(r["status"] == "fail" for r in rows), valid_items=nv, wall_s=round(t1 - t0, 1),
                items_per_min=round(60 * nv / max(t1 - t0, 1e-9), 1),
                median_call_s=sorted(float(r["seconds"]) for r in rows)[len(rows) // 2],
                max_limit=max(int(r["limit"]) for r in rows), **tok)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--cases", required=True)
    r.add_argument("--prompt", required=True)
    r.add_argument("--schema", required=True)
    r.add_argument("--id-field", default="case_id")
    r.add_argument("--chunk", type=int, default=25)
    r.add_argument("--parallel", type=int, default=6)
    r.add_argument("--max-parallel", type=int, default=12)
    r.add_argument("--raise-after", type=int, default=4)
    r.add_argument("--retries", type=int, default=3)
    r.add_argument("--timeout", type=int, default=1200)
    r.add_argument("--limit", type=int, default=0, help="annotate at most this many cases (pilots)")
    r.add_argument("--model", default=None)
    r.add_argument("--var", action="append", default=[], help="KEY=VALUE or KEY=@file")
    r.add_argument("--max-quota-wait-h", type=float, default=8.0,
                   help="total hours to wait for codex-lb quota resets before giving up")
    s = sub.add_parser("summary")
    s.add_argument("names", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "summary":
        for n in a.names:
            print(n, json.dumps(summarize_calls(RES / f"{n}_calls.csv")))
        return 0
    vars_ = {}
    for kv in a.var:
        k, v = kv.split("=", 1)
        vars_[k] = Path(v[1:]).read_text() if v.startswith("@") else v
    cases = load_cases(a.cases, a.id_field)
    if a.limit:
        cases = cases[:a.limit]
    out = run(a.name, cases, Path(a.prompt).read_text(), a.schema, a.chunk, a.parallel, a.max_parallel, a.retries,
              a.id_field, vars_, a.timeout, a.model, a.raise_after, max_quota_wait_h=a.max_quota_wait_h)
    print(json.dumps(out))
    return 0 if out["n_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
