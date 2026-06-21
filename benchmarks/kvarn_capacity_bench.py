#!/usr/bin/env python3
"""KVarN capacity/throughput grid benchmark (gfx906).

Sweeps concurrency x context against a running vLLM OpenAI-compatible server and
records aggregate decode throughput (tok/s) per cell, or ``OOM`` when the cell
cannot run (KV working set doesn't fit / engine dies / requests fail). Run it
once per KV-cache dtype into a shared results JSON, then ``--render`` the
combined cross-reference table (the one that goes in the README).

Why a custom harness instead of ``vllm bench serve``:
- exact context length via integer token-id prompts (no tokenizer dependency);
- prompts are randomized per request so prefix caching can't inflate numbers;
- a failed/declined cell is recorded as OOM rather than aborting the sweep;
- stdlib only, so it runs on the host against the mapped port.

Typical use (server already up on :8000 serving model name ``qwen3-4b``):

    # arm 1 — fp16 KV
    python3 kvarn_capacity_bench.py run --label fp16 --model qwen3-4b \
        --concurrency 8,16,32 --context 8192,16384,32768 --out grid.json
    # (restart server with kvarn_k4v2_g128, then)
    # arm 2 — KVarN
    python3 kvarn_capacity_bench.py run --label kvarn --model qwen3-4b \
        --concurrency 8,16,32 --context 8192,16384,32768 --out grid.json
    # combined table
    python3 kvarn_capacity_bench.py render --out grid.json --md table.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# Token-id range used to synthesize prompts. Avoid low ids (often special
# tokens) and stay well under typical vocab sizes (Qwen3 ~151k, Llama ~128k).
_TOK_LO, _TOK_HI = 1000, 90000


def _post(url: str, body: dict, timeout: float) -> tuple[int, dict | None, str]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode()), ""
    except urllib.error.HTTPError as e:
        return e.code, None, f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}"
    except Exception as e:  # connection reset = engine died, timeout, etc.
        return 0, None, f"{type(e).__name__}: {e}"


def _health_ok(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url + "/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def _make_prompt(rng: random.Random, n_tokens: int) -> list[int]:
    return [rng.randint(_TOK_LO, _TOK_HI) for _ in range(n_tokens)]


def run_cell(
    base_url: str,
    model: str,
    concurrency: int,
    input_len: int,
    output_len: int,
    num_prompts: int,
    req_timeout: float,
    seed: int,
) -> dict:
    """Drive one (concurrency, context) cell. Returns metrics or an OOM marker."""
    url = base_url + "/v1/completions"
    rng = random.Random(seed)
    # Distinct random prompt per request -> no shared prefix / cache hits.
    prompts = [_make_prompt(rng, input_len) for _ in range(num_prompts)]

    # Warmup one request of this shape (triggers any per-shape JIT) before timing.
    s, _, err = _post(
        url,
        {"model": model, "prompt": prompts[0], "max_tokens": 8,
         "temperature": 0.0, "ignore_eos": True, "stream": False},
        req_timeout,
    )
    if s != 200:
        return {"status": "OOM", "reason": f"warmup failed: {err or s}"}

    results: list[dict] = []
    lock = threading.Lock()
    start_barrier = threading.Event()

    def one(idx: int) -> None:
        start_barrier.wait()
        t0 = time.monotonic()
        st, resp, err = _post(
            url,
            {"model": model, "prompt": prompts[idx], "max_tokens": output_len,
             "temperature": 0.0, "ignore_eos": True, "stream": False},
            req_timeout,
        )
        t1 = time.monotonic()
        out_toks = 0
        if st == 200 and resp:
            out_toks = (resp.get("usage") or {}).get("completion_tokens", 0)
        with lock:
            results.append(
                {"ok": st == 200 and out_toks > 0, "status": st,
                 "out": out_toks, "lat": t1 - t0, "err": err}
            )

    t_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(one, i) for i in range(num_prompts)]
        start_barrier.set()
        for _ in as_completed(futs):
            pass
    t_end = time.monotonic()

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    # A cell is OOM/infeasible if a meaningful share of requests failed (engine
    # death, 5xx, declined). One stray failure shouldn't poison a good cell.
    if len(ok) == 0 or len(failed) > max(1, num_prompts // 10):
        sample = next((r["err"] for r in failed if r["err"]), f"{len(failed)} failed")
        return {"status": "OOM", "reason": sample,
                "ok": len(ok), "failed": len(failed)}
    if not _health_ok(base_url):
        return {"status": "OOM", "reason": "server unhealthy after cell"}

    total_out = sum(r["out"] for r in ok)
    elapsed = t_end - t_start
    lats = sorted(r["lat"] for r in ok)
    return {
        "status": "ok",
        "tok_s": round(total_out / elapsed, 1),       # aggregate output tg/s
        "elapsed_s": round(elapsed, 1),
        "completed": len(ok),
        "failed": len(failed),
        "out_tokens": total_out,
        "p50_lat_s": round(lats[len(lats) // 2], 1),
        "p99_lat_s": round(lats[min(len(lats) - 1, int(len(lats) * 0.99))], 1),
    }


def cmd_run(a: argparse.Namespace) -> None:
    base_url = a.base_url.rstrip("/")
    concurrencies = [int(x) for x in a.concurrency.split(",")]
    contexts = [int(x) for x in a.context.split(",")]

    if not _health_ok(base_url):
        sys.exit(f"server at {base_url} is not healthy; start it first")

    try:
        with open(a.out) as f:
            store = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        store = {}
    store.setdefault("_meta", {})[a.label] = {
        "model": a.model, "max_model_len": a.max_model_len,
        "output_len": a.output_len, "waves": a.waves,
    }
    grid = store.setdefault(a.label, {})

    for L in contexts:
        for C in concurrencies:
            key = f"c{C}_l{L}"
            if a.max_model_len and L > a.max_model_len:
                grid[key] = {"status": "n/a", "reason": "L > max_model_len",
                             "concurrency": C, "context": L}
                print(f"[{a.label}] C={C:>3} L={L:>6}  n/a (L>max_model_len)")
                _persist(a.out, store)
                continue
            # Physical KV wall: C conversations each holding L tokens need C*L
            # tokens of KV. If that exceeds the measured pool, the card can't
            # hold the working set -> OOM (don't bother running the cell).
            needed = C * L
            if a.kv_pool_tokens and needed > a.kv_pool_tokens:
                grid[key] = {"status": "OOM", "concurrency": C, "context": L,
                             "reason": f"working set {needed} tok > KV pool "
                                       f"{a.kv_pool_tokens} tok"}
                print(f"[{a.label}] C={C:>3} L={L:>6}  OOM "
                      f"(needs {needed} > pool {a.kv_pool_tokens} tok)")
                _persist(a.out, store)
                continue
            input_len = max(8, L - a.output_len)
            num_prompts = max(C * a.waves, C)
            print(f"[{a.label}] C={C:>3} L={L:>6}  running "
                  f"({num_prompts} prompts, in={input_len} out={a.output_len}) ...",
                  flush=True)
            res = run_cell(base_url, a.model, C, input_len, a.output_len,
                           num_prompts, a.req_timeout, seed=1000 + C + L)
            res["concurrency"], res["context"] = C, L
            grid[key] = res
            if res["status"] == "ok":
                print(f"    -> {res['tok_s']} tok/s agg "
                      f"(p50 {res['p50_lat_s']}s, {res['completed']}/{num_prompts} ok)")
            else:
                print(f"    -> {res['status'].upper()}: {res.get('reason','')}")
            _persist(a.out, store)
            if not _health_ok(base_url):
                print("    server unhealthy; pausing 20s for recovery", flush=True)
                time.sleep(20)
    print(f"saved -> {a.out}")


def _save(path: str, store: dict) -> None:
    with open(path, "w") as f:
        json.dump(store, f, indent=2)


_CSV_FIELDS = ["label", "concurrency", "context", "status", "tok_s",
               "completed", "failed", "p50_lat_s", "p99_lat_s",
               "out_tokens", "elapsed_s", "reason"]


def _rows(store: dict):
    for label in (k for k in store if k != "_meta"):
        for cell in store[label].values():
            yield {"label": label, **{k: cell.get(k, "") for k in _CSV_FIELDS[1:]}}


def _write_csv(path: str, store: dict) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for row in _rows(store):
            w.writerow(row)


def _persist(json_path: str, store: dict) -> None:
    """Write both the JSON source of truth and a flat CSV (same basename)."""
    _save(json_path, store)
    _write_csv(os.path.splitext(json_path)[0] + ".csv", store)


def _cell_str(grid: dict, C: int, L: int) -> str:
    r = grid.get(f"c{C}_l{L}")
    if not r:
        return "-"
    if r["status"] == "ok":
        return f"{r['tok_s']:g}"
    if r["status"] == "n/a":
        return "n/a"
    return "OOM"


def _grid_axes(store: dict, labels: list[str]) -> tuple[list[int], list[int]]:
    concurrencies = sorted(
        {int(k[1:k.index("_")]) for lab in labels for k in store[lab]}
    )
    contexts = sorted(
        {int(k[k.index("l") + 1:]) for lab in labels for k in store[lab]}
    )
    return concurrencies, contexts


def _lerp_hex(c0: str, c1: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    a = [int(c0[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(a[i] + (b[i] - a[i]) * t):02x}" for i in range(3))


def render_svg(store: dict, path: str) -> None:
    """Hand-rolled SVG (no external lib) of the conc x ctx grid, one panel per
    dtype. ok cells are green (shaded by relative tg/s), OOM red, n/a grey."""
    labels = [k for k in store if k != "_meta"]
    concurrencies, contexts = _grid_axes(store, labels)
    max_tok = max(
        [c["tok_s"] for lab in labels for c in store[lab].values()
         if c.get("status") == "ok"] or [1.0]
    )

    CW, CH, LW, HH, TH, GAP, M = 104, 46, 78, 36, 30, 46, 22
    panel_w = LW + len(contexts) * CW
    panel_h = TH + HH + len(concurrencies) * CH
    total_w = M * 2 + len(labels) * panel_w + (len(labels) - 1) * GAP
    total_h = M * 2 + panel_h + 56

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" '
         f'height="{total_h}" font-family="-apple-system,Segoe UI,Roboto,sans-serif">']
    s.append(f'<rect width="{total_w}" height="{total_h}" fill="#ffffff"/>')

    def text(x, y, t, size=14, fill="#222", weight="normal", anchor="middle"):
        return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
                f'font-weight="{weight}" text-anchor="{anchor}" '
                f'dominant-baseline="middle">{t}</text>')

    for pi, label in enumerate(labels):
        ox = M + pi * (panel_w + GAP)
        oy = M
        meta = store.get("_meta", {}).get(label, {})
        s.append(text(ox + panel_w / 2, oy + TH / 2,
                      f"{label}  ({meta.get('model','')})", 16, "#111", "bold"))
        gx, gy = ox, oy + TH
        # corner + context headers
        s.append(text(gx + LW / 2, gy + HH / 2, "conc \\ ctx", 11, "#666"))
        for ci, L in enumerate(contexts):
            s.append(text(gx + LW + ci * CW + CW / 2, gy + HH / 2,
                          f"{L // 1024}k tok", 13, "#444", "bold"))
        # rows
        for ri, C in enumerate(concurrencies):
            ry = gy + HH + ri * CH
            s.append(text(gx + LW / 2, ry + CH / 2, str(C), 14, "#444", "bold"))
            for ci, L in enumerate(contexts):
                cx = gx + LW + ci * CW
                r = store[label].get(f"c{C}_l{L}", {})
                st = r.get("status")
                if st == "ok":
                    fill = _lerp_hex("#aed8b0", "#1b5e20", r["tok_s"] / max_tok)
                    val, tcol = f"{r['tok_s']:g}", "#ffffff"
                    sub = f"tok/s"
                elif st == "OOM":
                    fill, val, tcol, sub = "#e53935", "OOM", "#ffffff", ""
                elif st == "n/a":
                    fill, val, tcol, sub = "#cfcfcf", "n/a", "#555", ""
                else:
                    fill, val, tcol, sub = "#f0f0f0", "-", "#999", ""
                s.append(f'<rect x="{cx + 2}" y="{ry + 2}" width="{CW - 4}" '
                         f'height="{CH - 4}" rx="5" fill="{fill}"/>')
                if sub:
                    s.append(text(cx + CW / 2, ry + CH / 2 - 6, val, 15, tcol, "bold"))
                    s.append(text(cx + CW / 2, ry + CH / 2 + 9, sub, 9, tcol))
                else:
                    s.append(text(cx + CW / 2, ry + CH / 2, val, 14, tcol, "bold"))

    # legend
    ly = M + panel_h + 26
    s.append(f'<rect x="{M}" y="{ly}" width="16" height="16" rx="3" fill="#1b5e20"/>')
    s.append(text(M + 24, ly + 8, "ran (aggregate output tok/s; darker = faster)",
                  12, "#444", "normal", "start"))
    s.append(f'<rect x="{M + 330}" y="{ly}" width="16" height="16" rx="3" fill="#e53935"/>')
    s.append(text(M + 354, ly + 8, "OOM / KV working set won't fit",
                  12, "#444", "normal", "start"))
    s.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(s) + "\n")


def cmd_render(a: argparse.Namespace) -> None:
    with open(a.out) as f:
        store = json.load(f)
    labels = [k for k in store if k != "_meta"]
    concurrencies = sorted(
        {int(k[1:k.index("_")]) for lab in labels for k in store[lab]}
    )
    contexts = sorted(
        {int(k[k.index("l") + 1:]) for lab in labels for k in store[lab]}
    )
    lines = []
    for label in labels:
        meta = store.get("_meta", {}).get(label, {})
        lines.append(f"### {label}  (model {meta.get('model','?')}, "
                     f"output_len {meta.get('output_len','?')})")
        lines.append("")
        header = "| conc \\ ctx | " + " | ".join(
            f"{L // 1024}k" for L in contexts) + " |"
        sep = "|" + "---|" * (len(contexts) + 1)
        lines.append(header)
        lines.append(sep)
        for C in concurrencies:
            row = [f"**{C}**"] + [_cell_str(store[label], C, L) for L in contexts]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    out = "\n".join(lines)
    print(out)
    if a.md:
        with open(a.md, "w") as f:
            f.write(out + "\n")
        print(f"wrote {a.md}", file=sys.stderr)
    if a.csv:
        _write_csv(a.csv, store)
        print(f"wrote {a.csv}", file=sys.stderr)
    if a.svg:
        render_svg(store, a.svg)
        print(f"wrote {a.svg}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="benchmark one dtype arm into the results JSON")
    r.add_argument("--base-url", default="http://localhost:8000")
    r.add_argument("--model", required=True, help="served model name")
    r.add_argument("--label", required=True, help="dtype label, e.g. fp16 / kvarn")
    r.add_argument("--concurrency", default="8,16,32")
    r.add_argument("--context", default="8192,16384,32768")
    r.add_argument("--output-len", type=int, default=256)
    r.add_argument("--waves", type=int, default=2,
                   help="num_prompts = concurrency * waves")
    r.add_argument("--max-model-len", type=int, default=0,
                   help="cells with context > this are marked n/a (0 = no cap)")
    r.add_argument("--kv-pool-tokens", type=int, default=0,
                   help="measured KV-cache token capacity; cells where "
                        "concurrency*context exceeds it are marked OOM (0 = off)")
    r.add_argument("--req-timeout", type=float, default=900.0)
    r.add_argument("--out", default="kvarn_grid.json")
    r.set_defaults(func=cmd_run)

    d = sub.add_parser("render", help="render the combined table (md/csv/svg)")
    d.add_argument("--out", default="kvarn_grid.json", help="results JSON to read")
    d.add_argument("--md", default="", help="write the markdown table here")
    d.add_argument("--csv", default="", help="write the flat CSV here")
    d.add_argument("--svg", default="", help="write the SVG comparison table here")
    d.set_defaults(func=cmd_render)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
