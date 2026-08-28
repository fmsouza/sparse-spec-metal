#!/usr/bin/env python3
"""Phase 0, task 0.2 — assemble agentic-coding traces + a WikiText control.

Traces are built from local Claude Code session logs (~/.claude/projects/**/*.jsonl).
Each trace is a (prefix, target) pair:

  prefix  — the conversation leading up to an assistant turn (user asks, prior
            assistant turns, tool results). This is *prefill*; no masks measured.
  target  — one assistant turn: prose + its tool calls rendered as the edits/diffs
            they are. This is the *decode* region, and the only region measured.

Rationale: MTP verification operates on model-generated output tokens, so the
activation statistics that matter are those of the decode region, not of the tool
output the model merely reads.

Output: analysis/traces/*.json (gitignored). Provenance summary printed and written
to analysis/traces/MANIFEST.json.
"""

import argparse
import hashlib
import json
import os
import pathlib
import random
import re
import sys

EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit", "Update"}
READ_TOOLS = {"Read", "Bash", "Grep", "Glob", "BashOutput"}

# file extension -> coarse language label, for provenance reporting
EXT_LANG = {
    ".ts": "ts", ".tsx": "ts", ".js": "js", ".jsx": "js",
    ".rs": "rust", ".py": "python", ".go": "go", ".swift": "swift",
    ".kt": "kotlin", ".java": "java", ".c": "c", ".h": "c", ".cpp": "cpp",
    ".m": "objc", ".mm": "objc", ".sh": "shell", ".yml": "yaml", ".yaml": "yaml",
    ".json": "json", ".toml": "toml", ".md": "markdown", ".sql": "sql",
    ".css": "css", ".html": "html", ".metal": "metal",
}


def render_tool_use(name, inp):
    """Render a tool_use block the way its content actually looks to a decoder."""
    if not isinstance(inp, dict):
        return f"[{name}]"
    fp = inp.get("file_path") or inp.get("path") or ""
    if name in ("Edit", "Update"):
        return (
            f"[Edit {fp}]\n<<<<<<< old\n{inp.get('old_string', '')}\n"
            f"=======\n{inp.get('new_string', '')}\n>>>>>>> new\n"
        )
    if name == "MultiEdit":
        parts = [f"[MultiEdit {fp}]"]
        for e in inp.get("edits", []) or []:
            parts.append(
                f"<<<<<<< old\n{e.get('old_string', '')}\n"
                f"=======\n{e.get('new_string', '')}\n>>>>>>> new"
            )
        return "\n".join(parts) + "\n"
    if name == "Write":
        return f"[Write {fp}]\n{inp.get('content', '')}\n"
    if name == "Bash":
        return f"[Bash]\n$ {inp.get('command', '')}\n"
    if name in ("Grep", "Glob"):
        return f"[{name}] {json.dumps(inp, ensure_ascii=False)[:400]}\n"
    if name == "Read":
        return f"[Read {fp}]\n"
    return f"[{name}] {json.dumps(inp, ensure_ascii=False)[:600]}\n"


def blocks_text(content):
    """Flatten a message content field to text, keeping tool structure visible."""
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return "", []
    out, tools = [], []
    for b in content:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            out.append(b.get("text", ""))
        elif t == "thinking":
            continue  # not part of the emitted answer stream we care about
        elif t == "tool_use":
            name = b.get("name", "?")
            tools.append((name, b.get("input")))
            out.append(render_tool_use(name, b.get("input")))
        elif t == "tool_result":
            c = b.get("content")
            if isinstance(c, list):
                c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
            out.append(f"[tool result]\n{str(c)[:4000]}\n")
    return "\n".join(x for x in out if x), tools


def iter_turns(path):
    """Yield (role, text, tools) in file order for one session log."""
    try:
        fh = open(path, "r", errors="replace")
    except OSError:
        return
    with fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = d.get("type")
            if t not in ("user", "assistant"):
                continue
            m = d.get("message")
            if not isinstance(m, dict):
                continue
            text, tools = blocks_text(m.get("content"))
            if not text.strip():
                continue
            yield t, text, tools


def exts_of(tools):
    out = set()
    for name, inp in tools:
        if not isinstance(inp, dict):
            continue
        fp = inp.get("file_path") or inp.get("path") or ""
        e = os.path.splitext(fp)[1].lower()
        if e in EXT_LANG:
            out.add(EXT_LANG[e])
    return out


def build(args):
    root = pathlib.Path(os.path.expanduser("~/.claude/projects"))
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    def ntok(s):
        return len(tok(s, add_special_tokens=False)["input_ids"])

    # ---- candidate harvest -------------------------------------------------
    self_proj = "sparse-spec-metal"          # exclude this investigation's own logs
    logs = [p for p in root.rglob("*.jsonl") if self_proj not in str(p)]
    logs.sort(key=lambda p: p.stat().st_size, reverse=True)
    logs = [p for p in logs if p.stat().st_size > 100_000]
    rng = random.Random(args.seed)
    rng.shuffle(logs)

    cands = []
    seen_hash = set()
    for p in logs:
        if len(cands) >= args.pool:
            break
        proj = p.parent.name.replace("-Users-fred-", "").split("--claude-worktrees")[0]
        hist = []
        for role, text, tools in iter_turns(p):
            if role == "assistant" and any(n in EDIT_TOOLS for n, _ in tools):
                tgt_tokens = ntok(text)
                if not (args.min_target <= tgt_tokens <= args.max_target):
                    hist.append((role, text))
                    continue
                h = hashlib.sha256(text.encode()).hexdigest()[:16]
                if h in seen_hash:
                    hist.append((role, text))
                    continue
                seen_hash.add(h)
                cands.append({
                    "project": proj,
                    "session": p.stem,
                    "sha16": h,
                    "langs": sorted(exts_of(tools)),
                    "tools": sorted({n for n, _ in tools}),
                    "target": text,
                    "target_tokens": tgt_tokens,
                    "history": list(hist[-args.hist_turns:]),
                })
            hist.append((role, text))

    if not cands:
        sys.exit("no candidate turns found")

    # ---- diverse selection: round-robin over (project, primary language) ----
    buckets = {}
    for c in cands:
        key = (c["project"], c["langs"][0] if c["langs"] else "mixed")
        buckets.setdefault(key, []).append(c)
    for v in buckets.values():
        rng.shuffle(v)
    order = sorted(buckets)

    def draw(n):
        picked, i = [], 0
        while len(picked) < n and any(buckets[k] for k in order):
            k = order[i % len(order)]
            if buckets[k]:
                picked.append(buckets[k].pop())
            i += 1
        return picked

    # calibration is drawn FIRST and popped from the pool, so the measurement
    # traces below are guaranteed disjoint from it (different assistant turns,
    # and usually different sessions/projects).
    calib = draw(args.n_calib)
    chosen = draw(args.n)

    # ---- render prefix via the model's own chat template --------------------
    manifest = []

    def emit(cs, kind, tag):
      for idx, c in enumerate(cs):
          msgs = []
          for role, text in c["history"]:
              if msgs and msgs[-1]["role"] == role:
                  msgs[-1]["content"] += "\n\n" + text
              else:
                  msgs.append({"role": role, "content": text})
          if not msgs or msgs[-1]["role"] != "user":
              msgs.append({"role": "user", "content": "Continue."})
          # trim prefix from the front until it fits
          while len(msgs) > 1:
              prefix = tok.apply_chat_template(
                  msgs, tokenize=False, add_generation_prompt=True)
              if ntok(prefix) <= args.max_prefix:
                  break
              msgs.pop(0)
          prefix = tok.apply_chat_template(
              msgs, tokenize=False, add_generation_prompt=True)

          rec = {
              "id": f"{tag}-{idx:02d}",
              "kind": kind,
              "project": c["project"],
              "session": c["session"],
              "sha16": c["sha16"],
              "langs": c["langs"],
              "tools": c["tools"],
              "prefix": prefix,
              "target": c["target"],
              "prefix_tokens": ntok(prefix),
              "target_tokens": c["target_tokens"],
          }
          (outdir / f"{rec['id']}.json").write_text(json.dumps(rec))
          manifest.append({k: v for k, v in rec.items() if k not in ("prefix", "target")})

    emit(calib, "calib", "calib")
    emit(chosen, "agentic", "agentic")

    # ---- WikiText-2 control -------------------------------------------------
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(t for t in ds["text"] if len(t.strip()) > 200)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    per = args.max_target
    for j in range(args.n_control + args.n_calib_control):
        is_cal = j < args.n_calib_control
        s = j * (per + args.max_prefix // 2)
        pre_ids = ids[s: s + 256]
        tgt_ids = ids[s + 256: s + 256 + per]
        if len(tgt_ids) < per:
            break
        rec = {
            "id": (f"calib-wiki-{j:02d}" if is_cal
                   else f"wikitext-{j - args.n_calib_control:02d}"),
            "kind": "calib" if is_cal else "control",
            "project": "wikitext-2-raw-v1/test",
            "session": "-", "sha16": "-", "langs": ["prose"], "tools": [],
            "prefix": tok.decode(pre_ids),
            "target": tok.decode(tgt_ids),
            "prefix_tokens": len(pre_ids),
            "target_tokens": len(tgt_ids),
        }
        (outdir / f"{rec['id']}.json").write_text(json.dumps(rec))
        manifest.append({k: v for k, v in rec.items() if k not in ("prefix", "target")})

    (outdir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    ag = [m for m in manifest if m["kind"] == "agentic"]
    cal = [m for m in manifest if m["kind"] == "calib"]
    print(f"agentic traces : {len(ag)}")
    print(f"control traces : {len([m for m in manifest if m['kind']=='control'])}")
    print(f"calib traces   : {len(cal)} "
          f"({sum(m['target_tokens'] for m in cal)} decode tokens)")
    print(f"projects       : {sorted({m['project'] for m in ag})}")
    print(f"languages      : {sorted({l for m in ag for l in m['langs']})}")
    print(f"tools          : {sorted({t for m in ag for t in m['tools']})}")
    print(f"decode tokens  : agentic {sum(m['target_tokens'] for m in ag)}, "
          f"control {sum(m['target_tokens'] for m in manifest if m['kind']=='control')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.8-27B-4bit")
    ap.add_argument("--out", default="analysis/traces")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--n-control", type=int, default=4)
    ap.add_argument("--n-calib", type=int, default=6)
    ap.add_argument("--n-calib-control", type=int, default=2)
    ap.add_argument("--pool", type=int, default=600)
    ap.add_argument("--min-target", type=int, default=384)
    ap.add_argument("--max-target", type=int, default=768)
    ap.add_argument("--max-prefix", type=int, default=2048)
    ap.add_argument("--hist-turns", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    build(ap.parse_args())
