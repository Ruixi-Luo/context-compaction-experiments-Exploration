# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai>=2.0",
#     "matplotlib>=3.8",
#     "numpy>=1.26",
# ]
# ///
"""Kimi (Moonshot) arm for the context-compaction membership experiment.

Moonshot's open platform exposes no server-side compaction endpoint, so this
arm implements the client-side LLM-as-compactor design used by Codex,
Claude Code, and Kimi Code's own /compact (GEN class in the paper's
terminology): the same model acts as condenser (summarizes the conversation)
and interpreter (answers membership queries from the summary alone).

Usage:
    export SILICONFLOW_API_KEY=...
    python3 kimi_experiment.py run --seed 42
    python3 kimi_experiment.py plot   # frontier + claude/openai/kimi points

Defaults target SiliconFlow (base-url https://api.siliconflow.cn/v1,
model moonshotai/Kimi-K2.7-Code); set MOONSHOT_BASE_URL/MOONSHOT_API_KEY
to run against Moonshot's official platform instead (e.g. model
kimi-k2.6).

Notes:
- kimi-k3 pins temperature=1.0/top_p=0.95; this script passes no sampling
  params to stay compatible with both k2.6 and k3.
- Records are appended to results/runs.jsonl in the same schema as
  experiment.py, with arm="kimi", so cmd_plot in experiment.py can also
  consume them (after adding "kimi" to its display dict).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import math
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from openai import OpenAI  # noqa: E402

log = logging.getLogger("kimi-experiment")

SCAN_PROMPT = (
    "You have recorded a set of strings. List every string in the set, "
    "one per line."
)
DONE_PROMPT = (
    "The set is complete. Later in this conversation you will be asked "
    "whether a given string was or was not in the set. Reply with OK."
)
QUERY_PROMPT = (
    "Earlier in this conversation you listed a set of strings. Was the "
    "following string in the set?\n\nString: {item}"
)

# Mirrors COMPACTION_INSTRUCTIONS in experiment.py, with an explicit
# character budget. The budget scales with n so that bits/item matches the
# Anthropic endpoint's operating point (~0.95 bits/item, 1,792 chars at
# n=15000), keeping points comparable across providers whose context
# windows force different n.
def target_summary_chars(n: int) -> int:
    return max(500, round(n * 0.954 / 8))
COMPACTION_INSTRUCTIONS = (
    "You are compacting this conversation. Afterward the original messages "
    "are discarded, and you must answer set membership queries using only "
    "your compacted summary: for a given string, whether it was among the "
    "strings listed above. Compact the conversation so as to minimize the "
    "number of membership queries you answer incorrectly. Use whatever "
    "representation best achieves this. Your compacted summary must be at "
    "most {budget} characters."
)

FAIL_FAST_QUERIES = 5
QUERY_WORKERS = 2  # SiliconFlow entry tiers have tight RPM limits
QUERY_TIMEOUT_S = 600
YES_NO_RE = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)


def gzip_bits(text: str) -> int:
    return 8 * len(gzip.compress(text.encode("utf-8"), 9))


def url_dataset(n: int, num_non_members: int, seed: int, csv_path: Path):
    """Same loader as experiment.py (Kaggle malicious_phish.csv)."""
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found; see data/README.md.")
    seen, urls = set(), []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            url = row["url"].strip()
            if url and url not in seen:
                seen.add(url)
                urls.append(url)
    if len(urls) < n + num_non_members:
        raise ValueError(f"need {n + num_non_members} URLs, have {len(urls)}")
    rng = random.Random(seed)
    rng.shuffle(urls)
    return urls[:n], urls[n : n + num_non_members]


def make_queries(members, non_members, num_queries, seed):
    rng = random.Random(seed)
    half = num_queries // 2
    return rng.sample(members, half), rng.sample(non_members, half)


def score(member_answers, non_member_answers) -> dict:
    fn = sum(1 for a in member_answers if a != "YES")
    fp = sum(1 for a in non_member_answers if a == "YES")
    total = len(member_answers) + len(non_member_answers)
    return {
        "num_members_queried": len(member_answers),
        "num_non_members_queried": len(non_member_answers),
        "error_rate": (fn + fp) / total,
        "false_negative_rate": fn / len(member_answers),
        "false_positive_rate": fp / len(non_member_answers),
        "num_failed_queries": sum(
            1 for a in member_answers + non_member_answers if a is None
        ),
    }


def run_kimi(members, member_qs, non_member_qs, model: str, base_url: str) -> dict:
    client = OpenAI(
        base_url=base_url,
        api_key=os.environ.get("SILICONFLOW_API_KEY")
        or os.environ["OPENAI_API_KEY"],
        timeout=600.0,
        max_retries=3,
    )
    n = len(members)

    log.info("compacting %d items with kimi model=%s", n, model)
    # Long-context condensing routinely exceeds 10 min server-side; the
    # default 600s client timeout would otherwise kill and retry it.
    condense = client.with_options(timeout=2400.0).chat.completions.create(
        model=model,
        max_tokens=16384,  # room for enumeration strategies; the paper's
        # server endpoint had no output cap, so we must not impose a tight one
        messages=[
            {"role": "system", "content": COMPACTION_INSTRUCTIONS.format(
                budget=target_summary_chars(n))},
            {"role": "user", "content": SCAN_PROMPT},
            {"role": "assistant", "content": "\n".join(members)},
            {"role": "user", "content": DONE_PROMPT},
            # Force the summary rather than an "OK" reply.
            {"role": "user", "content":
                "Now produce the compacted summary described in your "
                "instructions. Output ONLY the summary itself."},
        ],
    )
    if condense.choices[0].finish_reason == "length":
        raise SystemExit(
            "Compaction response hit max_tokens; the summary may be "
            "truncated. Raise the max_tokens cap in run_kimi()."
        )
    summary = condense.choices[0].message.content or ""
    if len(summary) > int(1.5 * target_summary_chars(n)):
        log.warning("summary %d chars exceeds requested budget", len(summary))
    log.info("compacted: summary %d chars (%.2f bits/item gzip)",
             len(summary), gzip_bits(summary) / n)
    log.info("compacted summary:\n%s", summary)

    base_messages = [
        {"role": "system", "content":
            "You will be shown a compacted summary of an earlier conversation "
            "and asked whether a given string appeared in a set recorded in "
            "that conversation. Answer with exactly YES or NO."},
        {"role": "user", "content":
            f"Compacted summary:\n{summary}\n\n"
            "Answer only YES or NO."},
    ]

    def ask(item: str):
        try:
            reply = client.chat.completions.create(
                model=model,
                messages=base_messages
                + [{"role": "user", "content": QUERY_PROMPT.format(item=item)}],
                max_tokens=16,
            )
            text = reply.choices[0].message.content or ""
            m = YES_NO_RE.search(text)
            return m.group(1).upper() if m else None
        except Exception as e:  # noqa: BLE001
            log.warning("query failed: %s", e)
            return None

    items = member_qs + non_member_qs
    probe = [ask(x) for x in items[:FAIL_FAST_QUERIES]]
    if all(a is None for a in probe):
        raise SystemExit("First queries all failed; aborting run.")
    answers = probe + [None] * (len(items) - len(probe))
    ex = ThreadPoolExecutor(max_workers=QUERY_WORKERS)
    futures = {ex.submit(ask, x): i
               for i, x in enumerate(items[FAIL_FAST_QUERIES:])}
    try:
        done = 0
        for fut in as_completed(futures, timeout=QUERY_TIMEOUT_S):
            answers[FAIL_FAST_QUERIES + futures[fut]] = fut.result()
            done += 1
            if done % 25 == 0:
                log.info("  %d/%d queries done", done, len(futures))
    except TimeoutError:
        log.warning("query phase timed out; rest count as failed")
    ex.shutdown(wait=False, cancel_futures=True)

    record = {
        "arm": "kimi",
        "model": model,
        "n": n,
        "bits_per_item": gzip_bits(summary) / n,
        "summary_gzip_bits": gzip_bits(summary),
        "summary_chars": len(summary),
        "summary": summary,
    }
    record.update(score(answers[: len(member_qs)], answers[len(member_qs):]))
    return record


def bloom_fpr(bits_per_item: float) -> float:
    return math.pow(2.0, -bits_per_item * math.log(2.0))


def lower_bound_fpr(bits_per_item: float) -> float:
    return math.pow(2.0, -bits_per_item)


def cmd_run(args):
    if args.num_queries < 2 or args.num_queries % 2:
        raise SystemExit("--num-queries must be even")
    members, non_members = url_dataset(
        args.n, args.num_queries, args.seed, args.urls_csv)
    member_qs, non_member_qs = make_queries(
        members, non_members, args.num_queries, args.seed + 1)
    record = run_kimi(members, member_qs, non_member_qs,
                      args.model, args.base_url)
    record.update(dataset="urls", seed=args.seed,
                  timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    log.info("kimi done: error=%.3f (FPR=%.3f, FNR=%.3f) -> %s",
             record["error_rate"], record["false_positive_rate"],
             record["false_negative_rate"], args.out)


def cmd_plot(args):
    records = [json.loads(l) for l in open(args.results) if l.strip()]
    claude = [r for r in records if r.get("arm") == "claude"]
    kimi = [r for r in records if r.get("arm") == "kimi"]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 13, "legend.fontsize": 10,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    n_items = (claude or kimi)[0]["n"]
    xmax = 20.0
    b = np.linspace(0.0, xmax, 400)
    bloom = [bloom_fpr(v * 1000 / n_items) * 0.5 for v in b]
    bound = [lower_bound_fpr(v * 1000 / n_items) * 0.5 for v in b]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.fill_between(b, 0, bloom, color="#2ca02c", alpha=0.08, lw=0)
    ax.axhspan(0.5, 1.0, color="#d62728", alpha=0.06)
    ax.text(0.04, 0.94, "Worse Than Random Guess",
            transform=ax.transAxes, fontsize=9, color="#b03030", style="italic")
    ax.text(0.04, 0.06, "Better Than Bloom Filter",
            transform=ax.transAxes, fontsize=9, color="#2e7d32", style="italic")
    ax.plot(b, bloom, "-", color="black", lw=2.2)
    ax.plot(b, bound, "--", color="#2ca02c", lw=1.8)
    ax.axhline(0.5, color="#d62728", linestyle=":", lw=1.5)

    for rs, marker, label in (
        (claude, "x", "Opus 4.8 (server compaction)"),
        (kimi, "s", "Kimi (client-side LLM compaction)"),
    ):
        xs = [r["summary_chars"] * 8 / 1000 for r in rs]
        ys = [r["error_rate"] for r in rs]
        ax.plot(xs, ys, marker=marker, linestyle="none", markersize=7,
                markerfacecolor="none", markeredgewidth=1.3, color="black",
                label=label)

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Compaction Budget (Kbits)")
    ax.set_ylabel("Error Rate")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color="0.9", lw=0.6)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95,
              edgecolor="0.8")
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "membership_pareto_with_kimi.pdf"
    fig.savefig(out, bbox_inches="tight")
    log.info("wrote %s", out)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run")
    run_p.add_argument("--model", default="moonshotai/Kimi-K2.7-Code")
    run_p.add_argument("--base-url",
                       default=os.environ.get("MOONSHOT_BASE_URL",
                                              "https://api.siliconflow.cn/v1"))
    run_p.add_argument("--n", type=int, default=6000)
    run_p.add_argument("--num-queries", type=int, default=200)
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--urls-csv", type=Path,
                       default=Path("data/malicious_phish.csv"))
    run_p.add_argument("--out", type=Path, default=Path("results/runs.jsonl"))
    run_p.set_defaults(func=cmd_run)

    plot_p = sub.add_parser("plot")
    plot_p.add_argument("--results", type=Path, default=Path("results/runs.jsonl"))
    plot_p.add_argument("--out-dir", type=Path, default=Path("results"))
    plot_p.set_defaults(func=cmd_plot)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
