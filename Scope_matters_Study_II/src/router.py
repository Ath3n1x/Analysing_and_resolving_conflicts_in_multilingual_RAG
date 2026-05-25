"""
src/router.py
Language-adaptive retrieval scope router.

Consumes per-language, per-scope metric dicts and outputs a routing table
mapping each language to its optimal retrieval scope (mono / cross / full)
using Hallucination-Penalised F1 (HP-F1) and Language Benefit Score (LBS).
"""

import csv
import json
from collections import Counter
from typing import Dict, List

from src.metrics import compute_lbs, best_scope as _best_scope

BOLD  = "\033[1m"
RESET = "\033[0m"
COL   = {"mono": "\033[95m", "cross": "\033[94m", "full": "\033[92m"}
LABEL = {"mono": "Mono", "cross": "Cross+EN", "full": "Full-Multi"}

LANG_NAMES: Dict[str, str] = {
    "en": "English",    "de": "German",   "es": "Spanish",
    "ar": "Arabic",     "hi": "Hindi",    "vi": "Vietnamese",
    "zh": "Chinese",    "el": "Greek",    "ru": "Russian",
    "tr": "Turkish",    "th": "Thai",
}


def build_routing_table(
    all_results: Dict[str, Dict[str, Dict]],
    lbs_threshold: float = 1.0,
    hall_disqualify: float = 50.0,
    combined_weights: Dict | None = None,
) -> List[Dict]:
    """
    Build a routing table row for each language.

    Args:
        all_results: {lang: {scope: {metric: value}}}
        lbs_threshold: minimum HP-F1 gain to route away from mono.
        hall_disqualify: HALL% threshold to disqualify a scope.
        combined_weights: Optional tiebreaker weights for XQuAD combined score.

    Returns:
        List of row dicts, sorted by mono F1 descending.
    """
    rows = []
    for lang, res in all_results.items():
        lbs      = compute_lbs(res)
        strategy = _best_scope(
            res,
            lbs_threshold=lbs_threshold,
            hall_disqualify=hall_disqualify,
            combined_weights=combined_weights,
        )
        row = {
            "lang":          lang,
            "name":          LANG_NAMES.get(lang, lang),
            "strategy":      strategy,
            **{f"{s}_{m}": res[s][m]
               for s in ("mono", "cross", "full")
               for m in ("em", "f1", "rhr", "lmr", "hall", "rur", "bertscore")
               if m in res[s]},
            **lbs,
        }
        rows.append(row)

    rows.sort(key=lambda r: r.get("mono_f1", 0), reverse=True)
    return rows


def print_routing_table(
    rows: List[Dict],
    all_results: Dict[str, Dict[str, Dict]],
    hall_disqualify: float = 50.0,
    lbs_threshold: float = 1.0,
) -> None:
    """Pretty-print the full metric table and routing summary to stdout."""

    print(f"\n{BOLD}{'─'*120}")
    print("  Metrics Table  (all values %, rounded to 1 dp)")
    print(f"{'─'*120}{RESET}")
    print(
        " " * 20 +
        f"{'── MONO ──────────────────────────────────':^46}"
        f"{'── CROSS+EN ───────────────────────────────':^46}"
        f"{'── FULL-MULTI ────────────────────────────':^46}"
    )
    header = (
        f"{'Lang':<6} {'Name':<13}"
        f" {'EM':>5} {'F1':>6} {'RHR':>5} {'LMR':>5} {'HALL':>5} {'RUR':>5} {'BS':>6}"
        f"  {'EM':>5} {'F1':>6} {'RHR':>5} {'LMR':>5} {'HALL':>5} {'RUR':>5} {'BS':>6}"
        f"  {'EM':>5} {'F1':>6} {'RHR':>5} {'LMR':>5} {'HALL':>5} {'RUR':>5} {'BS':>6}"
    )
    print(BOLD + header + RESET)
    print("─" * len(header))

    for r in rows:
        def _m(s: str, k: str) -> str:
            val = r.get(f"{s}_{k}", 0.0)
            fmt = f"{val:>6.1f}" if k == "bertscore" else f"{val:>5.1f}"
            if k == "hall" and val >= hall_disqualify:
                return f"\033[91m{fmt}\033[0m"
            return fmt

        print(
            f"{r['lang']:<6} {r['name']:<13}"
            f" {_m('mono','em')} {_m('mono','f1')} {_m('mono','rhr')}"
            f" {_m('mono','lmr')} {_m('mono','hall')} {_m('mono','rur')} {_m('mono','bertscore')}"
            f"  {_m('cross','em')} {_m('cross','f1')} {_m('cross','rhr')}"
            f" {_m('cross','lmr')} {_m('cross','hall')} {_m('cross','rur')} {_m('cross','bertscore')}"
            f"  {_m('full','em')} {_m('full','f1')} {_m('full','rhr')}"
            f" {_m('full','lmr')} {_m('full','hall')} {_m('full','rur')} {_m('full','bertscore')}"
        )

    print(f"\n{BOLD}{'─'*90}")
    print(f"  Routing Table   HP-F1 = F1×(1−HALL/100)   threshold={lbs_threshold}")
    print(f"{'─'*90}{RESET}")
    h2 = (
        f"{'Lang':<6} {'Name':<13}"
        f" {'HP-F1 M':>8} {'HP-F1 C':>8} {'HP-F1 F':>8}"
        f"  {'LBS-C(HP)':>10} {'LBS-F(HP)':>10}"
        f"  {'LBS-C(raw)':>11} {'LBS-F(raw)':>11}"
        f"  {'Strategy':>10}"
    )
    print(BOLD + h2 + RESET)
    print("─" * len(h2))

    for r in rows:
        lang = r["lang"]
        col  = COL.get(r["strategy"], "")
        lbl  = LABEL[r["strategy"]]
        c_dq = all_results[lang]["cross"]["hall"] >= hall_disqualify
        f_dq = all_results[lang]["full"]["hall"]  >= hall_disqualify
        lbs_c = (f"\033[91m{r['lbs_cross']:>+10.2f}[DQ]\033[0m" if c_dq
                 else f"{r['lbs_cross']:>+10.2f}")
        lbs_f = (f"\033[91m{r['lbs_full']:>+10.2f}[DQ]\033[0m" if f_dq
                 else f"{r['lbs_full']:>+10.2f}")
        print(
            f"{lang:<6} {r['name']:<13}"
            f" {r['hp_f1_mono']:>8.2f} {r['hp_f1_cross']:>8.2f} {r['hp_f1_full']:>8.2f}"
            f"  {lbs_c}  {lbs_f}"
            f"  {r['lbs_cross_raw']:>+11.4f} {r['lbs_full_raw']:>+11.4f}"
            f"  {col}{lbl:>10}{RESET}"
        )

    print("─" * len(h2))
    counts = Counter(r["strategy"] for r in rows)
    print(
        f"\nSummary → Mono: {counts['mono']}  "
        f"Cross+EN: {counts['cross']}  Full-Multi: {counts['full']}\n"
    )
    print(f"{BOLD}LANGUAGE_STRATEGY = {{{RESET}")
    for r in rows:
        print(f'    "{r["lang"]}": "{r["strategy"]}",  # {r["name"]}')
    print("}")


def save_routing_results(
    rows: List[Dict],
    all_results: Dict,
    results_dir: str,
    prefix: str = "results",
) -> None:
    import os
    os.makedirs(results_dir, exist_ok=True)

    csv_path = f"{results_dir}/{prefix}.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  CSV  → {csv_path}")

    json_path = f"{results_dir}/{prefix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"  JSON → {json_path}")
