"""
experiments/run_mlqa.py
MLQA experiment entry point.

Runs the full pipeline for each (generator, retriever) pair specified in
configs/mlqa.yaml across all 7 languages and 3 retrieval scopes.

Usage
-----
    export HF_TOKEN="hf_..."

    # Full run (all 4 generators × 2 retrievers):
    python experiments/run_mlqa.py

    # Subset:
    python experiments/run_mlqa.py --generators aya qwen --retrievers bgem3
    python experiments/run_mlqa.py --langs en hi zh --samples 20
    python experiments/run_mlqa.py --load-4bit --no-plot

Output
------
    results/mlqa/<generator>_<retriever>/results.csv
    results/mlqa/<generator>_<retriever>/results.json
    results/mlqa/<generator>_<retriever>/figures/

    results/mlqa/summary.json    — routing table across all pairs
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datasets import load_dataset
from huggingface_hub import login as hf_login

from src.chunker   import build_chunk_store
from src.retriever import load_retriever, build_scope_indexes, retrieve, RERANK_LANGS_MLQA
from src.generator import load_generator, unload_generator, generate_answer
from src.metrics   import (
    exact_match, f1_score, retrieval_hit_rate, language_match_rate,
    hallucination_rate, retrieval_utility_rate, bertscore_batch,
    aggregate_scope,
)
from src.router  import build_routing_table, print_routing_table, save_routing_results
from src.plotting import plot_all


# ── Config ───────────────────────────────────────────────────────────────────

def load_config(path: str = "configs/mlqa.yaml") -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Dataset loader ───────────────────────────────────────────────────────────

def load_mlqa(languages: List[str], n_samples: int) -> Dict[str, List[Dict]]:
    print(f"\n[MLQA] Loading dataset (up to {n_samples} samples/lang)")
    corpus: Dict[str, List[Dict]] = {}
    for lang in languages:
        config = f"mlqa.{lang}.{lang}"
        ds     = load_dataset("facebook/mlqa", config)["test"]
        rows: List[Dict] = []
        for item in ds:
            ans = item["answers"]
            answers = ans["text"] if isinstance(ans, dict) else ans
            if isinstance(answers, str):
                answers = [answers]
            rows.append({
                "id":       item["id"],
                "context":  item["context"],
                "question": item["question"],
                "answers":  answers,
            })
            if len(rows) >= n_samples:
                break
        corpus[lang] = rows
        print(f"  [{lang}] {len(rows)} samples")
    return corpus


# ── Per-language evaluation ───────────────────────────────────────────────────

def evaluate_language(
    lang: str,
    rows: List[Dict],
    indexes: Dict,
    retriever_model,
    gen_tok,
    gen_model,
    cfg: Dict,
) -> Dict[str, Dict[str, float]]:

    ret_cfg  = cfg["retrieval"]
    gen_cfg  = cfg["generation"]
    top_k    = ret_cfg["top_k"].get(lang, 6)
    rr_langs = set(ret_cfg.get("rerank_langs", []))
    max_ntok = gen_cfg["max_new_tokens"].get(lang, 45)
    max_atok = gen_cfg["max_answer_tokens"]
    rep_pen  = gen_cfg.get("repetition_penalty", 1.3)

    acc: Dict[str, Dict[str, List]] = {
        s: defaultdict(list) for s in ("mono", "cross", "full")
    }
    bert_buf: Dict[str, Dict[str, List]] = {
        s: {"preds": [], "refs": []} for s in ("mono", "cross", "full")
    }

    for row in tqdm(rows, desc=f"  [{lang}]", leave=False):
        q       = row["question"]
        answers = row["answers"]
        retrieved = retrieve(
            q, indexes, retriever_model, lang,
            top_k=top_k, keep_k=top_k, rerank_langs=rr_langs,
        )

        for scope in ("mono", "cross", "full"):
            chunks = retrieved[scope]

            pred = generate_answer(
                q, chunks, lang, gen_tok, gen_model,
                max_new_tokens=max_ntok,
                repetition_penalty=rep_pen,
                max_answer_tokens=max_atok,
            )

            best_ref = max(answers, key=lambda a: f1_score(pred, [a], lang))
            acc[scope]["em"].append(exact_match(pred, answers, lang))
            acc[scope]["f1"].append(f1_score(pred, answers, lang))
            acc[scope]["rhr"].append(retrieval_hit_rate(chunks, answers, lang))
            acc[scope]["lmr"].append(language_match_rate(pred, lang))
            acc[scope]["hall"].append(hallucination_rate(pred, chunks, answers, lang))
            acc[scope]["rur"].append(retrieval_utility_rate(pred, chunks, lang))
            bert_buf[scope]["preds"].append(pred)
            bert_buf[scope]["refs"].append(best_ref)

    results: Dict[str, Dict[str, float]] = {}
    for scope in ("mono", "cross", "full"):
        bs = bertscore_batch(bert_buf[scope]["preds"], bert_buf[scope]["refs"], lang)
        acc[scope]["bertscore"] = bs
        results[scope] = aggregate_scope(acc[scope])
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MLQA RAG pipeline")
    p.add_argument("--config",      default="configs/mlqa.yaml")
    p.add_argument("--generators",  nargs="+",
                   help="Generator IDs to run (default: all from config)")
    p.add_argument("--retrievers",  nargs="+",
                   help="Retriever IDs to run (default: all from config)")
    p.add_argument("--langs",       nargs="+",
                   help="Language subset (default: all from config)")
    p.add_argument("--samples",     type=int,
                   help="QA pairs per language (overrides config)")
    p.add_argument("--load-4bit",   action="store_true")
    p.add_argument("--no-plot",     action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    # Auth
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("Set HF_TOKEN environment variable.")
    hf_login(token=token)

    # Overrides
    ds_cfg     = cfg["dataset"]
    languages  = args.langs    or ds_cfg["languages"]
    n_samples  = args.samples  or ds_cfg["max_samples"]
    gen_cfgs   = cfg["generators"]
    ret_cfgs   = cfg["retrievers"]

    if args.generators:
        gen_cfgs = [g for g in gen_cfgs if g["id"] in args.generators]
    if args.retrievers:
        ret_cfgs = [r for r in ret_cfgs if r["id"] in args.retrievers]

    # Load dataset once
    corpus = load_mlqa(languages, n_samples)

    # Chunk once (language-aware, same for all retriever/generator pairs)
    print("\n[MLQA] Chunking passages...")
    store = build_chunk_store(corpus, cfg["chunking"])

    summary: Dict[str, Dict] = {}  # (gen_id, ret_id) → routing table rows

    for ret_cfg in ret_cfgs:
        print(f"\n{'='*60}")
        print(f"  Retriever: {ret_cfg['model_id']}")
        print(f"{'='*60}")
        retriever_model = load_retriever(
            ret_cfg["model_id"], use_fp16=ret_cfg.get("use_fp16", True)
        )

        # Pre-build scope indexes per language (encoding is the bottleneck)
        print("  Building scope indexes...")
        lang_indexes = {}
        for lang in languages:
            lang_indexes[lang] = build_scope_indexes(
                store, retriever_model, lang, languages
            )

        for gen_cfg_item in gen_cfgs:
            gen_id = gen_cfg_item["id"]
            ret_id = ret_cfg["id"]
            run_id = f"{gen_id}_{ret_id}"
            results_dir = f"{cfg['output']['results_dir']}/{run_id}"

            print(f"\n  Generator: {gen_cfg_item['model_id']}  [{run_id}]")
            gen_tok, gen_model = load_generator(
                gen_cfg_item["model_id"],
                torch_dtype_str=gen_cfg_item.get("torch_dtype", "float16"),
                load_in_4bit=args.load_4bit or gen_cfg_item.get("load_in_4bit", False),
            )

            all_results: Dict[str, Dict] = {}
            for lang in languages:
                print(f"\n  Language: {lang}")
                all_results[lang] = evaluate_language(
                    lang, corpus[lang], lang_indexes[lang],
                    retriever_model, gen_tok, gen_model, cfg,
                )
                for scope in ("mono", "cross", "full"):
                    r = all_results[lang][scope]
                    print(
                        f"    [{scope:5}] EM={r['em']:5.1f}  F1={r['f1']:5.1f}  "
                        f"RHR={r['rhr']:5.1f}  HALL={r['hall']:5.1f}  BS={r.get('bertscore',0):5.1f}"
                    )

            route_cfg = cfg["routing"]
            rows = build_routing_table(
                all_results,
                lbs_threshold=route_cfg["lbs_threshold"],
                hall_disqualify=route_cfg["hall_disqualify"],
            )
            print_routing_table(rows, all_results,
                                hall_disqualify=route_cfg["hall_disqualify"],
                                lbs_threshold=route_cfg["lbs_threshold"])
            save_routing_results(rows, all_results, results_dir)

            if not args.no_plot and cfg["output"].get("plot", True):
                plot_all(rows, results_dir, prefix=run_id,
                         lbs_threshold=route_cfg["lbs_threshold"])

            summary[run_id] = {r["lang"]: r["strategy"] for r in rows}

            unload_generator(gen_cfg_item["model_id"])

        del retriever_model  # free embedding model VRAM before next retriever

    # Cross-run summary
    out_dir = cfg["output"]["results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary → {out_dir}/summary.json")


if __name__ == "__main__":
    main()
