"""
src/metrics.py
Evaluation metrics for multilingual extractive QA.

Implements all metrics described in the paper:
  EM          — Exact Match
  F1          — Token-level F1 (character-level for Zh/Th)
  RHR         — Retrieval Hit Rate
  LMR         — Language Match Rate (Unicode-range based)
  HALL        — Hallucination Rate (graded, F1+overlap thresholds)
  RUR         — Retrieval Utility Rate
  BERTScore   — Semantic similarity via contextual embeddings
  HP-F1       — Hallucination-Penalised F1
  LBS         — Language Benefit Score (routing signal)
"""

import re
import unicodedata
from collections import Counter
from typing import Dict, List, Tuple

try:
    import torch
    from bert_score import score as _bert_score_fn
    _HAVE_BERTSCORE = True
except ImportError:
    _HAVE_BERTSCORE = False

try:
    import jieba
    jieba.setLogLevel(20)
    _HAVE_JIEBA = True
except ImportError:
    _HAVE_JIEBA = False

UNICODE_RANGES: Dict[str, List[Tuple[int, int]]] = {
    "ar": [(0x0600, 0x06FF)],
    "ru": [(0x0400, 0x04FF)],
    "hi": [(0x0900, 0x097F)],
    "th": [(0x0E00, 0x0E7F)],
    "zh": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x3000, 0x303F)],
    "el": [(0x0370, 0x03FF), (0x1F00, 0x1FFF)],
    "tr": [(0x0041, 0x007A), (0x00C0, 0x00FF),
           (0x011E, 0x011F), (0x0130, 0x0131),
           (0x015E, 0x015F), (0x00D6, 0x00D6), (0x00FC, 0x00FC)],
    "vi": [(0x0041, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF)],
    "es": [(0x0041, 0x007A), (0x00C0, 0x00FF)],
    "de": [(0x0041, 0x007A), (0x00C0, 0x00FF)],
    "en": [(0x0041, 0x007A)],
}

CHAR_LEVEL_LANGS = {"zh", "th"}

_TURKISH_SUFFIXES = sorted([
    "lar","ler","ın","in","un","ün","ı","i","u","ü",
    "da","de","ta","te","dan","den","tan","ten",
    "a","e","ya","ye","nın","nin","nun","nün",
    "yla","yle","la","le","ca","ce","çe","ça",
    "lık","lik","luk","lük","sız","siz","suz","süz",
    "yor","iyor","uyor","üyor","dı","di","du","dü",
    "tı","ti","tu","tü","mış","miş","muş","müş",
], key=len, reverse=True)

_GERMAN_SUFFIXES = sorted([
    "en","em","er","es","e","ern","est","ste","sten","stem","ster",
    "ung","ungen","heit","keit","schaft","lich","isch",
], key=len, reverse=True)


# ── Text normalisation ───────────────────────────────────────────────────────

def _stem_turkish(text: str) -> str:
    words, stemmed = text.lower().split(), []
    for w in words:
        for suf in _TURKISH_SUFFIXES:
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                w = w[:-len(suf)]
                break
        stemmed.append(w)
    return " ".join(stemmed)


def _stem_german(text: str) -> str:
    words, stemmed = text.lower().split(), []
    for w in words:
        for suf in _GERMAN_SUFFIXES:
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                w = w[:-len(suf)]
                break
        stemmed.append(w)
    return " ".join(stemmed)


def _norm_vietnamese(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text)
    out = []
    for ch in nfd:
        if unicodedata.category(ch) == "Mn":
            continue
        out.append("d" if ch == "đ" else "D" if ch == "Đ" else ch)
    return "".join(out)


def normalize_text(text: str, lang: str = "") -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    if lang == "ar":
        text = re.sub(r"[إأآا]", "ا", text)
        text = re.sub(r"ى",       "ي", text)
        text = re.sub(r"ة",       "ه", text)
    if lang in ("en", "es"):
        text = re.sub(r"\b(a|an|the|el|la|los|las|un|una)\s+", "", text, flags=re.I)
    if lang == "tr":
        text = _stem_turkish(text)
    if lang == "de":
        text = _stem_german(text)
    if lang == "vi":
        text = _norm_vietnamese(text)
    return text.strip()


def tokenize_text(text: str, lang: str = "") -> List[str]:
    if not text:
        return []
    text = normalize_text(text, lang)
    if lang == "zh":
        return (list(jieba.cut(text)) if _HAVE_JIEBA
                else [c for c in text if not c.isspace()])
    if lang == "th":
        return [c for c in text if not c.isspace()]
    return re.findall(r"\w+", text.lower())


# ── Core metrics ─────────────────────────────────────────────────────────────

def exact_match(pred: str, refs: List[str], lang: str) -> float:
    pn = normalize_text(pred, lang)
    return float(any(pn == normalize_text(r, lang) for r in refs))


def f1_score(pred: str, refs: List[str], lang: str) -> float:
    def _f1(p: str, r: str) -> float:
        pt = tokenize_text(p, lang)
        rt = tokenize_text(r, lang)
        if not pt or not rt:
            return 0.0
        common = sum((Counter(pt) & Counter(rt)).values())
        prec   = common / len(pt)
        rec    = common / len(rt)
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return max(_f1(pred, r) for r in refs)


def retrieval_hit_rate(chunks: List[str], refs: List[str], lang: str) -> float:
    """1 if any gold answer span appears in any retrieved chunk."""
    for ans in refs:
        norm_ans = normalize_text(ans, lang)
        if not norm_ans or len(norm_ans) <= 2:
            continue
        for chunk in chunks:
            if norm_ans in normalize_text(chunk, lang):
                return 1.0
    return 0.0


def language_match_rate(pred: str, lang: str) -> float:
    """
    Fraction of non-trivial characters in pred that fall within
    the expected Unicode range(s) for lang.
    """
    ranges   = UNICODE_RANGES.get(lang, [])
    _NEUTRAL = set('.,;:!?-\u2013\u2014()[]{}"\'/\\@#%&*+')
    script_chars = [
        c for c in pred
        if not c.isspace() and not c.isdigit() and c not in _NEUTRAL
    ]
    if not script_chars:
        return 1.0
    match = sum(
        1 for c in script_chars
        if any(lo <= ord(c) <= hi for lo, hi in ranges)
    )
    return match / len(script_chars)


def hallucination_rate(
    pred: str,
    chunks: List[str],
    refs: List[str],
    lang: str,
    f1_threshold: float = 0.3,
    overlap_threshold: float = 0.5,
) -> float:
    """
    Graded hallucination:
      0.0 if F1 > f1_threshold (model answered correctly)
      0.0 if token overlap with context > overlap_threshold (grounded)
      1.0 otherwise (ungrounded and incorrect)
    """
    if f1_score(pred, refs, lang) > f1_threshold:
        return 0.0
    pred_tok = set(tokenize_text(pred, lang))
    ctx_tok  = set(tokenize_text("\n".join(chunks), lang))
    if pred_tok and len(pred_tok & ctx_tok) / len(pred_tok) > overlap_threshold:
        return 0.0
    return 1.0


def retrieval_utility_rate(pred: str, chunks: List[str], lang: str) -> float:
    """
    Fraction of predicted tokens that appear in the retrieved context.
    RUR = |P ∩ C| / |P|
    """
    pred_tok = set(tokenize_text(pred, lang))
    ctx_tok  = set(tokenize_text("\n".join(chunks), lang))
    if not pred_tok:
        return 0.0
    return len(pred_tok & ctx_tok) / len(pred_tok)


def bertscore_batch(
    preds: List[str],
    refs: List[str],
    lang: str,
    model_type: str = "bert-base-multilingual-cased",
) -> List[float]:
    if not _HAVE_BERTSCORE or not preds:
        return [0.0] * len(preds)
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _, _, F = _bert_score_fn(
            preds, refs,
            model_type=model_type,
            lang=None,
            verbose=False,
            device=device,
        )
        return F.tolist()
    except Exception as e:
        print(f"  [warn] BERTScore failed ({e}) — returning 0.0")
        return [0.0] * len(preds)


# ── Routing metrics (HP-F1 + LBS) ───────────────────────────────────────────

def hp_f1(scope_metrics: Dict[str, float]) -> float:
    """
    Hallucination-Penalised F1:
      HP-F1 = F1 × (1 − HALL / 100)

    scope_metrics keys expected: "f1" (0–100), "hall" (0–100).
    """
    return scope_metrics["f1"] * (1.0 - scope_metrics["hall"] / 100.0)


def compute_lbs(lang_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Language Benefit Score for cross and full scopes relative to mono.
    Also returns raw F1 gain for reference.

    Returns dict with keys: lbs_cross, lbs_full, lbs_cross_raw, lbs_full_raw,
                             hp_f1_mono, hp_f1_cross, hp_f1_full.
    """
    hp_mono  = hp_f1(lang_results["mono"])
    hp_cross = hp_f1(lang_results["cross"])
    hp_full  = hp_f1(lang_results["full"])
    f1_mono  = lang_results["mono"]["f1"]
    return {
        "lbs_cross":     round(hp_cross - hp_mono, 4),
        "lbs_full":      round(hp_full  - hp_mono, 4),
        "lbs_cross_raw": round(lang_results["cross"]["f1"] - f1_mono, 4),
        "lbs_full_raw":  round(lang_results["full"]["f1"]  - f1_mono, 4),
        "hp_f1_mono":    round(hp_mono,  4),
        "hp_f1_cross":   round(hp_cross, 4),
        "hp_f1_full":    round(hp_full,  4),
    }


def best_scope(
    lang_results: Dict[str, Dict[str, float]],
    lbs_threshold: float = 1.0,
    hall_disqualify: float = 50.0,
    combined_weights: Dict[str, float] | None = None,
) -> str:
    """
    Select the best retrieval scope for a language using LBS + HP-F1.

    Scopes with HALL ≥ hall_disqualify are disqualified.
    If no scope beats lbs_threshold, returns 'mono'.
    When cross and full are tied, uses a combined score as a tiebreaker.
    """
    lbs = compute_lbs(lang_results)

    cross_hall = lang_results["cross"]["hall"]
    full_hall  = lang_results["full"]["hall"]
    lbs_cross  = lbs["lbs_cross"] if cross_hall < hall_disqualify else -999.0
    lbs_full   = lbs["lbs_full"]  if full_hall  < hall_disqualify else -999.0

    best_gain = max(lbs_cross, lbs_full)
    if best_gain <= lbs_threshold:
        return "mono"

    if combined_weights and abs(lbs_cross - lbs_full) < 1e-4:
        # Tiebreaker: combined score
        def _combined(key: str) -> float:
            r  = lang_results[key]
            cw = combined_weights
            return (cw.get("f1", 0.45)  * r["f1"] / 100
                  + cw.get("rhr", 0.20) * r["rhr"] / 100
                  + cw.get("em", 0.15)  * r["em"] / 100
                  + cw.get("hall", 0.20) * (1 - r["hall"] / 100))
        return "cross" if _combined("cross") >= _combined("full") else "full"

    return "cross" if lbs_cross >= lbs_full else "full"


def aggregate_scope(raw: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Convert per-sample lists → mean percentages for a single scope.
    Input keys: em, f1, rhr, lmr, hall, rur, bertscore (all 0–1 floats).
    Output: same keys, values 0–100 (rounded to 2 dp).
    """
    return {
        metric: round(100 * (sum(vals) / len(vals)), 2)
        for metric, vals in raw.items()
        if vals
    }
