"""
src/generator.py
Answer generation with four instruction-tuned multilingual models:
  aya    — CohereForAI/aya-23-8B
  qwen   — Qwen/Qwen2.5-7B-Instruct
  llama  — meta-llama/Meta-Llama-3-8B-Instruct
  eurollm— utter-project/EuroLLM-9B-Instruct

All models use the same extractive prompting strategy and identical
decoding settings (greedy, repetition_penalty=1.3) for comparability.
Language-specific prompts are used for Zh and Hi to improve script fidelity.
"""

import re
import os
import unicodedata
from typing import Dict, List

LANG_META: Dict[str, Dict] = {
    "en": {"name": "English",    "native": "English",    "script": "Latin"},
    "de": {"name": "German",     "native": "Deutsch",    "script": "Latin"},
    "es": {"name": "Spanish",    "native": "Español",    "script": "Latin"},
    "el": {"name": "Greek",      "native": "Ελληνικά",   "script": "Greek"},
    "ru": {"name": "Russian",    "native": "Русский",    "script": "Cyrillic"},
    "tr": {"name": "Turkish",    "native": "Türkçe",     "script": "Latin"},
    "ar": {"name": "Arabic",     "native": "العربية",    "script": "Arabic"},
    "vi": {"name": "Vietnamese", "native": "Tiếng Việt", "script": "Latin"},
    "th": {"name": "Thai",       "native": "ภาษาไทย",   "script": "Thai"},
    "zh": {"name": "Chinese",    "native": "中文",        "script": "Chinese"},
    "hi": {"name": "Hindi",      "native": "हिन्दी",     "script": "Devanagari"},
}

LANG_MAX_ANS_TOKENS: Dict[str, int] = {
    "zh": 8, "th": 8, "hi": 12, "ar": 12,
    "vi": 12, "de": 15, "es": 15, "en": 15,
    "ru": 15, "el": 12, "tr": 15,
}

CHAR_LEVEL_LANGS = {"zh", "th"}

_SCRIPT_INSTRUCTIONS: Dict[str, str] = {
    "ru": "ВАЖНО: Отвечайте ТОЛЬКО на русском языке кириллицей.\n",
    "el": (
        "ΣΗΜΑΝΤΙΚΟ: Απαντήστε ΜΟΝΟ στα ελληνικά με ελληνικό αλφάβητο.\n"
        "WARNING: Answer ONLY in Greek script.\n"
    ),
    "ar": "مهم: أجب باللغة العربية فقط.\n",
    "vi": "Quan trọng: Chỉ trả lời bằng tiếng Việt.\n",
    "de": "Wichtig: Antworten Sie NUR auf Deutsch.\n",
    "es": "Importante: Responda SOLO en español.\n",
    "th": (
        "สำคัญ: ตอบเป็นภาษาไทยเท่านั้น ห้ามใช้ภาษาอังกฤษ\n"
        "Maximum 5 Thai words.\n"
    ),
}

GENERIC_ZH = {
    "是的", "是", "对", "对的", "没有", "不是", "有", "无",
    "不", "可以", "不可以", "正确", "错误", "yes", "no",
    "是。", "对。", "是的。", "对的。",
}
GENERIC_HI = {
    "हां", "हाँ", "नहीं", "हा", "ना", "जी", "जी हां", "जी नहीं",
    "हां।", "नहीं।", "सही", "गलत", "हाँ।",
}


# ── Prompt builders ─────────────────────────────────────────────────────────

def build_prompt(question: str, chunks: List[str], lang: str) -> str:
    context = "\n\n--- chunk ---\n\n".join(chunks)[:3000]

    if lang == "zh":
        return (
            f"请阅读以下段落，从中找出问题的答案。\n"
            f"只输出答案本身，不要解释，不要完整句子，最多输出5个字。\n"
            f"答案必须直接来自段落原文。\n\n"
            f"### 段落\n{context}\n\n"
            f"### 问题\n{question}\n\n"
            f"### 答案（仅限原文中的词语，最多5个字）\n"
        )

    if lang == "hi":
        return (
            f"नीचे दिए गए अनुच्छेदों को पढ़ें और प्रश्न का उत्तर दें।\n"
            f"केवल उत्तर लिखें — कोई व्याख्या नहीं, कोई पूरा वाक्य नहीं।\n"
            f"उत्तर अनुच्छेद से सीधे लिया हुआ होना चाहिए। अधिकतम 5 शब्द।\n\n"
            f"### अनुच्छेद\n{context}\n\n"
            f"### प्रश्न\n{question}\n\n"
            f"### उत्तर (केवल मूल शब्द, अधिकतम 5 शब्द)\n"
        )

    m            = LANG_META.get(lang, LANG_META["en"])
    script_instr = _SCRIPT_INSTRUCTIONS.get(lang, "")
    return (
        f"You are a multilingual extractive question-answering assistant.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"{script_instr}"
        f"INSTRUCTIONS:\n"
        f"1. Answer using ONLY information from the context above.\n"
        f"2. Answer ONLY in {m['name']} ({m['native']}) using {m['script']} script.\n"
        f"3. Copy the EXACT answer phrase from the context — "
        f"no explanation, no full sentences.\n"
        f"4. If the answer is not in the context, reply: unanswerable\n\n"
        f"ANSWER in {m['name']}:"
    )


# ── Post-processing ──────────────────────────────────────────────────────────

def _clean_answer(raw: str, lang: str) -> str:
    answer = re.sub(
        r"^(the answer is|answer:|answer is|it is|the answer)\s*[:\-]?\s*",
        "", raw, flags=re.I
    ).strip()

    if lang == "hi":
        cleaned = []
        for ch in answer:
            cp = ord(ch)
            is_dev = 0x0900 <= cp <= 0x097F
            if is_dev or ch.isspace() or ch.isdigit() or ch in "।॥,.":
                cleaned.append(ch)
        answer = "".join(cleaned).strip()
    else:
        cleaned = []
        for ch in answer:
            cat = unicodedata.category(ch)
            if cat.startswith("C") and ch not in (" ", "\n"):
                continue
            if 0x2500 <= ord(ch) <= 0x259F:
                continue
            cleaned.append(ch)
        answer = "".join(cleaned).strip()

    # Keep first sentence only
    answer = answer.split(".")[0].split("\n")[0].strip()

    # Reject generic non-answers
    if lang == "zh" and answer in GENERIC_ZH:
        return ""
    if lang == "hi" and answer in GENERIC_HI:
        return ""

    return answer if len(answer) > 1 else ""


def _truncate(answer: str, lang: str, max_tokens: Dict[str, int]) -> str:
    max_tok = max_tokens.get(lang, 15)
    if lang in CHAR_LEVEL_LANGS:
        chars = [c for c in answer if not c.isspace()]
        return "".join(chars[:max_tok])
    tokens = answer.split()
    return " ".join(tokens[:max_tok]) if len(tokens) > max_tok else answer


# ── Model loader ─────────────────────────────────────────────────────────────

_LOADED: Dict[str, tuple] = {}  # model_id → (tokenizer, model)


def load_generator(model_id: str, torch_dtype_str: str = "float16",
                   load_in_4bit: bool = False):
    """
    Load tokenizer + model. Cached globally; subsequent calls are free.
    """
    if model_id in _LOADED:
        return _LOADED[model_id]

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        token=os.environ.get("HF_TOKEN"),
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    dtype = torch.float16 if torch_dtype_str == "float16" else torch.bfloat16

    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb,
                device_map="auto",
                token=os.environ.get("HF_TOKEN"),
            )
        except ImportError:
            load_in_4bit = False

    if not load_in_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            token=os.environ.get("HF_TOKEN"),
        )

    model.eval()
    _LOADED[model_id] = (tok, model)
    return tok, model


def unload_generator(model_id: str) -> None:
    """Release a loaded generator to free VRAM before loading the next."""
    import torch
    if model_id in _LOADED:
        tok, model = _LOADED.pop(model_id)
        del model, tok
        torch.cuda.empty_cache()


# ── Inference ────────────────────────────────────────────────────────────────

def generate_answer(
    question: str,
    chunks: List[str],
    lang: str,
    tokenizer,
    model,
    max_new_tokens: int = 50,
    repetition_penalty: float = 1.3,
    max_answer_tokens: Dict[str, int] | None = None,
) -> str:
    """
    Generate an extractive answer using the loaded model.

    Falls back to the first sentence of the first chunk on any error.
    """
    import torch

    if max_answer_tokens is None:
        max_answer_tokens = LANG_MAX_ANS_TOKENS

    prompt = build_prompt(question, chunks, lang)

    # Apply chat template if available (Qwen, LLaMA, EuroLLM)
    if getattr(tokenizer, "chat_template", None):
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    inputs    = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=3800
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]

    try:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
            )
        raw    = tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()
        answer = _clean_answer(raw, lang)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        answer = ""

    if not answer:
        first = chunks[0] if chunks else ""
        sents = re.split(r"(?<=[.!?。!?])\s*", first)
        return sents[0].strip() if sents else first

    return _truncate(answer, lang, max_answer_tokens)
