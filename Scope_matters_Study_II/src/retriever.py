"""
src/retriever.py
Dense retrieval with BGE-M3 or multilingual-E5-large.

Builds three scope indexes per query language:
  mono  — query language chunks only
  cross — query language + English chunks (English via FAISS)
  full  — all language chunks (numpy dot-product)

Reranking is applied for select languages using BGE-M3's
ColBERT+sparse+dense combined score.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

try:
    import faiss
    _HAVE_FAISS = True
except ImportError:
    _HAVE_FAISS = False

logging.getLogger("FlagEmbedding").setLevel(logging.ERROR)

ENCODE_BATCH = 512

# Languages where dense reranking improves RHR measurably
RERANK_LANGS_MLQA  = {"ar", "hi", "vi", "zh"}
RERANK_LANGS_XQUAD = {"th", "hi", "vi", "el"}


def load_retriever(model_id: str, use_fp16: bool = True):
    """
    Load a retriever. Supports BAAI/bge-m3 and intfloat/multilingual-e5-large.
    Returns the model object; call encode_texts() to get vectors.
    """
    if "bge-m3" in model_id.lower():
        from FlagEmbedding import BGEM3FlagModel
        model = BGEM3FlagModel(model_id, use_fp16=use_fp16)
        model._retriever_type = "bgem3"
    else:
        # multilingual-e5-large via sentence-transformers
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_id)
        model._retriever_type = "e5"
    return model


def encode_texts(
    model,
    texts: List[str],
    batch_size: int = ENCODE_BATCH,
    prefix: str = "",
) -> np.ndarray:
    """
    Encode a list of texts to L2-normalised float32 vectors.

    For E5 models, prepend the 'query: ' or 'passage: ' prefix as required.
    """
    if model._retriever_type == "bgem3":
        all_emb = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            out   = model.encode(
                batch,
                batch_size=batch_size,
                max_length=512,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            all_emb.append(np.array(out["dense_vecs"]))
    else:  # e5
        prefixed = [f"{prefix}{t}" for t in texts] if prefix else texts
        all_emb  = []
        for i in range(0, len(prefixed), batch_size):
            batch = prefixed[i : i + batch_size]
            emb   = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
            all_emb.append(emb)

    vecs = np.vstack(all_emb).astype("float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= np.maximum(norms, 1e-9)
    return vecs


def _build_faiss_index(vecs: np.ndarray) -> "faiss.IndexFlatIP":
    if not _HAVE_FAISS:
        raise ImportError("pip install faiss-cpu")
    dim = vecs.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(vecs)
    return idx


class ScopeIndex:
    """
    Holds pre-encoded embeddings for a single (lang, scope) pair and
    exposes a search() method that returns top-k chunk dicts.
    """

    def __init__(
        self,
        chunks: List[Dict],
        embeddings: np.ndarray,
        use_faiss: bool = False,
    ):
        self.chunks = chunks
        if use_faiss and _HAVE_FAISS and len(chunks) > 0:
            self._faiss = _build_faiss_index(embeddings)
            self._emb   = None
        else:
            self._faiss = None
            self._emb   = embeddings

    def search(self, q_vec: np.ndarray, top_k: int) -> List[Dict]:
        if not self.chunks:
            return []
        k = min(top_k, len(self.chunks))
        if self._faiss is not None:
            q = q_vec.reshape(1, -1)
            scores, idx = self._faiss.search(q, k)
            return [self.chunks[i] for i in idx[0] if i >= 0]
        sims    = (self._emb @ q_vec)
        top_idx = np.argsort(sims)[::-1][:k]
        return [self.chunks[i] for i in top_idx]


def build_scope_indexes(
    store: Dict[str, List[Dict]],
    model,
    query_lang: str,
    all_langs: List[str],
) -> Dict[str, ScopeIndex]:
    """
    Encode all chunks and build mono / cross / full indexes for query_lang.

    The English FAISS index is shared across cross-scope calls for all
    non-English query languages.
    """
    # Encode per language (cached externally if calling in a loop)
    lang_emb: Dict[str, np.ndarray] = {}
    prefix = "passage: " if model._retriever_type == "e5" else ""
    for lang in all_langs:
        texts = [c["text"] for c in store[lang]]
        if texts:
            lang_emb[lang] = encode_texts(model, texts, prefix=prefix)
        else:
            lang_emb[lang] = np.empty((0, 1024), dtype="float32")

    # Mono
    mono_idx = ScopeIndex(store[query_lang], lang_emb[query_lang])

    # Cross: query_lang numpy + English FAISS (separate objects)
    en_chunks = store.get("en", [])
    en_emb    = lang_emb.get("en", np.empty((0, 1), dtype="float32"))
    if query_lang == "en" or not en_chunks:
        cross_idx = mono_idx  # no separate EN corpus
        en_faiss  = None
    else:
        cross_idx = ScopeIndex(store[query_lang], lang_emb[query_lang])
        en_faiss  = ScopeIndex(en_chunks, en_emb, use_faiss=True)

    # Full: all languages concatenated
    full_chunks = [c for lang in all_langs for c in store[lang]]
    full_emb    = np.vstack([
        lang_emb[lang] for lang in all_langs if len(lang_emb[lang]) > 0
    ])
    full_idx = ScopeIndex(full_chunks, full_emb)

    return {
        "mono":  mono_idx,
        "cross": cross_idx,
        "full":  full_idx,
        "_en_faiss": en_faiss,
    }


def retrieve(
    question: str,
    indexes: Dict,
    model,
    lang: str,
    top_k: int,
    keep_k: int | None = None,
    rerank_langs: set | None = None,
) -> Dict[str, List[str]]:
    """
    Retrieve top-k chunks for each scope. Returns {scope: [chunk_text, ...]}.

    Cross scope: top (keep_k-3) from query_lang + top 3 from English FAISS.
    Reranking applied for languages in rerank_langs (BGE-M3 only).
    """
    keep_k = keep_k or top_k
    rerank_langs = rerank_langs or set()

    prefix = "query: " if model._retriever_type == "e5" else ""
    q_vec  = encode_texts(model, [question], prefix=prefix)[0]

    results: Dict[str, List[str]] = {}

    for scope in ("mono", "full"):
        idx      = indexes[scope]
        chunks   = idx.search(q_vec, top_k)
        texts    = _maybe_rerank(question, chunks, model, keep_k, lang, rerank_langs)
        results[scope] = texts

    # Cross scope
    en_faiss: ScopeIndex | None = indexes.get("_en_faiss")
    if en_faiss is not None:
        lang_keep = keep_k - 3
        lang_idx  = indexes["cross"]
        lang_chunks = lang_idx.search(q_vec, max(lang_keep, 1))
        en_chunks   = en_faiss.search(q_vec, 3)
        merged = lang_chunks[:lang_keep] + en_chunks
        seen, deduped = set(), []
        for c in merged:
            key = c["text"][:80]
            if key not in seen:
                seen.add(key)
                deduped.append(c)
        results["cross"] = [c["text"] for c in deduped[:keep_k]]
    else:
        results["cross"] = results["mono"]

    return results


def _maybe_rerank(
    question: str,
    chunks: List[Dict],
    model,
    keep_k: int,
    lang: str,
    rerank_langs: set,
) -> List[str]:
    if lang not in rerank_langs or model._retriever_type != "bgem3":
        return [c["text"] for c in chunks[:keep_k]]
    try:
        pairs  = [[question, c["text"]] for c in chunks]
        result = model.compute_score(pairs, max_passage_length=512)
        if isinstance(result, dict):
            scores = result.get("colbert+sparse+dense", list(result.values())[0])
        else:
            scores = result
        ranked = sorted(zip(scores, [c["text"] for c in chunks]),
                        key=lambda x: x[0], reverse=True)
        return [t for _, t in ranked[:keep_k]]
    except Exception:
        return [c["text"] for c in chunks[:keep_k]]
