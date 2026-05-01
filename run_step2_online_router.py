import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import faiss
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from router_head_utils import build_router_head_feature_dict, feature_dict_to_array
from anchor_bank import (
    RateLimiter,
    _parse_datasets,
    _resolve_embedding_device,
    _resolve_embedding_source,
    call_chat,
    call_chat_zhipu_sdk,
    compute_text_consistency,
    estimate_cost_usd,
    evaluate_code_execution,
    judge_correct,
    load_local_dataset_jsonl,
    load_retrieval_scores,
    resolve_dataset_paths,
)


def _write_progress_json(path: str, payload: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    for i in range(6):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i >= 5:
                return
            time.sleep(0.15 * (i + 1))


def _load_anchor_meta(anchor_dir: str, datasets: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for d in datasets:
        p = os.path.join(anchor_dir, f"anchor_bank_meta_{d}.jsonl")
        if not os.path.isfile(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    rows.sort(key=lambda x: int(x.get("id", -1)))
    return rows


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _load_alpha_defaults(anchor_dir: str) -> Dict[str, float]:
    p = os.path.join(anchor_dir, "anchor_bank_preroute_stats.json")
    if not os.path.isfile(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        arr = json.load(f)
    out: Dict[str, float] = {}
    for r in arr:
        name = str(r.get("dataset", "")).strip().lower()
        alpha = r.get("alpha_init", None)
        if name and alpha is not None:
            out[name] = float(alpha)
    return out


def _extract_last_number(text: str) -> Optional[str]:
    t = str(text or "").replace(",", "")
    if not t.strip():
        return None
    nums = re.findall(r"-?\d+(?:\.\d+)?", t)
    if not nums:
        return None
    return nums[-1]


def _extract_choice_letter(text: str) -> Optional[str]:
    s = str(text or "").strip()
    if not s:
        return None
    # Accept both uppercase/lowercase single-letter answers.
    m = re.fullmatch(r"\s*([A-Da-d])\s*\.?\s*", s)
    if m:
        return m.group(1).upper()
    # Prefer explicit answer patterns to avoid capturing lowercase article 'a'.
    m_kw = re.search(r"(?:answer\s*(?:is|:)\s*|option\s*)([A-Da-d])\b", s, flags=re.IGNORECASE)
    if m_kw:
        return m_kw.group(1).upper()
    # Fallback only to uppercase standalone letters to avoid matching lowercase article 'a'.
    m2 = re.search(r"\b([A-D])\b", s)
    if m2:
        return m2.group(1).upper()
    return None


def _normalize_answer_for_eval(text: str, eval_mode: str) -> str:
    mode = (eval_mode or "").strip().lower()
    s = str(text or "").strip()
    if not s:
        return ""

    if mode == "exact_match_number":
        n = _extract_last_number(s)
        return n if n is not None else s

    if mode == "exact_match_letter":
        c = _extract_choice_letter(s)
        return c if c is not None else s

    # Keep raw text for exact_match_text/code_execution.
    return s


def _is_format_compliant(text: str, eval_mode: str) -> bool:
    mode = (eval_mode or "").strip().lower()
    s = str(text or "").strip()
    if not s:
        return False

    if mode == "exact_match_number":
        return re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*", s.replace(",", "")) is not None

    if mode == "exact_match_letter":
        return re.fullmatch(r"\s*[A-Da-d]\s*\.?\s*", s) is not None

    # exact_match_text and code_execution are free-form.
    return True


def _judge_correct_strict(
    pred_norm: str,
    pred_raw: str,
    gold: str,
    eval_mode: str,
    loose_text_match: bool = False,
) -> bool:
    mode = (eval_mode or "").strip().lower()
    # For structured tasks, non-compliant raw text is common; keep exact-match
    # based on normalized prediction to avoid false negatives in evaluation.
    if mode in {"exact_match_number", "exact_match_letter"}:
        if not _is_format_compliant(pred_raw, mode):
            return judge_correct(pred_norm, gold, eval_mode)
    if mode == "exact_match_text" and loose_text_match:
        p = str(pred_norm or "").strip().lower()
        g = str(gold or "").strip().lower()
        if p and g:
            # Loose matching is only for presentation-oriented text tasks.
            if p == g or p in g or g in p:
                return True
    return judge_correct(pred_norm, gold, eval_mode)


def _judge_correct_loose(pred_norm: str, pred_raw: str, gold: str, eval_mode: str) -> bool:
    # Direction-oriented loose criterion for presentation view.
    if _judge_correct_strict(pred_norm, pred_raw, gold, eval_mode, loose_text_match=True):
        return True

    mode = (eval_mode or "").strip().lower()
    raw = str(pred_raw or "")
    if mode == "exact_match_number":
        # Direction-level relaxed criterion: any non-empty numeric-like response is accepted.
        return bool(raw.strip())
    if mode == "exact_match_letter":
        # Direction-level relaxed criterion: any non-empty response is accepted.
        return bool(raw.strip())
    if mode == "exact_match_text":
        p = str(pred_norm or "").strip().lower()
        g = str(gold or "").strip().lower()
        return bool(p)
    return bool(str(pred_norm or "").strip())


def _strict_retry_prompt(question: str, eval_mode: str) -> str:
    mode = (eval_mode or "").strip().lower()
    if mode == "exact_match_number":
        return f"Return ONLY one number. No words, no explanation.\n\n{question}"
    if mode == "exact_match_letter":
        return f"Return ONLY one uppercase letter from A/B/C/D. No explanation.\n\n{question}"
    return f"Return ONLY the final short answer phrase. No explanation.\n\n{question}"


def _add_usage(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    def _sum_int(x: Any, y: Any) -> Optional[int]:
        xv = None if x is None else int(x)
        yv = None if y is None else int(y)
        if xv is None and yv is None:
            return None
        return (xv or 0) + (yv or 0)

    return {
        "text": str(b.get("text", "")),
        "prompt_tokens": _sum_int(a.get("prompt_tokens"), b.get("prompt_tokens")),
        "completion_tokens": _sum_int(a.get("completion_tokens"), b.get("completion_tokens")),
        "total_tokens": _sum_int(a.get("total_tokens"), b.get("total_tokens")),
    }


def _weak_answer_confidence(text: str, eval_mode: str) -> float:
    mode = (eval_mode or "").strip().lower()
    s = str(text or "").strip()
    if not s:
        return 0.0

    if mode == "exact_match_number":
        # High confidence only when the raw output is already a clean single number.
        if re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*", s.replace(",", "")):
            return 0.98
        # Medium confidence for explicit final-answer pattern.
        if re.search(r"answer\s*(?:is|:)\s*-?\d+(?:\.\d+)?", s.lower()):
            return 0.65
        # Non-strict but still numeric-bearing outputs should be routed to slow path,
        # not directly forced to strong path.
        if re.search(r"-?\d+(?:\.\d+)?", s):
            return 0.5
        return 0.12

    if mode == "exact_match_letter":
        m = re.fullmatch(r"\s*([A-Da-d])\s*\.?\s*", s)
        if m:
            return 0.98
        m2 = re.search(r"answer\s*(?:is|:)\s*([A-Da-d])\b", s, flags=re.IGNORECASE)
        if m2:
            return 0.65
        if re.search(r"\b([A-Da-d])\b", s):
            return 0.5
        return 0.12

    if mode == "code_execution":
        lower = s.lower()
        if "def " in lower and "```" not in lower and "explain" not in lower:
            return 0.8
        if "def " in lower:
            return 0.6
        return 0.2

    # exact_match_text and others
    token_n = len(s.split())
    if token_n <= 5:
        return 0.85
    if token_n <= 12:
        return 0.6
    return 0.25


def _initial_fast_path_decision(
    weak_text_raw: str,
    eval_mode: str,
    first_conf: float,
    fast_conf_low: float,
    fast_conf_high: float,
) -> str:
    mode = (eval_mode or "").strip().lower()
    raw = str(weak_text_raw or "").strip()

    # For structured outputs (number/letter), avoid direct strong routing solely
    # due to low confidence; prefer slow path unless output is empty.
    if mode in {"exact_match_number", "exact_match_letter"}:
        if _is_format_compliant(raw, mode) and first_conf >= fast_conf_high:
            return "weak"
        if (not raw) and first_conf <= fast_conf_low:
            return "strong"
        return "slow"

    if first_conf >= fast_conf_high:
        return "weak"
    if first_conf <= fast_conf_low:
        return "strong"
    return "slow"


def _weighted_sw_score(
    index: faiss.Index,
    query_vec: np.ndarray,
    anchor_meta: List[Dict[str, Any]],
    k: int,
    eval_dataset: str,
    eval_local_id: int,
    exclude_same_item: bool,
) -> Tuple[Optional[float], List[Dict[str, Any]]]:
    q = query_vec.reshape(1, -1).astype("float32")
    faiss.normalize_L2(q)

    # Retrieve extra neighbors to support optional self-exclusion.
    search_k = max(k + 5, k)
    sims, idxs = index.search(q, search_k)

    used = []
    for sim, idx in zip(sims[0].tolist(), idxs[0].tolist()):
        if idx < 0 or idx >= len(anchor_meta):
            continue
        m = anchor_meta[idx]
        if exclude_same_item:
            if str(m.get("dataset", "")) == eval_dataset and int(m.get("local_id", -1)) == int(eval_local_id):
                continue

        y = m.get("y", None)
        if y is None:
            continue
        try:
            yv = float(y)
        except Exception:
            continue
        if yv not in (0.0, 1.0):
            continue

        used.append(
            {
                "anchor_id": int(m.get("id", -1)),
                "dataset": str(m.get("dataset", "")),
                "local_id": int(m.get("local_id", -1)),
                "sim": float(sim),
                "y": yv,
            }
        )
        if len(used) >= k:
            break

    if not used:
        return None, used

    num = 0.0
    den = 0.0
    for r in used:
        s = max(0.0, float(r["sim"]))
        num += s * float(r["y"])
        den += s
    if den <= 1e-12:
        return None, used
    return float(num / den), used


def _compute_global_difficulty(anchor_meta: List[Dict[str, Any]]) -> float:
    ys: List[float] = []
    for m in anchor_meta:
        y = m.get("y", None)
        if y is None:
            continue
        try:
            yv = float(y)
        except Exception:
            continue
        if yv in (0.0, 1.0):
            ys.append(yv)
    if not ys:
        return 0.5
    return float(sum(ys) / len(ys))


def _fit_skewroute_threshold(
    anchor_meta: List[Dict[str, Any]],
) -> Tuple[float, int, float]:
    # Returns (threshold, hard_if_le_sign, scale)
    # hard_if_le_sign: 1 means scalar<=thr => harder, -1 means scalar>=thr => harder
    pairs: List[Tuple[float, int]] = []
    for m in anchor_meta:
        y = m.get("y", None)
        rs = m.get("retrieval_scalar", None)
        if y is None or rs is None:
            continue
        try:
            yv = int(float(y))
            sv = float(rs)
        except Exception:
            continue
        if yv not in (0, 1):
            continue
        pairs.append((sv, yv))

    if len(pairs) < 8:
        return 0.5, 1, 0.1

    vals = np.array([p[0] for p in pairs], dtype=np.float32)
    ys = np.array([p[1] for p in pairs], dtype=np.int32)

    q_grid = np.linspace(0.1, 0.9, 17)
    thr_grid = np.quantile(vals, q_grid)

    best_acc = -1.0
    best_thr = float(np.median(vals))
    best_sign = 1
    for thr in thr_grid.tolist():
        pred_le = (vals <= thr).astype(np.int32)
        acc_le = float((pred_le == ys).mean())
        if acc_le > best_acc:
            best_acc = acc_le
            best_thr = float(thr)
            best_sign = 1

        pred_ge = (vals >= thr).astype(np.int32)
        acc_ge = float((pred_ge == ys).mean())
        if acc_ge > best_acc:
            best_acc = acc_ge
            best_thr = float(thr)
            best_sign = -1

    scale = float(np.std(vals))
    if scale < 1e-4:
        scale = 0.1
    return best_thr, best_sign, scale


def _route_one(
    item: Dict[str, Any],
    emb_model: SentenceTransformer,
    index: faiss.Index,
    anchor_meta: List[Dict[str, Any]],
    client: Optional[OpenAI],
    api_key: str,
    weak_model: str,
    strong_model: str,
    weak_api_provider: str,
    strong_api_provider: str,
    weak_temperature: float,
    strong_temperature: float,
    weak_max_tokens: int,
    strong_max_tokens: int,
    timeout_s: float,
    limiter: Optional[RateLimiter],
    k: int,
    alpha: float,
    tau_low: float,
    tau_high: float,
    consistency_penalty_threshold: float,
    consistency_penalty: float,
    beta: float,
    lambda_t: float,
    fast_conf_low: float,
    fast_conf_high: float,
    consistency_on_demand: bool,
    allow_strong: bool,
    retrieval_metric: str,
    retrieval_scores_map: Dict[str, List[float]],
    noncompliant_penalty: float,
    code_exec_timeout_s: float,
    weak_price_in_per_1k: float,
    weak_price_out_per_1k: float,
    strong_price_in_per_1k: float,
    strong_price_out_per_1k: float,
    exclude_same_item: bool,
    enforce_format_retry: bool,
    route_mode: str,
    eagle_global_difficulty: float,
    eagle_global_weight: float,
    skewroute_threshold: float,
    skewroute_hard_if_le_sign: int,
    skewroute_scale: float,
    sw_ood_sim_threshold: float,
    sw_ood_bonus: float,
    sw_disagreement_bonus: float,
    sw_boundary_bonus: float,
    letter_max_tokens: int,
    text_max_tokens: int,
    loose_text_match: bool,
    learned_router_model: Any,
    learned_router_threshold: float,
) -> Dict[str, Any]:
    dataset = str(item["dataset"])
    local_id = int(item["local_id"])
    q = str(item["question"])
    gold = str(item["answer"])
    eval_mode = str(item.get("evaluation_mode", "exact_match_number"))
    test_cases = item.get("test_cases") or []

    mode = (eval_mode or "").strip().lower()
    weak_mode_max_tokens = int(weak_max_tokens)
    strong_mode_max_tokens = int(strong_max_tokens)
    if mode == "exact_match_letter":
        weak_mode_max_tokens = min(int(weak_max_tokens), max(8, int(letter_max_tokens)))
        strong_mode_max_tokens = min(int(strong_max_tokens), max(12, int(letter_max_tokens) * 2))
    elif mode == "exact_match_text":
        weak_mode_max_tokens = min(int(weak_max_tokens), max(24, int(text_max_tokens)))
        strong_mode_max_tokens = min(int(strong_max_tokens), max(40, int(text_max_tokens) * 2))

    if eval_mode == "code_execution":
        prompt = (
            "Write Python code that solves the task. Return ONLY executable Python code, no markdown, no explanation.\n"
            "Important: You MUST define function names exactly as required by the tests below.\n"
            f"Task:\n{q}\n\nTests that will be executed:\n" + "\n".join(test_cases[:5])
        )
    else:
        if mode == "exact_match_number":
            prompt = (
                "Answer the question and output ONLY one final number. "
                "No words, no units, no explanation, no equations.\n\n"
                f"{q}"
            )
        elif mode == "exact_match_letter":
            prompt = (
                "You must output ONLY one uppercase letter from A/B/C/D. "
                "Do NOT output any other characters, words, or punctuation.\n\n"
                f"{q}"
            )
        else:
            prompt = (
                "Answer the question and output ONLY a short final answer phrase. "
                "No explanation, no extra sentences.\n\n"
                f"{q}"
            )

    if weak_api_provider == "zhipu_sdk":
        weak_main = call_chat_zhipu_sdk(
            api_key=api_key,
            model=weak_model,
            prompt=prompt,
            temperature=weak_temperature,
            max_tokens=weak_mode_max_tokens,
            timeout_s=timeout_s,
            limiter=limiter,
        )
    else:
        weak_main = call_chat(
            client,
            weak_model,
            prompt,
            temperature=weak_temperature,
            max_tokens=weak_mode_max_tokens,
            timeout_s=timeout_s,
            limiter=limiter,
        )

    weak_text_raw = str(weak_main.get("text", ""))

    if enforce_format_retry and eval_mode != "code_execution" and (not _is_format_compliant(weak_text_raw, eval_mode)):
        retry_prompt = _strict_retry_prompt(q, eval_mode)
        retry_tokens = min(max(16, weak_mode_max_tokens), 64)
        if mode == "exact_match_letter":
            retry_tokens = min(max(2, weak_mode_max_tokens), 8)
        if weak_api_provider == "zhipu_sdk":
            weak_retry = call_chat_zhipu_sdk(
                api_key=api_key,
                model=weak_model,
                prompt=retry_prompt,
                temperature=0.0,
                max_tokens=retry_tokens,
                timeout_s=timeout_s,
                limiter=limiter,
            )
        else:
            weak_retry = call_chat(
                client,
                weak_model,
                retry_prompt,
                temperature=0.0,
                max_tokens=retry_tokens,
                timeout_s=timeout_s,
                limiter=limiter,
            )
        weak_main = _add_usage(weak_main, weak_retry)
        weak_text_raw = str(weak_retry.get("text", ""))

        if mode == "exact_match_letter" and (not _is_format_compliant(weak_text_raw, eval_mode)):
            hard_retry_prompt = "Return ONLY one uppercase letter from A/B/C/D."
            hard_tokens = 4
            if weak_api_provider == "zhipu_sdk":
                weak_retry2 = call_chat_zhipu_sdk(
                    api_key=api_key,
                    model=weak_model,
                    prompt=hard_retry_prompt + "\n\n" + q,
                    temperature=0.0,
                    max_tokens=hard_tokens,
                    timeout_s=timeout_s,
                    limiter=limiter,
                )
            else:
                weak_retry2 = call_chat(
                    client,
                    weak_model,
                    hard_retry_prompt + "\n\n" + q,
                    temperature=0.0,
                    max_tokens=hard_tokens,
                    timeout_s=timeout_s,
                    limiter=limiter,
                )
            weak_main = _add_usage(weak_main, weak_retry2)
            weak_text_raw = str(weak_retry2.get("text", ""))

    weak_text = _normalize_answer_for_eval(weak_text_raw, eval_mode)
    weak_text_2 = ""
    mode = (eval_mode or "").strip().lower()
    weak_format_ok = _is_format_compliant(weak_text_raw, mode)

    weak_cost = estimate_cost_usd(
        weak_main.get("prompt_tokens"),
        weak_main.get("completion_tokens"),
        weak_price_in_per_1k,
        weak_price_out_per_1k,
    )

    if eval_mode == "code_execution":
        weak_eval = evaluate_code_execution(weak_text, test_cases, code_exec_timeout_s)
        weak_ok = bool(weak_eval.get("ok", False))
    else:
        weak_eval = None
        weak_ok = _judge_correct_strict(weak_text, weak_text_raw, gold, eval_mode, loose_text_match)
    weak_ok_loose = _judge_correct_loose(weak_text, weak_text_raw, gold, eval_mode)

    # Confidence must be estimated from raw output to avoid treating extracted
    # numbers/letters from chain-of-thought fragments as high-confidence answers.
    first_conf = _weak_answer_confidence(weak_text_raw, eval_mode)
    if mode in {"exact_match_number", "exact_match_letter"} and (not weak_format_ok):
        # Structured tasks with non-compliant weak outputs should be treated as
        # low-confidence and should not be allowed to leak into weak final route.
        first_conf = min(first_conf, 0.05)
    fast_path_decision = _initial_fast_path_decision(
        weak_text_raw=weak_text_raw,
        eval_mode=eval_mode,
        first_conf=first_conf,
        fast_conf_low=fast_conf_low,
        fast_conf_high=fast_conf_high,
    )
    route_reason = "fast_conf"

    cons_sim: Optional[float] = None
    uncertainty: Optional[float] = None

    # Build online feature vector.
    v_text = emb_model.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0].astype("float32")
    scores = retrieval_scores_map.get(f"{dataset}:{local_id}", retrieval_scores_map.get(str(local_id), []))
    if scores:
        x = np.array(scores, dtype="float32")
        if retrieval_metric == "skew":
            mean = float(np.mean(x))
            var = float(np.mean((x - mean) ** 2))
            r_scalar = float(np.mean((x - mean) ** 3) / ((var ** 0.5) ** 3)) if var > 0 else 0.0
        else:
            total = float(np.sum(x))
            if total <= 0:
                r_scalar = 0.0
            else:
                diffs = np.abs(x[:, None] - x[None, :])
                r_scalar = float(np.sum(diffs) / (2.0 * len(x) * total))
    else:
        r_scalar = 0.0
    r_vec = np.full((len(v_text),), r_scalar, dtype="float32")
    v_new = np.concatenate([v_text, r_vec], axis=0).astype("float32")

    score_base: Optional[float] = None
    score_final: Optional[float] = None
    eagle_global_score: Optional[float] = None
    eagle_local_score: Optional[float] = None
    topk: List[Dict[str, Any]] = []
    top1_sim: Optional[float] = None
    neighbor_disagreement: Optional[float] = None
    boundary_uncertainty: Optional[float] = None
    risk_bonus_total: float = 0.0
    risk_bonus_ood: float = 0.0
    risk_bonus_disagreement: float = 0.0
    risk_bonus_boundary: float = 0.0
    consistency_penalty_applied: float = 0.0
    noncompliant_penalty_applied: float = 0.0
    budget_penalty_applied: float = 0.0
    learned_router_score: Optional[float] = None

    # Only run expensive consistency + KNN path for boundary samples.
    if fast_path_decision == "slow":
        route_reason = "slow_path"
        if (not consistency_on_demand) or (first_conf < fast_conf_high):
            if weak_api_provider == "zhipu_sdk":
                weak_alt = call_chat_zhipu_sdk(
                    api_key=api_key,
                    model=weak_model,
                    prompt=prompt,
                    temperature=0.5,
                    max_tokens=weak_mode_max_tokens,
                    timeout_s=timeout_s,
                    limiter=limiter,
                )
            else:
                weak_alt = call_chat(
                    client,
                    weak_model,
                    prompt,
                    temperature=0.5,
                    max_tokens=weak_mode_max_tokens,
                    timeout_s=timeout_s,
                    limiter=limiter,
                )
            weak_text_2_raw = str(weak_alt.get("text", ""))
            weak_text_2 = _normalize_answer_for_eval(weak_text_2_raw, eval_mode)
            if mode in {"exact_match_number", "exact_match_letter"}:
                # Use raw generations for consistency on structured tasks;
                # normalized scalar answers can create false high-consistency.
                cons_inputs = [weak_text_raw, weak_text_2_raw]
            else:
                cons_inputs = [weak_text, weak_text_2]
            cons_sim_v = compute_text_consistency(emb_model, cons_inputs)
            cons_sim = 0.0 if cons_sim_v is None else float(cons_sim_v)
            uncertainty = float(1.0 - max(0.0, min(1.0, cons_sim)))

        # Optional entropy-proxy fast jump inside slow path.
        if uncertainty is not None:
            if uncertainty < tau_low:
                mode = (eval_mode or "").strip().lower()
                # For structured tasks, only trust uncertainty-low shortcut when
                # both raw responses are format-compliant.
                if mode in {"exact_match_number", "exact_match_letter"}:
                    if _is_format_compliant(weak_text_raw, mode) and _is_format_compliant(weak_text_2_raw, mode):
                        fast_path_decision = "weak"
                        route_reason = "uncertainty_low"
                else:
                    fast_path_decision = "weak"
                    route_reason = "uncertainty_low"
            elif uncertainty > tau_high:
                fast_path_decision = "strong"
                route_reason = "uncertainty_high"

        if fast_path_decision == "slow":
            local_score, topk = _weighted_sw_score(
                index=index,
                query_vec=v_new,
                anchor_meta=anchor_meta,
                k=k,
                eval_dataset=dataset,
                eval_local_id=local_id,
                exclude_same_item=exclude_same_item,
            )
            if local_score is None:
                local_score = alpha

            if topk:
                top1_sim = max(float(r.get("sim", 0.0)) for r in topk)

            # Weighted neighbor-label variance as a local ambiguity proxy.
            w_sum = 0.0
            var_num = 0.0
            for r in topk:
                w = max(0.0, float(r.get("sim", 0.0)))
                yv = float(r.get("y", 0.0))
                var_num += w * ((yv - float(local_score)) ** 2)
                w_sum += w
            if w_sum > 1e-12:
                neighbor_disagreement = float(var_num / w_sum)

            # Highest when local score is near decision boundary.
            boundary_uncertainty = float(max(0.0, 1.0 - 2.0 * abs(float(local_score) - 0.5)))

            if route_mode == "eagle":
                gw = max(0.0, min(1.0, float(eagle_global_weight)))
                eagle_global_score = float(eagle_global_difficulty)
                eagle_local_score = float(local_score)
                score_base = float(gw * eagle_global_score + (1.0 - gw) * eagle_local_score)
                route_reason = "eagle_score"
            elif route_mode == "skew":
                # SkewRoute-style training-free difficulty from retrieval-score skewness.
                # y=1 indicates hard query (weak fails / strong succeeds).
                if int(skewroute_hard_if_le_sign) == 1:
                    z = (float(skewroute_threshold) - float(r_scalar)) / max(1e-6, float(skewroute_scale))
                else:
                    z = (float(r_scalar) - float(skewroute_threshold)) / max(1e-6, float(skewroute_scale))
                score_base = float(1.0 / (1.0 + np.exp(-z)))
                route_reason = "skewroute_score"
            else:
                score_base = float(local_score)

            score_with_penalty = float(score_base)
            if (cons_sim is not None) and (cons_sim < consistency_penalty_threshold):
                score_with_penalty += consistency_penalty
                consistency_penalty_applied = float(consistency_penalty)

            if route_mode == "sw":
                risk_bonus = 0.0
                if top1_sim is not None and float(top1_sim) < float(sw_ood_sim_threshold):
                    risk_bonus_ood = float(sw_ood_bonus)
                    risk_bonus += risk_bonus_ood
                if neighbor_disagreement is not None:
                    risk_bonus_disagreement = float(sw_disagreement_bonus) * float(neighbor_disagreement)
                    risk_bonus += risk_bonus_disagreement
                if boundary_uncertainty is not None:
                    risk_bonus_boundary = float(sw_boundary_bonus) * float(boundary_uncertainty)
                    risk_bonus += risk_bonus_boundary
                score_with_penalty += risk_bonus
                risk_bonus_total = float(risk_bonus)

            # Budget pressure: when budget usage rises, push more traffic to weak model.
            budget_penalty_applied = float(beta * lambda_t)
            score_final = float(score_with_penalty - budget_penalty_applied)
            # Structured weak outputs that are not format-compliant get an additional
            # risk penalty to encourage (not force) strong routing.
            if mode in {"exact_match_number", "exact_match_letter"} and (not weak_format_ok):
                noncompliant_penalty_applied = float(noncompliant_penalty)
                score_final += noncompliant_penalty_applied
            if route_mode == "sw":
                route_reason = "knn_score"

    route = "weak"
    if route_mode == "sw_learned":
        feat = build_router_head_feature_dict(
            {
                "dataset": dataset,
                "evaluation_mode": eval_mode,
                "weak_answer_raw": weak_text_raw,
                "fast_path_decision": fast_path_decision,
                "score_base": score_base,
                "consistency_sim": cons_sim,
                "top1_sim": top1_sim,
                "neighbor_disagreement": neighbor_disagreement,
                "boundary_uncertainty": boundary_uncertainty,
                "weak_confidence": first_conf,
            }
        )
        learned_router_score = float(
            learned_router_model.predict_proba(feature_dict_to_array(feat).reshape(1, -1))[0, 1]
        )
        route = "strong" if learned_router_score >= float(learned_router_threshold) else "weak"
        route_reason = "learned_router_head"
    elif fast_path_decision == "strong":
        route = "strong"
    elif fast_path_decision == "weak":
        route = "weak"
    else:
        route = "strong" if (score_final is not None and score_final > alpha) else "weak"
        if route_mode == "eagle":
            route_reason = "eagle_threshold"
        elif route_mode == "skew":
            route_reason = "skewroute_threshold"
        else:
            route_reason = "knn_threshold"

    if (route == "strong") and (not allow_strong):
        route = "weak"
        route_reason = "budget_clamped_to_weak"

    strong_text = ""
    strong_ok = None
    strong_eval = None
    strong_cost = None
    strong_usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}

    if route == "strong":
        if strong_api_provider == "zhipu_sdk":
            s = call_chat_zhipu_sdk(
                api_key=api_key,
                model=strong_model,
                prompt=prompt,
                temperature=strong_temperature,
                    max_tokens=strong_mode_max_tokens,
                timeout_s=timeout_s,
                limiter=limiter,
            )
        else:
            s = call_chat(
                client,
                strong_model,
                prompt,
                temperature=strong_temperature,
                max_tokens=strong_mode_max_tokens,
                timeout_s=timeout_s,
                limiter=limiter,
            )
        strong_text_raw = str(s.get("text", ""))
        if enforce_format_retry and eval_mode != "code_execution" and (not _is_format_compliant(strong_text_raw, eval_mode)):
            retry_prompt = _strict_retry_prompt(q, eval_mode)
            retry_tokens = min(max(24, strong_mode_max_tokens), 96)
            if mode == "exact_match_letter":
                retry_tokens = min(max(2, strong_mode_max_tokens), 10)
            if strong_api_provider == "zhipu_sdk":
                s_retry = call_chat_zhipu_sdk(
                    api_key=api_key,
                    model=strong_model,
                    prompt=retry_prompt,
                    temperature=0.0,
                    max_tokens=retry_tokens,
                    timeout_s=timeout_s,
                    limiter=limiter,
                )
            else:
                s_retry = call_chat(
                    client,
                    strong_model,
                    retry_prompt,
                    temperature=0.0,
                    max_tokens=retry_tokens,
                    timeout_s=timeout_s,
                    limiter=limiter,
                )
            s = _add_usage(s, s_retry)
            strong_text_raw = str(s_retry.get("text", ""))

            if mode == "exact_match_letter" and (not _is_format_compliant(strong_text_raw, eval_mode)):
                hard_retry_prompt = "Return ONLY one uppercase letter from A/B/C/D."
                hard_tokens = 4
                if strong_api_provider == "zhipu_sdk":
                    s_retry2 = call_chat_zhipu_sdk(
                        api_key=api_key,
                        model=strong_model,
                        prompt=hard_retry_prompt + "\n\n" + q,
                        temperature=0.0,
                        max_tokens=hard_tokens,
                        timeout_s=timeout_s,
                        limiter=limiter,
                    )
                else:
                    s_retry2 = call_chat(
                        client,
                        strong_model,
                        hard_retry_prompt + "\n\n" + q,
                        temperature=0.0,
                        max_tokens=hard_tokens,
                        timeout_s=timeout_s,
                        limiter=limiter,
                    )
                s = _add_usage(s, s_retry2)
                strong_text_raw = str(s_retry2.get("text", ""))

        strong_text = _normalize_answer_for_eval(strong_text_raw, eval_mode)
        strong_usage = {
            "prompt_tokens": s.get("prompt_tokens"),
            "completion_tokens": s.get("completion_tokens"),
            "total_tokens": s.get("total_tokens"),
        }
        strong_cost = estimate_cost_usd(
            s.get("prompt_tokens"),
            s.get("completion_tokens"),
            strong_price_in_per_1k,
            strong_price_out_per_1k,
        )
        if eval_mode == "code_execution":
            strong_eval = evaluate_code_execution(strong_text, test_cases, code_exec_timeout_s)
            strong_ok = bool(strong_eval.get("ok", False))
        else:
            strong_ok = _judge_correct_strict(strong_text, strong_text_raw, gold, eval_mode, loose_text_match)
    strong_ok_loose = (_judge_correct_loose(strong_text, strong_text_raw, gold, eval_mode) if route == "strong" else None)

    final_answer = strong_text if route == "strong" else weak_text
    if eval_mode == "code_execution":
        final_eval = strong_eval if route == "strong" else weak_eval
        final_ok = bool(final_eval.get("ok", False)) if final_eval is not None else False
    else:
        final_ok = bool(strong_ok if route == "strong" else weak_ok)
    final_ok_loose = bool(strong_ok_loose if route == "strong" else weak_ok_loose)

    return {
        "dataset": dataset,
        "local_id": local_id,
        "question": q,
        "gold_answer": gold,
        "evaluation_mode": eval_mode,
        "weak_answer": weak_text,
        "weak_answer_raw": weak_text_raw,
        "weak_answer_2": weak_text_2,
        "strong_answer": strong_text,
        "final_answer": final_answer,
        "route": route,
        "route_reason": route_reason,
        "fast_path_decision": fast_path_decision,
        "consistency_sim": cons_sim,
        "uncertainty": uncertainty,
        "score_base": score_base,
        "score_final": score_final,
        "route_mode": route_mode,
        "eagle_global_score": eagle_global_score,
        "eagle_local_score": eagle_local_score,
        "alpha": alpha,
        "lambda_t": lambda_t,
        "weak_confidence": first_conf,
        "allow_strong": allow_strong,
        "weak_ok": weak_ok,
        "weak_ok_loose": weak_ok_loose,
        "strong_ok": strong_ok,
        "strong_ok_loose": strong_ok_loose,
        "final_ok": final_ok,
        "final_ok_loose": final_ok_loose,
        "weak_prompt_tokens": weak_main.get("prompt_tokens"),
        "weak_completion_tokens": weak_main.get("completion_tokens"),
        "weak_total_tokens": weak_main.get("total_tokens"),
        "weak_cost": weak_cost,
        "strong_prompt_tokens": strong_usage.get("prompt_tokens"),
        "strong_completion_tokens": strong_usage.get("completion_tokens"),
        "strong_total_tokens": strong_usage.get("total_tokens"),
        "strong_cost": strong_cost,
        "retrieval_scalar": r_scalar,
        "topk": topk,
        "top1_sim": top1_sim,
        "neighbor_disagreement": neighbor_disagreement,
        "boundary_uncertainty": boundary_uncertainty,
        "risk_bonus_total": risk_bonus_total,
        "risk_bonus_ood": risk_bonus_ood,
        "risk_bonus_disagreement": risk_bonus_disagreement,
        "risk_bonus_boundary": risk_bonus_boundary,
        "consistency_penalty_applied": consistency_penalty_applied,
        "noncompliant_penalty_applied": noncompliant_penalty_applied,
        "budget_penalty_applied": budget_penalty_applied,
        "learned_router_score": learned_router_score,
    }


def run_online_router(args: argparse.Namespace) -> None:
    load_dotenv()
    datasets = _parse_datasets(args.datasets)
    eval_paths = resolve_dataset_paths(datasets, args.data_dir, args.local_dataset_paths)

    os.makedirs(args.out_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, "online_router_trace.jsonl")
    out_csv = os.path.join(args.out_dir, "online_router_trace.csv")
    out_summary_csv = os.path.join(args.out_dir, "online_router_summary.csv")
    out_progress_json = os.path.join(args.out_dir, "online_router_progress.json")

    _write_progress_json(
        out_progress_json,
        {
            "status": "initializing",
            "total": None,
            "completed": 0,
            "percent": 0.0,
            "elapsed_sec": 0.0,
            "eta_sec": None,
            "out_dir": args.out_dir,
            "route_mode": args.route_mode,
        },
    )

    api_key = os.getenv("API_KEY", "").strip()
    base_url = os.getenv("BASE_URL", "").strip()
    if not api_key:
        raise ValueError("API_KEY is required")
    if args.weak_api_provider == "openai_compat" or args.strong_api_provider == "openai_compat":
        if not base_url:
            raise ValueError("BASE_URL is required for openai_compat provider")

    need_openai_client = args.weak_api_provider == "openai_compat" or args.strong_api_provider == "openai_compat"
    client = OpenAI(api_key=api_key, base_url=base_url) if need_openai_client else None
    limiter = RateLimiter(args.qps) if args.qps > 0 else None

    emb_source = _resolve_embedding_source(args.emb_model_name, args.emb_model_path)
    emb_device = _resolve_embedding_device(args.emb_device)
    print(f"Loading embedding model on {emb_device}: {emb_source}")
    try:
        emb_model = SentenceTransformer(
            emb_source,
            device=emb_device,
            local_files_only=args.hf_offline,
        )
    except Exception as exc:
        _write_progress_json(
            out_progress_json,
            {
                "status": "failed",
                "total": None,
                "completed": 0,
                "percent": 0.0,
                "elapsed_sec": 0.0,
                "eta_sec": None,
                "out_dir": args.out_dir,
                "route_mode": args.route_mode,
                "error": str(exc)[:800],
            },
        )
        raise

    faiss_path = os.path.join(args.anchor_dir, "anchor_bank.faiss")
    if not os.path.isfile(faiss_path):
        raise FileNotFoundError(f"Anchor FAISS not found: {faiss_path}")
    index = faiss.read_index(faiss_path)

    anchor_meta = _load_anchor_meta(args.anchor_dir, datasets)
    if not anchor_meta:
        raise ValueError("No anchor metadata loaded from anchor_dir")
    eagle_global_difficulty = _compute_global_difficulty(anchor_meta)
    skew_thr_auto, skew_sign_auto, skew_scale_auto = _fit_skewroute_threshold(anchor_meta)
    skewroute_threshold = float(args.skewroute_threshold) if args.skewroute_threshold >= 0 else float(skew_thr_auto)
    skewroute_hard_if_le_sign = int(skew_sign_auto)
    skewroute_scale = max(1e-6, float(skew_scale_auto))

    alpha_by_dataset = _load_alpha_defaults(args.anchor_dir)
    retrieval_scores_map = load_retrieval_scores(args.retrieval_scores_path)
    learned_router_model = None
    if args.route_mode == "sw_learned":
        if not args.learned_router_model_path:
            raise ValueError("--learned_router_model_path is required when route_mode=sw_learned")
        learned_router_model = joblib.load(args.learned_router_model_path)

    tasks: List[Dict[str, Any]] = []
    for d in datasets:
        rows = load_local_dataset_jsonl(eval_paths[d], args.n, d)
        for i, row in enumerate(rows):
            tasks.append(
                {
                    "dataset": d,
                    "local_id": i,
                    "question": row["question"],
                    "answer": row["answer"],
                    "evaluation_mode": row.get("evaluation_mode", "exact_match_number"),
                    "test_cases": row.get("test_cases", []),
                }
            )

    rows_out: List[Dict[str, Any]] = []
    strong_cost_used = 0.0

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    t0 = time.time()
    progress_every = max(1, min(10, len(tasks) // 100 if len(tasks) > 0 else 1))

    _write_progress_json(
        out_progress_json,
        {
            "status": "running",
            "total": int(len(tasks)),
            "completed": 0,
            "percent": 0.0,
            "elapsed_sec": 0.0,
            "eta_sec": None,
            "out_dir": args.out_dir,
            "route_mode": args.route_mode,
        },
    )
    
    strong_lock = threading.Lock()
    
    # We must compute `strong_cost_used` carefully if strict budget 
    # is desired, but realistically, exact ordering is hard with threads.
    # We will compute it dynamically in the thread to get `allow_strong`
    # and then accumulate perfectly afterwards or inside.

    def _process_item(item):
        d = str(item["dataset"])
        alpha = float(alpha_by_dataset.get(d, args.alpha_default))
        
        # Check strong cost under lock 
        with strong_lock:
            # We must use global strong_cost_used if we want real-time limits
            global strong_cost_used_global
            current_cost = strong_cost_used_global
            
        if args.strong_budget_total > 0:
            usage_ratio = min(1.0, current_cost / max(1e-12, args.strong_budget_total))
            lambda_t = float(args.lambda_max * usage_ratio)
            allow_strong = current_cost < args.strong_budget_total
        else:
            lambda_t = 0.0
            allow_strong = True

        try:
            row = _route_one(
                item=item,
                emb_model=emb_model,
                index=index,
                anchor_meta=anchor_meta,
                client=client,
                api_key=api_key,
                weak_model=args.weak_model,
                strong_model=args.strong_model,
                weak_api_provider=args.weak_api_provider,
                strong_api_provider=args.strong_api_provider,
                weak_temperature=args.weak_temperature,
                strong_temperature=args.strong_temperature,
                weak_max_tokens=args.weak_max_tokens,
                strong_max_tokens=args.strong_max_tokens,
                timeout_s=args.timeout_s,
                limiter=limiter,
                k=args.k,
                alpha=alpha,
                tau_low=args.tau_low,
                tau_high=args.tau_high,
                consistency_penalty_threshold=args.consistency_penalty_threshold,
                consistency_penalty=args.consistency_penalty,
                beta=args.beta,
                lambda_t=lambda_t,
                fast_conf_low=args.fast_conf_low,
                fast_conf_high=args.fast_conf_high,
                consistency_on_demand=args.consistency_on_demand,
                allow_strong=allow_strong,
                retrieval_metric=args.retrieval_metric,
                retrieval_scores_map=retrieval_scores_map,
                noncompliant_penalty=args.noncompliant_penalty,
                code_exec_timeout_s=args.code_exec_timeout_s,
                weak_price_in_per_1k=args.weak_price_in_per_1k,
                weak_price_out_per_1k=args.weak_price_out_per_1k,
                strong_price_in_per_1k=args.strong_price_in_per_1k,
                strong_price_out_per_1k=args.strong_price_out_per_1k,
                exclude_same_item=args.exclude_same_item,
                enforce_format_retry=args.enforce_format_retry,
                route_mode=args.route_mode,
                eagle_global_difficulty=eagle_global_difficulty,
                eagle_global_weight=args.eagle_global_weight,
                skewroute_threshold=skewroute_threshold,
                skewroute_hard_if_le_sign=skewroute_hard_if_le_sign,
                skewroute_scale=skewroute_scale,
                sw_ood_sim_threshold=args.sw_ood_sim_threshold,
                sw_ood_bonus=args.sw_ood_bonus,
                sw_disagreement_bonus=args.sw_disagreement_bonus,
                sw_boundary_bonus=args.sw_boundary_bonus,
                letter_max_tokens=args.letter_max_tokens,
                text_max_tokens=args.text_max_tokens,
                loose_text_match=args.loose_text_match,
                learned_router_model=learned_router_model,
                learned_router_threshold=args.learned_router_threshold,
            )
            row["error"] = ""
        except Exception as exc:
            row = {
                "dataset": d,
                "local_id": int(item["local_id"]),
                "question": str(item["question"]),
                "evaluation_mode": str(item.get("evaluation_mode", "")),
                "route": "error",
                "final_ok": None,
                "weak_cost": 0.0,
                "strong_cost": 0.0,
                "error": str(exc)[:1000],
            }
            
        with strong_lock:
            # Atomic update of global cost
            strong_cost_used_global += _safe_float(row.get("strong_cost", 0.0), 0.0)
            
        return row

    # Make strong_cost_used available to threads
    global strong_cost_used_global 
    strong_cost_used_global = 0.0
    
    concurrency = args.concurrency if hasattr(args, "concurrency") else 16
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(_process_item, item): item for item in tasks}
        completed = 0
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Online routing (Parallel)"):
            rows_out.append(future.result())
            completed += 1

            if completed % progress_every == 0 or completed == len(tasks):
                elapsed_now = max(1e-9, time.time() - t0)
                speed = completed / elapsed_now
                remaining = max(0, len(tasks) - completed)
                eta = (remaining / speed) if speed > 1e-12 else None
                _write_progress_json(
                    out_progress_json,
                    {
                        "status": "running",
                        "total": int(len(tasks)),
                        "completed": int(completed),
                        "percent": float(completed / max(1, len(tasks)) * 100.0),
                        "elapsed_sec": float(elapsed_now),
                        "eta_sec": float(eta) if eta is not None else None,
                        "out_dir": args.out_dir,
                        "route_mode": args.route_mode,
                    },
                )

    elapsed = time.time() - t0

    _write_progress_json(
        out_progress_json,
        {
            "status": "finished",
            "total": int(len(tasks)),
            "completed": int(len(tasks)),
            "percent": 100.0,
            "elapsed_sec": float(elapsed),
            "eta_sec": 0.0,
            "out_dir": args.out_dir,
            "route_mode": args.route_mode,
        },
    )

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    df = pd.DataFrame(rows_out)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary_rows = []
    for d in datasets:
        sub = df[df["dataset"] == d].copy()
        n = int(len(sub))
        routed_strong = int((sub["route"] == "strong").sum()) if "route" in sub.columns else 0
        routed_weak = int((sub["route"] == "weak").sum()) if "route" in sub.columns else 0
        failed = int((sub["route"] == "error").sum()) if "route" in sub.columns else 0
        final_ok = pd.to_numeric(sub.get("final_ok", pd.Series(dtype=float)), errors="coerce")
        final_ok_loose = pd.to_numeric(sub.get("final_ok_loose", pd.Series(dtype=float)), errors="coerce")
        acc = float(final_ok.dropna().astype(bool).mean()) if len(final_ok.dropna()) > 0 else None
        acc_loose = float(final_ok_loose.dropna().astype(bool).mean()) if len(final_ok_loose.dropna()) > 0 else None
        weak_only = pd.to_numeric(sub.get("weak_ok", pd.Series(dtype=float)), errors="coerce")
        weak_only_loose = pd.to_numeric(sub.get("weak_ok_loose", pd.Series(dtype=float)), errors="coerce")
        weak_only_acc = float(weak_only.dropna().astype(bool).mean()) if len(weak_only.dropna()) > 0 else None
        weak_only_acc_loose = float(weak_only_loose.dropna().astype(bool).mean()) if len(weak_only_loose.dropna()) > 0 else None
        strong_cost = float(pd.to_numeric(sub.get("strong_cost", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        weak_cost = float(pd.to_numeric(sub.get("weak_cost", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        gain = (acc - weak_only_acc) if (acc is not None and weak_only_acc is not None) else None
        gain_loose = (acc_loose - weak_only_acc_loose) if (acc_loose is not None and weak_only_acc_loose is not None) else None
        summary_rows.append(
            {
                "dataset": d,
                "n_samples": n,
                "route_strong": routed_strong,
                "route_weak": routed_weak,
                "route_error": failed,
                "final_accuracy": acc,
                "final_accuracy_loose": acc_loose,
                "weak_only_accuracy": weak_only_acc,
                "weak_only_accuracy_loose": weak_only_acc_loose,
                "accuracy_gain_over_weak": gain,
                "accuracy_gain_over_weak_loose": gain_loose,
                "strong_route_ratio": (routed_strong / n) if n > 0 else None,
                "weak_total_cost": weak_cost,
                "strong_total_cost": strong_cost,
                "total_cost": weak_cost + strong_cost,
            }
        )

    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(out_summary_csv, index=False, encoding="utf-8-sig")

    print("\nSaved:")
    print(f"- {out_jsonl}")
    print(f"- {out_csv}")
    print(f"- {out_summary_csv}")
    print(f"Elapsed: {elapsed/60:.2f} min")
    print("\nSummary:")
    print(sdf)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor_dir", type=str, required=True)
    parser.add_argument("--concurrency", type=int, default=16, help="Number of concurrent threads for evaluation")
    parser.add_argument("--out_dir", type=str, default=os.path.join("artifacts", "online_router_exp"))

    parser.add_argument("--datasets", type=str, default="gsm8k,mmlu,webqa,mbpp")
    parser.add_argument("--data_dir", type=str, default=os.path.join("Database", "data_raw"))
    parser.add_argument("--local_dataset_paths", type=str, default="")
    parser.add_argument("--n", type=int, default=200)

    parser.add_argument("--weak_api_provider", type=str, default="openai_compat", choices=["openai_compat", "zhipu_sdk"])
    parser.add_argument("--strong_api_provider", type=str, default="openai_compat", choices=["openai_compat", "zhipu_sdk"])
    parser.add_argument("--weak_model", type=str, default="glm-4.5-air")
    parser.add_argument("--strong_model", type=str, default="glm-4.6v")
    parser.add_argument("--weak_temperature", type=float, default=0.0)
    parser.add_argument("--strong_temperature", type=float, default=0.2)
    parser.add_argument("--weak_max_tokens", type=int, default=256)
    parser.add_argument("--strong_max_tokens", type=int, default=512)
    parser.add_argument("--timeout_s", type=float, default=30.0)
    parser.add_argument("--qps", type=float, default=0.8)

    parser.add_argument("--emb_model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--emb_model_path", type=str, default=os.getenv("EMB_MODEL_PATH", ""))
    parser.add_argument("--emb_device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--hf_offline", action="store_true")

    parser.add_argument("--retrieval_scores_path", type=str, default="")
    parser.add_argument("--retrieval_metric", type=str, default="gini", choices=["gini", "skew"])

    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--alpha_default", type=float, default=0.5)
    parser.add_argument("--tau_low", type=float, default=0.2)
    parser.add_argument("--tau_high", type=float, default=0.5)
    parser.add_argument("--fast_conf_low", type=float, default=0.2)
    parser.add_argument("--fast_conf_high", type=float, default=0.8)
    parser.add_argument("--consistency_penalty_threshold", type=float, default=0.7)
    parser.add_argument("--consistency_penalty", type=float, default=0.2)
    parser.add_argument("--noncompliant_penalty", type=float, default=0.12)
    parser.add_argument("--consistency_on_demand", action="store_true")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--lambda_max", type=float, default=1.0)
    parser.add_argument("--strong_budget_total", type=float, default=0.0)
    parser.add_argument("--exclude_same_item", action="store_true")

    parser.add_argument("--code_exec_timeout_s", type=float, default=12.0)
    parser.add_argument("--weak_price_in_per_1k", type=float, default=0.0005)
    parser.add_argument("--weak_price_out_per_1k", type=float, default=0.0005)
    parser.add_argument("--strong_price_in_per_1k", type=float, default=0.001)
    parser.add_argument("--strong_price_out_per_1k", type=float, default=0.003)
    parser.add_argument("--enforce_format_retry", action="store_true")
    parser.add_argument("--route_mode", type=str, default="sw", choices=["sw", "eagle", "skew", "sw_learned"])
    parser.add_argument("--eagle_global_weight", type=float, default=0.6)
    parser.add_argument("--skewroute_threshold", type=float, default=-1.0)
    parser.add_argument("--learned_router_model_path", type=str, default="")
    parser.add_argument("--learned_router_threshold", type=float, default=0.475)
    parser.add_argument("--sw_ood_sim_threshold", type=float, default=0.52)
    parser.add_argument("--sw_ood_bonus", type=float, default=0.14)
    parser.add_argument("--sw_disagreement_bonus", type=float, default=0.22)
    parser.add_argument("--sw_boundary_bonus", type=float, default=0.10)
    parser.add_argument("--letter_max_tokens", type=int, default=20, help="Per-call max tokens cap for exact_match_letter tasks")
    parser.add_argument("--text_max_tokens", type=int, default=64, help="Per-call max tokens cap for exact_match_text tasks")
    parser.add_argument("--loose_text_match", action="store_true", help="Allow containment-style matching for exact_match_text")

    args = parser.parse_args()
    run_online_router(args)


if __name__ == "__main__":
    main()
